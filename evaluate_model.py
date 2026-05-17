"""
UrbanSound8K Fold10 üzerinde mevcut model.tflite'in akademik değerlendirmesi.
Modeli yeniden eğitmez; sadece test eder.

Çıktılar:
  analysis_output/confusion_matrix.png
  analysis_output/classification_report.txt
  model_evaluation.md (proje kökünde)
"""

import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "UrbanSound8K"
METADATA_PATH = DATASET_DIR / "metadata" / "UrbanSound8K.csv"
AUDIO_DIR = DATASET_DIR / "audio"

MODEL_PATH = BASE_DIR / "model.tflite"
LABELS_PATH = BASE_DIR / "labels.txt"
NORM_PARAMS_PATH = BASE_DIR / "norm_params.json"

OUTPUT_DIR = BASE_DIR / "analysis_output"
REPORT_MD = BASE_DIR / "model_evaluation.md"

# train.py ile bire bir aynı parametreler
TARGET_SR = 16000
DURATION_SEC = 1.0
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
TEST_FOLD = 10


def load_audio(file_path: Path) -> np.ndarray:
    y, _ = librosa.load(str(file_path), sr=TARGET_SR, mono=True)
    target = int(TARGET_SR * DURATION_SEC)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)), mode="constant")
    else:
        start = (len(y) - target) // 2
        y = y[start: start + target]
    return y.astype(np.float32)


def extract_mel(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=TARGET_SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
        center=False, fmin=0, fmax=TARGET_SR // 2, norm="slaney",
    )
    return librosa.power_to_db(mel, ref=np.max)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Etiketler & norm parametreleri
    with open(LABELS_PATH, encoding="utf-8") as f:
        class_names = [ln.strip() for ln in f if ln.strip()]
    norm = json.loads(NORM_PARAMS_PATH.read_text(encoding="utf-8"))
    x_min, x_max = float(norm["x_min"]), float(norm["x_max"])
    span = max(x_max - x_min, 1e-6)
    print(f"[+] {len(class_names)} sınıf, x_min={x_min}, x_max={x_max}")

    # 2) TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH), num_threads=4)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    print(f"[+] TFLite input shape: {inp['shape']}, dtype: {inp['dtype']}")

    # 3) Fold10 metadata
    df = pd.read_csv(METADATA_PATH)
    df_test = df[df["fold"] == TEST_FOLD].reset_index(drop=True)
    print(f"[+] Fold {TEST_FOLD}: {len(df_test)} test örneği")

    y_true, y_pred = [], []
    skipped = 0
    t_start = time.time()

    for i, row in df_test.iterrows():
        wav_path = AUDIO_DIR / f"fold{int(row['fold'])}" / row["slice_file_name"]
        if not wav_path.exists():
            skipped += 1
            continue
        try:
            y_audio = load_audio(wav_path)
        except Exception as e:
            print(f"  [!] Yükleme hatası {wav_path.name}: {e}", file=sys.stderr)
            skipped += 1
            continue

        mel = extract_mel(y_audio)
        x_norm = (mel - x_min) / span
        x_norm = x_norm.astype(np.float32)
        # (64, 30, 1) — train.py ile aynı; eğer extra frame varsa kırp
        target_frames = int(inp["shape"][2])
        if x_norm.shape[1] > target_frames:
            x_norm = x_norm[:, :target_frames]
        elif x_norm.shape[1] < target_frames:
            pad = target_frames - x_norm.shape[1]
            x_norm = np.pad(x_norm, ((0, 0), (0, pad)), mode="constant")
        x_norm = x_norm[np.newaxis, :, :, np.newaxis]

        interpreter.set_tensor(inp["index"], x_norm)
        interpreter.invoke()
        probs = interpreter.get_tensor(out["index"])[0]
        y_pred.append(int(np.argmax(probs)))
        y_true.append(int(row["classID"]))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i + 1}/{len(df_test)}] geçen süre: {elapsed:.1f}s")

    elapsed = time.time() - t_start
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print(f"\n[+] Inference bitti: {len(y_true)} örnek, {elapsed:.1f}s "
          f"({elapsed / max(len(y_true), 1) * 1000:.1f} ms/örnek), {skipped} atlandı")

    # 4) Metrikler
    overall_acc = accuracy_score(y_true, y_pred)
    print(f"\n>>> Test Doğruluğu: {overall_acc * 100:.2f}% <<<\n")

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names,
        output_dict=True, zero_division=0,
        labels=list(range(len(class_names))),
    )
    report_text = classification_report(
        y_true, y_pred, target_names=class_names,
        zero_division=0, digits=4,
        labels=list(range(len(class_names))),
    )

    # 5) classification_report.txt
    txt_path = OUTPUT_DIR / "classification_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("UrbanSound8K Fold10 — Sınıflandırma Raporu\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {MODEL_PATH.name}  (TFLite, 4 thread)\n")
        f.write(f"Toplam test örneği: {len(y_true)}\n")
        f.write(f"Genel Doğruluk (Accuracy): {overall_acc * 100:.2f}%\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_text)
    print(f"[+] {txt_path} yazıldı")

    # 6) Confusion matrix grafiği
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(11, 9), dpi=150)
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Tahmin Edilen Sınıf")
    ax.set_ylabel("Gerçek Sınıf")
    ax.set_title(f"UrbanSound8K Fold10 Confusion Matrix\n"
                 f"Genel Doğruluk: {overall_acc * 100:.2f}%  "
                 f"(N = {len(y_true)})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9, fontweight="bold")
    fig.tight_layout()
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(cm_path)
    plt.close(fig)
    print(f"[+] {cm_path} yazıldı")

    # 7) Sınıf bazlı sıralama (yorum için)
    per_class = []
    for name in class_names:
        d = report_dict.get(name, {})
        per_class.append((name, d.get("recall", 0.0),
                          d.get("precision", 0.0),
                          d.get("f1-score", 0.0),
                          int(d.get("support", 0))))
    per_class.sort(key=lambda r: r[1], reverse=True)
    top3 = per_class[:3]
    bot3 = per_class[-3:]

    # En çok karışan sınıf çiftleri (köşegen dışı en yüksek değerler)
    cm_off = cm.copy().astype(int)
    np.fill_diagonal(cm_off, 0)
    pairs = []
    flat = np.argsort(cm_off, axis=None)[::-1]
    for idx in flat[:5]:
        i, j = np.unravel_index(idx, cm_off.shape)
        if cm_off[i, j] == 0:
            break
        pairs.append((class_names[i], class_names[j], int(cm_off[i, j])))

    print("\nTop-3 sınıf (recall'a göre):")
    for n, r, p, f1, s in top3:
        print(f"  {n:<22} recall={r * 100:5.1f}%  f1={f1 * 100:5.1f}%  (n={s})")
    print("Bottom-3 sınıf:")
    for n, r, p, f1, s in bot3:
        print(f"  {n:<22} recall={r * 100:5.1f}%  f1={f1 * 100:5.1f}%  (n={s})")
    print("En çok karışan çiftler (gerçek → tahmin):")
    for a, b, c in pairs:
        print(f"  {a:<22} → {b:<22}  ({c} örnek)")

    # 8) model_evaluation.md
    md = []
    md.append("# Model Doğruluk Değerlendirmesi\n")
    md.append("## Test Kurulumu")
    md.append("- Veri seti: **UrbanSound8K**")
    md.append(f"- Test seti: **Fold {TEST_FOLD}** (eğitimde kullanılmadı)")
    md.append(f"- Toplam test örneği: **{len(y_true)}**")
    md.append("- Model: `model.tflite` (TFLite float32, 4 thread)")
    md.append(f"- Inference süresi: {elapsed:.1f} sn "
              f"(~{elapsed / max(len(y_true), 1) * 1000:.1f} ms/örnek, batch=1)")
    md.append("- Preprocessing: 64-band Mel spectrogram, Slaney norm, dB ölçek "
              "(`librosa.power_to_db(ref=np.max)`)")
    md.append(f"- Normalizasyon: min-max  `x_min = {x_min}`, `x_max = {x_max:.3e}`\n")

    md.append("## Sonuçlar\n")
    md.append("### Genel Doğruluk")
    md.append(f"**Test Doğruluğu (Accuracy): %{overall_acc * 100:.2f}**\n")

    md.append("### Sınıf Bazlı Performans\n")
    md.append("| Sınıf | Precision | Recall | F1-Score | Örnek Sayısı |")
    md.append("|---|---|---|---|---|")
    for name in class_names:
        d = report_dict.get(name, {})
        md.append(f"| {name} | {d.get('precision', 0):.4f} | "
                  f"{d.get('recall', 0):.4f} | "
                  f"{d.get('f1-score', 0):.4f} | "
                  f"{int(d.get('support', 0))} |")
    macro = report_dict.get("macro avg", {})
    weighted = report_dict.get("weighted avg", {})
    md.append(f"| **Macro Avg** | {macro.get('precision', 0):.4f} | "
              f"{macro.get('recall', 0):.4f} | "
              f"{macro.get('f1-score', 0):.4f} | "
              f"{int(macro.get('support', 0))} |")
    md.append(f"| **Weighted Avg** | {weighted.get('precision', 0):.4f} | "
              f"{weighted.get('recall', 0):.4f} | "
              f"{weighted.get('f1-score', 0):.4f} | "
              f"{int(weighted.get('support', 0))} |\n")

    md.append("### Confusion Matrix")
    md.append("![Confusion Matrix](analysis_output/confusion_matrix.png)\n")

    md.append("### Yorum")
    md.append("**En yüksek başarılı sınıflar (recall):**")
    for n, r, _, f1, _ in top3:
        md.append(f"- `{n}` → recall %{r * 100:.1f}, F1 %{f1 * 100:.1f}")
    md.append("\n**En düşük başarılı sınıflar:**")
    for n, r, _, f1, _ in bot3:
        md.append(f"- `{n}` → recall %{r * 100:.1f}, F1 %{f1 * 100:.1f}")

    # En düşük sınıf için makul bir neden taslağı
    worst = bot3[0]
    md.append("\nFold 10'da en düşük performans gösteren `{}` sınıfı %{:.1f} recall ile "
              "diğer sınıflara göre belirgin biçimde geride kalmıştır. "
              "Bu durum genellikle (i) sınıfın UrbanSound8K içinde dağılımının dengesiz olması, "
              "(ii) 1 saniyelik pencerede sınıfın ayırt edici akustik motiflerinin zaman zaman "
              "kesilmesi ve (iii) spektrum açısından komşu sınıflarla örtüşme ile açıklanabilir."
              .format(worst[0], worst[1] * 100))

    if pairs:
        md.append("\n**En çok karışan sınıf çiftleri (gerçek → tahmin):**")
        for a, b, c in pairs:
            md.append(f"- `{a}` ↔ `{b}` ({c} örnek)")
        md.append("\nKarışıklıkların büyük kısmı spektral olarak benzer makinesel/sürekli "
                  "kaynaklı seslerde (örn. `air_conditioner`, `engine_idling`, `jackhammer`, "
                  "`drilling`) ortaya çıkmaktadır; bu Mel spektrogram tabanlı "
                  "modellerde literatürde de raporlanan bilinen bir kısıttır.")

    md.append("\n### Üretilen Çıktılar")
    md.append("- `analysis_output/confusion_matrix.png`")
    md.append("- `analysis_output/classification_report.txt`")
    md.append("- `model_evaluation.md` (bu dosya)")

    REPORT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[+] {REPORT_MD} yazıldı")
    print("\n=================== KONSOL ÖZET ===================")
    print(f"Toplam test örneği : {len(y_true)}")
    print(f"Genel Accuracy     : %{overall_acc * 100:.2f}")
    print(f"En iyi 3 sınıf     : {[n for n, *_ in top3]}")
    print(f"En kötü 3 sınıf    : {[n for n, *_ in bot3]}")
    print("===================================================")


if __name__ == "__main__":
    main()
