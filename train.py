"""
UrbanSound8K - Environmental Sound Classification with Compact CNN
=================================================================
Bu script UrbanSound8K veri seti ile çevresel ses sınıflandırması için
hafif bir CNN modeli eğitir ve mobil cihazlar için TensorFlow Lite formatında export eder.

Data augmentation, label smoothing, mixup ve geliştirilmiş CNN mimarisi ile.
"""

import json
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from pathlib import Path

# ============================================================================
# YAPILANDIRMA - Tüm hiperparametreler tek yerde
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "UrbanSound8K"
METADATA_PATH = DATASET_DIR / "metadata" / "UrbanSound8K.csv"
AUDIO_DIR = DATASET_DIR / "audio"

# Ses ön işleme parametreleri
TARGET_SR = 16000          # Hedef örnekleme frekansı (Hz) - mobil uyumlu
DURATION_SEC = 1.0         # Her örnek tam 1 saniye (pad veya trim)
N_MELS = 64                # Mel filtre bankası sayısı
N_FFT = 1024               # FFT penceresi boyutu
HOP_LENGTH = 512           # Frame atlama uzunluğu

# Fold bölünmesi (UrbanSound8K standart split)
TRAIN_FOLDS = list(range(1, 10))   # Fold 1-9: eğitim
TEST_FOLD = 10                     # Fold 10: test

# Model parametreleri
EPOCHS = 100               # EarlyStopping ile erken durdurulacak
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.1     # Eğitim foldlarından %10 validation (fold10 KULLANILMAZ)
LABEL_SMOOTHING = 0.1      # Categorical crossentropy label smoothing
MIXUP_ALPHA = 0.1          # Mixup augmentation alpha değeri (düşük = daha az agresif)
WARMUP_EPOCHS = 5           # Cosine decay warmup epoch sayısı

# Augmentation parametreleri
AUGMENT_PROB = 0.5          # Her örneğe augmentation uygulama olasılığı
NOISE_SNR_RANGE = (10, 25)  # Gaussian gürültü SNR aralığı (dB)
TIME_STRETCH_RANGE = (0.85, 1.15)  # Time stretch rate aralığı
PITCH_SHIFT_RANGE = (-2, 2)  # Pitch shift semitone aralığı
VOLUME_GAIN_RANGE = (-6, 6)  # Volume perturbation gain aralığı (dB)
SPEC_AUGMENT_FREQ_MASKS = 2  # SpecAugment frekans maskesi sayısı
SPEC_AUGMENT_FREQ_WIDTH = 8  # SpecAugment frekans maskesi max genişlik
SPEC_AUGMENT_TIME_MASKS = 2  # SpecAugment zaman maskesi sayısı
SPEC_AUGMENT_TIME_WIDTH = 5  # SpecAugment zaman maskesi max genişlik

# Çıktı dosyaları
OUTPUT_MODEL_TFLITE = BASE_DIR / "model.tflite"
OUTPUT_LABELS = BASE_DIR / "labels.txt"
OUTPUT_NORM_PARAMS = BASE_DIR / "norm_params.json"  # Mobil çıkarım için


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """
    UrbanSound8K metadata CSV dosyasını yükler.
    Kolonlar: slice_file_name, fsID, start, end, salience, fold, classID, class
    """
    df = pd.read_csv(metadata_path)
    return df


def get_audio_path(row: pd.Series) -> Path:
    """Ses dosyasının tam yolunu oluşturur (örn: audio/fold1/100032-3-0-0.wav)."""
    fold = int(row["fold"])
    filename = row["slice_file_name"]
    return AUDIO_DIR / f"fold{fold}" / filename


def load_and_preprocess_audio(file_path: Path) -> np.ndarray:
    """
    Ses dosyasını yükler ve ön işlem uygular:
    1. librosa ile yükle (orijinal sr'yi otomatik oku)
    2. 16 kHz'e örnekle
    3. Tam 1 saniye yap (pad veya trim)
    """
    y, _ = librosa.load(str(file_path), sr=TARGET_SR, mono=True)
    target_samples = int(TARGET_SR * DURATION_SEC)

    if len(y) < target_samples:
        # Kısa ise sıfırla pad et (sonuna)
        y = np.pad(y, (0, target_samples - len(y)), mode="constant", constant_values=0)
    else:
        # Uzun ise ortadan trim et
        start = (len(y) - target_samples) // 2
        y = y[start : start + target_samples]

    return y.astype(np.float32)


def extract_mel_spectrogram(y: np.ndarray) -> np.ndarray:
    """
    Mel Spectrogram çıkarır:
    - n_mels=64, n_fft=1024, hop_length=512
    - dB ölçeğine dönüştürür (log scale)
    Çıktı shape: (n_mels, time_steps) -> (64, 32) yaklaşık
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=TARGET_SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        center=False,
        fmin=0,
        fmax=TARGET_SR // 2,
        norm="slaney",  # Android MelSpectrogramExtractor ile aynı
    )
    # Log scale (dB) - daha iyi dinamik aralık
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


# ============================================================================
# DATA AUGMENTATION - Ham audio üzerine uygulanan augmentasyonlar
# ============================================================================

def add_gaussian_noise(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gaussian gürültü ekle (SNR 10-25 dB arası)."""
    snr_db = rng.uniform(*NOISE_SNR_RANGE)
    signal_power = np.mean(y ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=y.shape).astype(np.float32)
    return y + noise


def time_stretch(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Time stretching (rate 0.85 - 1.15), ardından orijinal uzunluğa pad/trim."""
    rate = rng.uniform(*TIME_STRETCH_RANGE)
    y_stretched = librosa.effects.time_stretch(y, rate=rate)
    target_len = int(TARGET_SR * DURATION_SEC)
    if len(y_stretched) < target_len:
        y_stretched = np.pad(y_stretched, (0, target_len - len(y_stretched)), mode="constant")
    else:
        start = (len(y_stretched) - target_len) // 2
        y_stretched = y_stretched[start : start + target_len]
    return y_stretched.astype(np.float32)


def pitch_shift(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pitch shifting (±2 semitone)."""
    n_steps = rng.uniform(*PITCH_SHIFT_RANGE)
    y_shifted = librosa.effects.pitch_shift(y, sr=TARGET_SR, n_steps=n_steps)
    return y_shifted.astype(np.float32)


def volume_perturbation(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Volume perturbation (gain ±6 dB)."""
    gain_db = rng.uniform(*VOLUME_GAIN_RANGE)
    gain_linear = 10 ** (gain_db / 20)
    return (y * gain_linear).astype(np.float32)


def spec_augment(mel_spec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    SpecAugment: mel spectrogram üzerinde rastgele frekans ve zaman bantları maskeleme.
    - 2 frekans bandı, max 8 mel band genişlik
    - 2 zaman bandı, max 5 frame genişlik
    Maskeleme değeri olarak mel'in minimum değerini kullanır (-80 dB),
    çünkü 0.0 kullanmak dB alanında max değere denk gelir ve artifact yaratır.
    """
    mel = mel_spec.copy()
    n_mels, n_frames = mel.shape
    mask_value = mel.min()  # Genellikle -80.0 dB

    # Frekans maskeleme
    for _ in range(SPEC_AUGMENT_FREQ_MASKS):
        f = rng.integers(1, SPEC_AUGMENT_FREQ_WIDTH + 1)
        f0 = rng.integers(0, max(1, n_mels - f))
        mel[f0 : f0 + f, :] = mask_value

    # Zaman maskeleme
    for _ in range(SPEC_AUGMENT_TIME_MASKS):
        t = rng.integers(1, SPEC_AUGMENT_TIME_WIDTH + 1)
        t0 = rng.integers(0, max(1, n_frames - t))
        mel[:, t0 : t0 + t] = mask_value

    return mel


# Audio augmentasyon fonksiyonları listesi (SpecAugment hariç, o mel üzerine)
AUDIO_AUGMENTATIONS = [add_gaussian_noise, time_stretch, pitch_shift, volume_perturbation]


def augment_audio(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Ham audio üzerine rastgele 2-3 augmentasyon uygula.
    %50 ihtimalle augmente et, yoksa orijinali döndür.
    """
    if rng.random() > AUGMENT_PROB:
        return y

    n_augs = rng.integers(2, 4)  # 2 veya 3 augmentasyon
    chosen = rng.choice(len(AUDIO_AUGMENTATIONS), size=n_augs, replace=False)
    for idx in chosen:
        y = AUDIO_AUGMENTATIONS[idx](y, rng)
    return y


def prepare_dataset(df: pd.DataFrame, augment: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Metadata'dan tüm örnekleri yükler, ön işler ve Mel Spectrogram'a dönüştürür.
    augment=True ise: orijinal + augmente edilmiş kopya eklenir.
    Dönen: (X, y)
    """
    rng = np.random.default_rng(42)
    X_list = []
    y_list = []

    for idx, row in df.iterrows():
        file_path = get_audio_path(row)
        if not file_path.exists():
            print(f"Uyarı: Dosya bulunamadı: {file_path}")
            continue

        y_audio = load_and_preprocess_audio(file_path)
        class_id = int(row["classID"])

        # Orijinal örneği her zaman ekle
        mel_spec = extract_mel_spectrogram(y_audio)
        X_list.append(mel_spec)
        y_list.append(class_id)

        # Augmente edilmiş kopya (eğitim seti için)
        if augment:
            y_aug = augment_audio(y_audio.copy(), rng)
            mel_aug = extract_mel_spectrogram(y_aug)
            # SpecAugment mel üzerine uygulanır
            mel_aug = spec_augment(mel_aug, rng)
            X_list.append(mel_aug)
            y_list.append(class_id)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    # Conv2D için channel dimension ekle: (N, height, width) -> (N, height, width, 1)
    X = np.expand_dims(X, axis=-1)

    return X, y


def normalize_features(X: np.ndarray) -> np.ndarray:
    """
    Özellikleri normalize et (min-max veya z-score).
    dB değerleri genellikle negatif; -80 ile 0 arası.
    Model performansı için normalization önemli.
    """
    # Global min-max (veri seti üzerinden hesaplanacak)
    x_min = X.min()
    x_max = X.max()
    if x_max - x_min > 1e-6:
        X_norm = (X - x_min) / (x_max - x_min)
    else:
        X_norm = X
    return X_norm.astype(np.float32), x_min, x_max


def build_model(input_shape: tuple, num_classes: int) -> tf.keras.Model:
    """
    Güçlendirilmiş CNN mimarisi (mobil uyumlu, residual connection ile):
    - Conv2D(32) + BatchNorm + ReLU + MaxPool(2,2)
    - Conv2D(64) + BatchNorm + ReLU + MaxPool(2,2)
    - Conv2D(128) + BatchNorm + ReLU + Conv2D(128) + BatchNorm + ReLU (residual)
    - GlobalAveragePooling2D
    - Dense(256) + ReLU + Dropout(0.5)
    - Dense(num_classes, softmax)
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Block 1: Conv2D(32) + BatchNorm + ReLU + MaxPool
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Block 2: Conv2D(64) + BatchNorm + ReLU + MaxPool
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Block 3: Residual block with Conv2D(128)
    # 1x1 projection: channel sayısını 64 -> 128 yaparak residual connection'ı mümkün kıl
    shortcut = tf.keras.layers.Conv2D(128, kernel_size=(1, 1), padding="same")(x)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Residual connection: input'u output'a topla
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation("relu")(x)

    # Head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def mixup(X: np.ndarray, y: np.ndarray, alpha: float = 0.2,
          rng: np.random.Generator = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Mixup augmentation: iki rastgele örneği karıştırır.
    X: (N, H, W, C), y: (N, num_classes) one-hot encoded
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(X)
    indices = rng.permutation(n)
    lam = rng.beta(alpha, alpha, size=(n, 1, 1, 1)).astype(np.float32)
    lam_y = lam.reshape(n, 1)

    X_mixed = lam * X + (1 - lam) * X[indices]
    y_mixed = lam_y * y + (1 - lam_y) * y[indices]

    return X_mixed, y_mixed


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    İlk warmup_steps adımda LR'yi 0'dan target_lr'ye lineer artır,
    sonra cosine decay uygula.
    """

    def __init__(self, warmup_steps: int, target_lr: float,
                 cosine_schedule: tf.keras.optimizers.schedules.CosineDecay):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.cosine_schedule = cosine_schedule

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)

        # Warmup: lineer artış
        warmup_lr = self.target_lr * (step / tf.maximum(warmup_steps, 1.0))

        # Cosine decay (warmup sonrası)
        cosine_lr = self.cosine_schedule(step - warmup_steps)

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "warmup_steps": self.warmup_steps,
            "target_lr": self.target_lr,
            "cosine_schedule": self.cosine_schedule.get_config(),
        }


def main():
    print("=" * 60)
    print("UrbanSound8K - Environmental Sound Classification")
    print("=" * 60)

    # 1) Metadata yükle
    print("\n[1] Metadata yükleniyor...")
    df = load_metadata(METADATA_PATH)
    print(f"    Toplam {len(df)} örnek, {df['class'].nunique()} sınıf")

    # 2) Fold bazlı bölme ve sınıf isimleri (classID sırasına göre)
    print("\n[2] Fold bazlı split uygulanıyor...")
    df_train = df[df["fold"].isin(TRAIN_FOLDS)].reset_index(drop=True)
    df_test = df[df["fold"] == TEST_FOLD].reset_index(drop=True)
    class_names = df.sort_values("classID")["class"].drop_duplicates().tolist()
    print(f"    Eğitim: {len(df_train)} örnek (fold {TRAIN_FOLDS})")
    print(f"    Test:   {len(df_test)} örnek (fold {TEST_FOLD})")

    # 3) Veri hazırlama (eğitim seti augmentation ile)
    print("\n[3] Ses dosyaları yüklenip Mel Spectrogram'a dönüştürülüyor...")
    print("    (Augmentation aktif, bu işlem birkaç dakika sürebilir...)")
    X_train, y_train = prepare_dataset(df_train, augment=True)
    X_test, y_test = prepare_dataset(df_test, augment=False)

    print(f"    X_train shape: {X_train.shape}")
    print(f"    X_test shape:  {X_test.shape}")
    print(f"    Sınıflar: {class_names}")

    # 4) Normalizasyon (eğitim istatistikleri ile)
    print("\n[4] Özellik normalizasyonu...")
    X_train_norm, x_min, x_max = normalize_features(X_train)
    X_test_norm = (X_test - x_min) / (x_max - x_min) if (x_max - x_min) > 1e-6 else X_test
    X_test_norm = X_test_norm.astype(np.float32)

    # x_min ve x_max değerlerini yazdır
    # NOT: x_max ≈ 0 (örn: 3.81e-06) normaldir çünkü librosa.power_to_db(ref=np.max)
    # kullanılıyor ve max değer 0 dB oluyor. x_min genellikle -80 dB civarıdır.
    print(f"    x_min = {x_min}")
    print(f"    x_max = {x_max}")

    # 5) One-hot encoding (label smoothing için gerekli)
    num_classes = len(class_names)
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes).astype(np.float32)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes).astype(np.float32)

    # 6) Eğitim verisini karıştır ve validation split (fold10 KULLANILMAZ)
    print("\n[5] Validation split (eğitim foldlarının %10'u)...")
    rng = np.random.default_rng(42)
    shuffle_idx = rng.permutation(len(X_train_norm))
    X_train_shuf = X_train_norm[shuffle_idx]
    y_train_shuf = y_train_onehot[shuffle_idx]
    split_at = int(len(X_train_shuf) * (1 - VALIDATION_SPLIT))
    X_tr, X_val = X_train_shuf[:split_at], X_train_shuf[split_at:]
    y_tr, y_val = y_train_shuf[:split_at], y_train_shuf[split_at:]
    print(f"    Eğitim: {len(X_tr)}, Validation: {len(X_val)} (fold10 hariç)")

    # 7) Mixup augmentation (eğitim verisi üzerinde)
    print("\n[6] Mixup augmentation uygulanıyor (alpha={})...".format(MIXUP_ALPHA))
    X_tr, y_tr = mixup(X_tr, y_tr, alpha=MIXUP_ALPHA, rng=rng)

    # 8) Model oluştur
    input_shape = X_train_norm.shape[1:]
    print(f"\n[7] Model oluşturuluyor... Input shape: {input_shape}")

    model = build_model(input_shape, num_classes)

    # CosineDecay learning rate scheduler (warmup ile)
    steps_per_epoch = len(X_tr) // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=total_steps - warmup_steps,
        alpha=1e-6,  # minimum learning rate
    )
    # Warmup: ilk birkaç epoch'ta LR'yi lineer artır
    warmup_schedule = WarmupCosineDecay(
        warmup_steps=warmup_steps,
        target_lr=LEARNING_RATE,
        cosine_schedule=lr_schedule,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=warmup_schedule),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    model.summary()

    # 9) Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # 10) Eğitim (validation: eğitim foldlarından %10, fold10 DEĞİL)
    print("\n[8] Eğitim başlıyor...")
    model.fit(
        X_tr,
        y_tr,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # 11) Fold10 ile nihai test doğruluğu (eğitimde hiç kullanılmadı)
    print("\n[9] Fold10 test seti değerlendirmesi...")
    test_loss, test_acc = model.evaluate(X_test_norm, y_test_onehot, verbose=1)
    print(f"\n    >>> Final Test Accuracy (fold10): {test_acc:.4f} ({test_acc*100:.2f}%) <<<")

    # 12) TensorFlow Lite export
    print("\n[10] TensorFlow Lite model export ediliyor...")

    # TFLite Converter - float32 model (mobil uyumlu)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(OUTPUT_MODEL_TFLITE, "wb") as f:
        f.write(tflite_model)
    print(f"    Model kaydedildi: {OUTPUT_MODEL_TFLITE}")

    # Class labels
    with open(OUTPUT_LABELS, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"    Etiketler kaydedildi: {OUTPUT_LABELS}")

    # Normalizasyon parametreleri (mobil çıkarımda aynı preprocess gerekli)
    with open(OUTPUT_NORM_PARAMS, "w", encoding="utf-8") as f:
        json.dump({"x_min": float(x_min), "x_max": float(x_max)}, f)
    print(f"    Norm parametreleri: {OUTPUT_NORM_PARAMS}")

    # Mobil uyumluluk doğrulama
    print("\n[11] Mobil çıkarım uyumluluğu doğrulanıyor...")
    interpreter = tf.lite.Interpreter(model_path=str(OUTPUT_MODEL_TFLITE))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"    TFLite input shape: {input_details[0]['shape']}")
    print(f"    TFLite output shape: {output_details[0]['shape']}")
    print(f"    Beklenen input: (batch, {input_shape[0]}, {input_shape[1]}, {input_shape[2]})")

    print("\n" + "=" * 60)
    print("Eğitim tamamlandı!")
    print("=" * 60)


if __name__ == "__main__":
    main()
