# EdgeAssistNoiseClassifier

Mobil cihazda **Edge (on-device TFLite)**, **Cloud (FastAPI inference server)** ve **Adaptive** çalışma modlarını karşılaştıran çevresel ses sınıflandırma sistemi. UrbanSound8K üzerinde eğitilmiş 64×30 Mel-Spectrogram tabanlı bir CNN modelini Android tarafında TFLite ile, sunucu tarafında ise aynı `.tflite` ağırlığıyla servis ederek **latency**, **bandwidth**, **batarya** ve **doğruluk** ölçümlerini almak için tasarlandı.

> Mobil Programlama dersi kapsamında geliştirilmiştir.

---

## Sistem Mimarisi

```
┌────────────────────────────┐         ┌────────────────────────┐
│   Android App (Kotlin)     │         │   FastAPI Server       │
│  ┌──────────────────────┐  │         │  ┌──────────────────┐  │
│  │ AudioRecorder        │  │         │  │  /predict        │  │
│  │ MelSpectrogram (64×30)│ │  HTTP   │  │  /ping           │  │
│  │ TFLiteClassifier      ├──┼────────►│  TFLite Interpreter│  │
│  │ AdaptiveDecisionEngine│ │         │  │  (num_threads=4) │  │
│  │ MetricsLogger / CSV   │ │         │  │  server_metrics  │  │
│  └──────────────────────┘  │         │  └──────────────────┘  │
└────────────────────────────┘         └────────────────────────┘
        Edge / Adaptive                       Cloud / Adaptive
```

**Üç çalışma modu:**
- **Edge** — Tahmin tamamen telefonda, `model.tflite` ile yapılır.
- **Cloud** — Mel-spectrogram FastAPI sunucusuna gönderilir, sonuç döner.
- **Adaptive** — `AdaptiveDecisionEngine` ağ koşullarına ve sessizliğe bakarak Edge/Cloud arasında otomatik seçim yapar.

---

## Veri Seti & Model

- **Veri seti:** UrbanSound8K (10 sınıf)
- **Test seti:** Fold 10 (eğitimde tutuldu)
- **Giriş:** 64-band Mel-spectrogram, 30 frame, Slaney norm, dB ölçek (`librosa.power_to_db(ref=np.max)`)
- **Normalizasyon:** min-max — `x_min = -80.0`, `x_max = 3.815e-06` (`norm_params.json`)
- **Çıkış:** 10 sınıf softmax (bkz. `labels.txt`)
- **Model formatı:** TFLite float32, 4 thread

### Sınıflar
`air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `gun_shot`, `jackhammer`, `siren`, `street_music`

### Doğruluk (Test — Fold 10, 837 örnek)
| Metrik | Değer |
|---|---|
| Accuracy | **%76.22** |
| Macro F1 | 0.7773 |
| Weighted F1 | 0.7614 |
| Inference süresi | ~25.2 ms/örnek (batch=1, CPU) |

Detay: [model_evaluation.md](model_evaluation.md)

---

## Latency & Enerji Karşılaştırması

Telefon üzerinde 6424 inference (~28 dk, WiFi):

| Mod | Ortalama (ms) | Medyan (ms) | P95 (ms) | Batarya (%/dk) |
|---|---|---|---|---|
| Edge | **35.9** | 34.0 | 44.0 | -0.24 |
| Cloud | 81.8 | 63.0 | 121.7 | **-0.13** |
| Adaptive | 89.7 | 73.0 | 206.9 | -0.19 |

- Cloud, Edge'e göre ~**2.3× daha yavaş** ama batarya açısından daha hafif.
- Cloud bandwidth: ~**20.14 KB/inference** (4 inference/sn ⇒ ~80.6 KB/sn).
- Adaptive WiFi'de %91.7 cloud, %8.3 edge seçti.

Detay: [analysis_output/summary.txt](analysis_output/summary.txt)

---

## Proje Yapısı

```
mobil makine/
├── app/                        # Android uygulaması (Kotlin + Compose)
│   └── src/main/java/com/edgeassist/noiseclassifier/
│       ├── audio/              # AudioRecorder (16 kHz PCM)
│       ├── ml/                 # MelSpectrogramExtractor, TFLiteClassifier, smoothing
│       ├── network/            # CloudApiClient (Retrofit)
│       ├── adaptive/           # AdaptiveDecisionEngine
│       ├── metrics/            # CSV logger (latency, bandwidth, battery)
│       └── ui/                 # Jetpack Compose ekranları
├── server.py                   # FastAPI cloud inference sunucusu
├── train.py                    # Model eğitim scripti
├── evaluate_model.py           # Fold 10 test seti değerlendirmesi
├── analyze_results.py          # Edge/Cloud/Adaptive log analizi + grafikler
├── verify_mel.py               # Android ↔ Python mel-spectrogram eşitlik testi
├── model.tflite                # Eğitilmiş TFLite model
├── labels.txt                  # 10 sınıf etiketleri
├── norm_params.json            # x_min / x_max
├── requirements.txt
├── README_SERVER.md            # Sunucu özelinde detaylı doküman
└── model_evaluation.md         # Doğruluk raporu
```

---

## Kurulum

### 1) Cloud Inference Sunucusu

```bash
pip install -r requirements.txt
python server.py
```

Sunucu `http://0.0.0.0:8000` adresinde başlar. Endpoint detayları ve LAN üzerinden gerçek cihaza bağlama adımları için **[README_SERVER.md](README_SERVER.md)** dosyasına bakın.

Hızlı sağlık kontrolü:
```bash
curl http://localhost:8000/ping
```

### 2) Android Uygulaması

**Gereksinimler:** Android Studio (Giraffe+), JDK 17, Android SDK 34, minSdk 26.

```bash
./gradlew :app:assembleDebug
# veya Android Studio'da "Run 'app'"
```

Bağlantı:
- **Emülatör:** `CloudApiClient.kt` içinde `DEFAULT_BASE_URL = "http://10.0.2.2:8000"` zaten ayarlı, ekstra ayar gerekmez.
- **Gerçek telefon (LAN):** Bilgisayarın LAN IP'sini (`ipconfig`) `DEFAULT_BASE_URL`'e yazıp 8000/TCP portunu firewall'da açın. Telefon ile bilgisayar aynı Wi-Fi'da olmalı.

### 3) Modeli Yeniden Eğitmek (opsiyonel)

```bash
# UrbanSound8K'i indirip ./UrbanSound8K/ olarak yerleştirin
python train.py            # model.tflite + norm_params.json üretir
python evaluate_model.py   # fold 10 üzerinde doğruluk raporu
python verify_mel.py       # Android ↔ Python preprocess parite testi
```

---

## Ölçüm & Analiz

Uygulama her inference'ı CSV'ye yazar (`MetricsLogger`). Toplanan loglarla grafik üretmek için:

```bash
python analyze_results.py
```

Çıktılar `analysis_output/`:
- `latency_comparison.png`, `latency_distribution.png`, `latency_breakdown.png`
- `bandwidth_usage.png`
- `battery_timeline.png`
- `adaptive_decisions.png`
- `accuracy_comparison.png`, `confusion_matrix.png`
- `summary.txt` (sayısal özet)

---

## Edge ↔ Cloud Preprocess Eşitliği

Karşılaştırmanın adil olabilmesi için **Android `MelSpectrogramExtractor.kt`** ile **`server.py` / `train.py`** içindeki mel-spectrogram çıkarımı bire bir aynıdır:

- 16 kHz mono PCM
- 64 mel bandı, 30 frame, Slaney norm
- `librosa.power_to_db(ref=np.max)` ile aynı dB davranışı
- `(x − x_min) / (x_max − x_min)` ile min-max normalizasyon
- TFLite girişi: `(1, 64, 30, 1)` float32

`train.py` veya `MelSpectrogramExtractor.kt` değişirse `server.py` de eşzamanlı güncellenmeli — `verify_mel.py` bu pariteyi otomatik kontrol eder.

---

## Lisans

Akademik amaçlı, ders kapsamında geliştirilmiştir. UrbanSound8K veri seti kendi lisansına tabidir.
