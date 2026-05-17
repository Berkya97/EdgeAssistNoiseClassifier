# Cloud Inference Sunucusu

FastAPI tabanlı, Android uygulamasının `CloudApiClient.kt`'si ile birebir uyumlu
inference sunucusu. Edge vs Cloud karşılaştırması için aynı `model.tflite`,
`labels.txt` ve `norm_params.json` dosyalarını proje kökünden yükler ve aynı
preprocessing'i uygular.

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
python server.py
```

Sunucu `http://0.0.0.0:8000` adresinde dinler. Başlatma sırasında konsola:

- `BASE_DIR`, model yolu
- Etiketler ve `norm_params` (`x_min`, `x_max`)
- TFLite input/output shape ve dtype

bilgilerini yazdırır.

## Bağlantı

### Android emülatörü
Hiçbir ek ayara gerek yok — `CloudApiClient.kt` içinde
`DEFAULT_BASE_URL = "http://10.0.2.2:8000"` zaten kuruludur. (10.0.2.2,
emülatörün host makineye yönlendirdiği özel adres.)

### Gerçek telefon (LAN)
1. Bilgisayarın LAN IP'sini öğren:
   ```cmd
   ipconfig
   ```
   `IPv4 Address` satırına bak (örn. `192.168.1.42`).

2. `CloudApiClient.kt` içindeki `DEFAULT_BASE_URL`'yi geçici olarak değiştir:
   ```kotlin
   const val DEFAULT_BASE_URL = "http://192.168.1.42:8000"
   ```

3. Windows Firewall'da 8000/TCP portunu aç (yönetici komut isteminde):
   ```cmd
   netsh advfirewall firewall add rule name="FastAPI 8000" dir=in action=allow protocol=TCP localport=8000
   ```

4. Telefon ve bilgisayar **aynı Wi-Fi**'da olmalı.

## Endpoint'ler

### `GET /ping`
RTT ölçümü için. ML kodu çalıştırmaz.

```bash
curl http://localhost:8000/ping
# {"status":"ok","timestamp":1712841600.123}
```

### `POST /predict`
Mel spectrogram (64 × 30, ham dB) gönderir, sınıflandırma sonucu döner.

İstek (Android `CloudApiClient.PredictRequest` ile birebir):
```json
{
  "mfcc": [
    [-72.1, -65.3, ...],
    [-70.0, -64.8, ...],
    ...
  ]
}
```

Yanıt (Android `CloudApiClient.PredictResponse` ile birebir):
```json
{
  "label": "dog_bark",
  "confidence": 0.873,
  "inference_ms": 12
}
```

Manuel test (64 × 30 sıfır matrisi):
```bash
python -c "import json; print(json.dumps({'mfcc': [[0.0]*30]*64}))" > body.json
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     --data-binary @body.json
```

## Metric Logging

Her `/predict` çağrısı `server_metrics.csv`'ye eklenir:

```
timestamp, request_bytes, inference_ms, total_handler_ms, prediction, confidence
```

Dosya yoksa otomatik oluşturulur ve başlık satırı yazılır.

## Preprocessing — Edge ile Eşitlik

Sunucu, Android `TFLiteClassifier.classifyWithModel` ile aynı işlemi yapar:

- Beklenen mel shape: `(64, 30)`. Daha küçükse 0 ile padlenir, daha büyükse kırpılır.
- Normalizasyon: `(x − x_min) / (x_max − x_min)` (`norm_params.json`'dan).
- Model girişi: `(1, 64, 30, 1)` float32, HWC sırasıyla aynı bellek düzeni.
- TFLite Interpreter `num_threads=4` (Android tarafıyla aynı).

`train.py` veya `MelSpectrogramExtractor.kt`'deki preprocessing değişirse,
`server.py`'nin de eşzamanlı güncellenmesi gerekir; aksi halde Edge vs Cloud
karşılaştırması haksız olur.
