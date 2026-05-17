"""
Cloud Inference Server — Edge-Assist Noise Classifier
======================================================
FastAPI tabanlı sunucu. Android tarafındaki CloudApiClient.kt ile birebir
uyumludur. Akademik tez için Edge vs Cloud karşılaştırması yapıldığından,
preprocessing (normalizasyon) Android tarafıyla %100 aynıdır:

    x_norm = (x - x_min) / (x_max - x_min)

Endpoint'ler:
    POST /predict   — Mel features (64 x T) alır, sınıflandırma döner.
    GET  /ping      — RTT ölçümü için hızlı yanıt.

Çalıştırma:
    pip install -r requirements.txt
    python server.py
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Sabitler — train.py ve Android tarafıyla birebir aynı olmalı
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.tflite"
LABELS_PATH = BASE_DIR / "labels.txt"
NORM_PARAMS_PATH = BASE_DIR / "norm_params.json"
METRICS_CSV_PATH = BASE_DIR / "server_metrics.csv"

EXPECTED_MEL_BANDS = 64   # height
EXPECTED_TIME_FRAMES = 30  # width

# ---------------------------------------------------------------------------
# Yükleme
# ---------------------------------------------------------------------------
def _load_labels(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _load_norm_params(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data["x_min"]), float(data["x_max"])


print(f"[server] BASE_DIR = {BASE_DIR}")
print(f"[server] Model dosyası: {MODEL_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"model.tflite bulunamadı: {MODEL_PATH}")
if not LABELS_PATH.exists():
    raise FileNotFoundError(f"labels.txt bulunamadı: {LABELS_PATH}")
if not NORM_PARAMS_PATH.exists():
    raise FileNotFoundError(f"norm_params.json bulunamadı: {NORM_PARAMS_PATH}")

LABELS: List[str] = _load_labels(LABELS_PATH)
X_MIN, X_MAX = _load_norm_params(NORM_PARAMS_PATH)
NORM_RANGE = X_MAX - X_MIN

print(f"[server] Etiket sayısı: {len(LABELS)} -> {LABELS}")
print(f"[server] norm_params: x_min={X_MIN}, x_max={X_MAX}, range={NORM_RANGE:.6e}")
if abs(NORM_RANGE) < 1e-5:
    print("[server] UYARI: norm_params range neredeyse sıfır!")

# TFLite Interpreter — 4 thread (Android tarafıyla aynı)
interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH), num_threads=4)
interpreter.allocate_tensors()
INPUT_DETAILS = interpreter.get_input_details()
OUTPUT_DETAILS = interpreter.get_output_details()

print(f"[server] TFLite input  details: shape={INPUT_DETAILS[0]['shape'].tolist()}, "
      f"dtype={INPUT_DETAILS[0]['dtype']}")
print(f"[server] TFLite output details: shape={OUTPUT_DETAILS[0]['shape'].tolist()}, "
      f"dtype={OUTPUT_DETAILS[0]['dtype']}")

# ---------------------------------------------------------------------------
# CSV metric logging
# ---------------------------------------------------------------------------
CSV_HEADER = [
    "timestamp",
    "request_bytes",
    "inference_ms",
    "total_handler_ms",
    "prediction",
    "confidence",
]

if not METRICS_CSV_PATH.exists():
    with METRICS_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(CSV_HEADER)
    print(f"[server] {METRICS_CSV_PATH.name} oluşturuldu (başlık yazıldı).")


def _append_metric(row: list) -> None:
    try:
        with METRICS_CSV_PATH.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
    except Exception as exc:  # pragma: no cover
        print(f"[server] CSV log hatası: {exc}")


# ---------------------------------------------------------------------------
# FastAPI uygulaması
# ---------------------------------------------------------------------------
app = FastAPI(title="Edge-Assist Noise Classifier — Cloud Inference")


class PredictRequest(BaseModel):
    # Android: data class PredictRequest(val mfcc: Array<FloatArray>)
    # Gson 2D float array -> JSON: {"mfcc": [[...], [...], ...]}  (64 x T)
    mfcc: List[List[float]] = Field(..., description="Mel spectrogram (64 x T)")


def _preprocess(mel: List[List[float]]) -> np.ndarray:
    """Android `TFLiteClassifier.classifyWithModel` ile birebir aynı:
    - Beklenen shape: (64, 30). Daha küçükse 0 ile padlenir, daha büyükse kırpılır.
    - Normalizasyon: (x - x_min) / (x_max - x_min)
    - Çıkış shape: (1, 64, 30, 1) float32
    """
    out = np.zeros((EXPECTED_MEL_BANDS, EXPECTED_TIME_FRAMES), dtype=np.float32)
    rows = min(len(mel), EXPECTED_MEL_BANDS)
    for m in range(rows):
        row = mel[m]
        cols = min(len(row), EXPECTED_TIME_FRAMES)
        if cols:
            out[m, :cols] = row[:cols]

    if abs(NORM_RANGE) > 1e-6:
        out = (out - X_MIN) / NORM_RANGE
    # else: tüm değerler 0 olarak kalır (Android tarafı da bu durumda
    # geçerli bir çıktı üretmez; yalnızca güvenlik için)

    return out.reshape(1, EXPECTED_MEL_BANDS, EXPECTED_TIME_FRAMES, 1).astype(np.float32)


@app.get("/ping")
def ping():
    """RTT ölçümü için hızlı endpoint. ML kodu çalıştırmaz."""
    return {"status": "ok", "timestamp": time.time()}


@app.post("/predict")
async def predict(req: PredictRequest, request: Request):
    handler_start = time.perf_counter()

    # İstek boyutu (Android'in gönderdiği byte sayısıyla aynı şekilde ölç)
    raw_bytes = await request.body() if False else None  # body zaten parse edildi
    try:
        content_length = int(request.headers.get("content-length", "0"))
    except ValueError:
        content_length = 0

    mel = req.mfcc
    if not mel or not isinstance(mel, list) or not isinstance(mel[0], list):
        raise HTTPException(status_code=400, detail="mfcc alanı 2D liste olmalı")

    try:
        x = _preprocess(mel)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"preprocessing hatası: {exc}")

    # Inference
    inf_start = time.perf_counter()
    interpreter.set_tensor(INPUT_DETAILS[0]["index"], x)
    interpreter.invoke()
    output = interpreter.get_tensor(OUTPUT_DETAILS[0]["index"])
    inf_ms = (time.perf_counter() - inf_start) * 1000.0

    probs = np.asarray(output, dtype=np.float32).reshape(-1)
    if probs.size == 0:
        raise HTTPException(status_code=500, detail="Model boş çıktı verdi")

    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    label = LABELS[top_idx] if 0 <= top_idx < len(LABELS) else f"class_{top_idx}"

    handler_ms = (time.perf_counter() - handler_start) * 1000.0

    # CSV log
    _append_metric([
        time.time(),
        content_length,
        f"{inf_ms:.3f}",
        f"{handler_ms:.3f}",
        label,
        f"{confidence:.6f}",
    ])

    print(
        f"[predict] bytes={content_length} inf={inf_ms:.2f}ms "
        f"handler={handler_ms:.2f}ms -> {label} ({confidence:.3f})"
    )

    # Android: PredictResponse(label: String, confidence: Float, inference_ms: Long)
    # inference_ms Long olarak parse ediliyor → integer döndür.
    return {
        "label": label,
        "confidence": confidence,
        "inference_ms": int(round(inf_ms)),
    }


# ---------------------------------------------------------------------------
# Çalıştırma
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[server] http://0.0.0.0:8000 üzerinde başlatılıyor...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
