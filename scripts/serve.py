"""
MODUL 4 - LAB SESI 2 - Langkah 4: Packaging - FastAPI

Uji lokal dulu sebelum build Docker:
    uvicorn serve:app --host 0.0.0.0 --port 8080

Cek:
    GET  http://localhost:8080/         -> Status & panduan endpoint
    GET  http://localhost:8080/health   -> {"status": "ok"}
    GET  http://localhost:8080/predict  -> Prediction via GET / query params
    POST http://localhost:8080/predict  -> {"prediction": ..., "probability": ...}
"""
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Sebelum di-package ke Docker, path ini diganti ke "./model"
# (lihat Dockerfile: COPY PATH_MODEL ./model)
MODEL_PATH = "./model"

app = FastAPI(title="MLOps Training - Model Serving")
model = mlflow.sklearn.load_model(MODEL_PATH)


class PredictRequest(BaseModel):
    age: int
    income: float
    lama_bekerja_tahun: float = 0.0
    skor_kredit_internal: int = 650
    jumlah_pinjaman_aktif: int = 0
    jumlah_tanggungan: int = 0


@app.get("/")
def root():
    return {
        "status": "online",
        "message": "MLOps Model Serving API is running",
        "endpoints": {
            "health": "/health",
            "predict_post": "/predict (POST)",
            "predict_get": "/predict (GET)",
            "documentation": "/docs",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict_get(
    age: int = 30,
    income: float = 5000000.0,
    lama_bekerja_tahun: float = 0.0,
    skor_kredit_internal: int = 650,
    jumlah_pinjaman_aktif: int = 0,
    jumlah_tanggungan: int = 0,
):
    data = {
        "age": age,
        "income": income,
        "lama_bekerja_tahun": lama_bekerja_tahun,
        "skor_kredit_internal": skor_kredit_internal,
        "jumlah_pinjaman_aktif": jumlah_pinjaman_aktif,
        "jumlah_tanggungan": jumlah_tanggungan,
    }
    X = pd.DataFrame([data])
    if hasattr(model, "feature_names_in_"):
        X = X[model.feature_names_in_]
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]
    return {
        "prediction": int(pred),
        "probability": round(float(proba), 4),
        "input": data,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    data = req.dict() if hasattr(req, "dict") else req.model_dump()
    X = pd.DataFrame([data])
    if hasattr(model, "feature_names_in_"):
        X = X[model.feature_names_in_]
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]
    return {"prediction": int(pred), "probability": round(float(proba), 4)}
