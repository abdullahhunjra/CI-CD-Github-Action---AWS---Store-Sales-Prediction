from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3, os, joblib
import pandas as pd
import numpy as np
from datetime import datetime

# -------- Config ---------
BUCKET = os.getenv("BUCKET", "rossmann-sales-bucket")
MODEL_KEY = os.getenv("MODEL_KEY", "rossmann-artifacts/model.joblib")
ENCODER_KEY = os.getenv("ENCODER_KEY", "rossmann-artifacts/label_encoders.pkl")
REGION = os.getenv("AWS_REGION", "us-east-1")

# -------- Globals ---------
s3 = boto3.client("s3", region_name=REGION)
model = None
encoders = None

# -------- Input Schema --------
class Features(BaseModel):
    Store: int
    DayOfWeek: int
    Promo: int
    StateHoliday: str
    SchoolHoliday: int
    StoreType: str
    Assortment: str
    CompetitionDistance: float
    Year: int
    Month: int
    WeekOfYear: int
    Day: int
    IsWeekend: int
    IsPromoMonth: int
    Promo2Active: int
    CompetitionOpenTimeMonths: float

# -------- Download Helper --------
def download_from_s3(key, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(BUCKET, key, local_path)

# -------- Load model & encoders on startup --------
@app.on_event("startup")
def load_artifacts():
    global model, encoders
    try:
        model_path = "/tmp/model.joblib"
        enc_path = "/tmp/label_encoders.pkl"

        download_from_s3(MODEL_KEY, model_path)
        download_from_s3(ENCODER_KEY, enc_path)

        model = joblib.load(model_path)
        encoders = joblib.load(enc_path)

        print("✅ Model and encoders loaded")
    except Exception as e:
        print(f"❌ Failed loading artifacts: {e}")

# -------- API --------
app = FastAPI()

@app.get("/")
def health():
    return {"status": "alive", "model_loaded": model is not None}

@app.post("/predict")
def predict(features: Features):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame([features.dict()])

        # Apply label encoders
        for col in ["StateHoliday", "Assortment", "StoreType", "Store"]:
            if col in encoders:
                df[col] = encoders[col].transform(df[col])

        # Order columns exactly like training
        ordered_cols = [
            "Store","DayOfWeek","Promo","StateHoliday","SchoolHoliday",
            "StoreType","Assortment","CompetitionDistance","Year","Month",
            "WeekOfYear","Day","IsWeekend","IsPromoMonth","Promo2Active",
            "CompetitionOpenTimeMonths"
        ]
        df = df[ordered_cols]

        pred = model.predict(df)[0]
        return {"prediction": float(pred)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
