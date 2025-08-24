from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import pandas as pd
import os
import tarfile

# ---------------- CONFIG ------------------------
BUCKET = "rossmann-sales-bucket"
MODEL_KEY = "rf-hpo-output/rf-hpo-2025-08-24-02-16-36-009-8c46cc4d/output/model.tar.gz"

SELECTED_FEATURES = [
    "Store",
    "DayOfWeek",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "Year",
    "Month",
    "WeekOfYear",
    "Day",
    "IsWeekend",
    "IsPromoMonth",
    "Promo2Active",
    "CompetitionOpenTimeMonths"
]

# Initialize FastAPI app
app = FastAPI()

# Initialize boto3 client
s3 = boto3.client("s3")

# Global variable for model
model = None

# ---------------- Request Schema ------------------
class RossmannFeatures(BaseModel):
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
    CompetitionOpenTimeMonths: int


@app.get("/")
def health_check():
    return {"message": "Rossmann Sales Prediction API is healthy"}


# ---------------- Download and Extract Model ------------------
def download_and_extract_model():
    """Download model.tar.gz from S3 and extract"""
    model_dir = "/tmp/model_dir"
    os.makedirs(model_dir, exist_ok=True)

    local_tar_path = "/tmp/model.tar.gz"
    s3.download_file(BUCKET, MODEL_KEY, local_tar_path)

    with tarfile.open(local_tar_path, "r:gz") as tar:
        tar.extractall(path=model_dir)

    return os.path.join(model_dir, "model.joblib")


@app.on_event("startup")
def load_model():
    """Load model at startup"""
    global model
    try:
        model_path = download_and_extract_model()
        model = joblib.load(model_path)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


# ---------------- Prediction Endpoint ------------------
@app.post("/predict")
def predict(input: RossmannFeatures):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input.dict()])

        # Ensure column order matches training
        df = df[SELECTED_FEATURES]

        # Predict sales
        prediction = model.predict(df)[0]

        return {"predicted_sales": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
