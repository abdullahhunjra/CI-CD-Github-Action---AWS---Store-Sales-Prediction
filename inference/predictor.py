import os
import joblib
import json
import boto3
import numpy as np
import pandas as pd
from datetime import datetime
from io import StringIO

# ---- GLOBAL CONFIG ----
BUCKET = "rossmann-sales-bucket"
ENCODER_PATH = "rossmann-artifacts/label_encoders.pkl"

# ---- Load Model & Preprocessor ----
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))

    # Download encoders from S3
    s3 = boto3.client("s3")
    s3.download_file(BUCKET, ENCODER_PATH, "/tmp/label_encoders.pkl")
    label_encoders = joblib.load("/tmp/label_encoders.pkl")

    return {"model": model, "label_encoders": label_encoders}

# ---- Input Handler ----
def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        return pd.DataFrame([input_data])
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

# ---- Preprocessing (same as training) ----
def preprocess(df, encoders):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    if "Open" in df.columns:
        df = df[df["Open"] == 1].drop(columns=["Open"])

    df["PromoInterval"] = df["PromoInterval"].fillna("NoPromo")
    df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0).astype(int)
    df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0).astype(int)
    df["MonthStr"] = df["Date"].dt.strftime("%b")
    df["IsPromoMonth"] = df.apply(
        lambda row: int(row["MonthStr"] in row["PromoInterval"].split(",")) if row["PromoInterval"] != "NoPromo" else 0,
        axis=1
    )
    df.drop(columns=["MonthStr"], inplace=True)
    df["Promo2Active"] = ((df["Promo2"] == 1) & (df["IsPromoMonth"] == 1)).astype(int)

    df = df[df["CompetitionDistance"].notna()]
    df["CompetitionDistance"] = np.log1p(df["CompetitionDistance"])

    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0).astype(int)
    df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(0).astype(int)

    df["CompetitionOpenSinceDate"] = df.apply(
        lambda row: datetime(
            year=row["CompetitionOpenSinceYear"],
            month=row["CompetitionOpenSinceMonth"],
            day=1
        ) if row["CompetitionOpenSinceYear"] > 0 and row["CompetitionOpenSinceMonth"] > 0 else row["Date"], axis=1
    )

    df["CompetitionOpenTimeMonths"] = (
        (df["Date"].dt.year - df["CompetitionOpenSinceDate"].dt.year) * 12 +
        (df["Date"].dt.month - df["CompetitionOpenSinceDate"].dt.month)
    ).apply(lambda x: max(x, 0))

    df["CompetitionOpenTimeMonths"] = np.log1p(df["CompetitionOpenTimeMonths"])

    # Drop unused
    drop_cols = [
        "Date", "Customers", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
        "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
        "CompetitionOpenSinceDate"
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors="ignore")

    # Encode
    df["StateHoliday"] = df["StateHoliday"].astype(str)
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))

    return df

# ---- Predict ----
def predict_fn(input_data, model_bundle):
    model = model_bundle["model"]
    encoders = model_bundle["label_encoders"]

    processed = preprocess(input_data.copy(), encoders)
    preds = model.predict(processed)

    # Reverse log1p if necessary
    final_preds = np.expm1(preds)
    return final_preds

# ---- Output Formatter ----
def output_fn(prediction, content_type):
    return json.dumps({"prediction": prediction.tolist()})
