import joblib
import os
import numpy as np
import json
import sys
import traceback
import pandas as pd
from datetime import datetime

# ---------------- FINAL FEATURE ORDER ---------------- #
FEATURE_COLUMNS = [
    "Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday",
    "StoreType", "Assortment", "CompetitionDistance", "Year", "Month",
    "WeekOfYear", "Day", "IsWeekend", "IsPromoMonth", "Promo2Active",
    "CompetitionOpenTimeMonths"
]

model = None
label_encoders = None


# ---------------- LOAD MODEL + ARTIFACTS ---------------- #
def model_fn(model_dir):
    global model, label_encoders
    try:
        model_path = os.path.join(model_dir, "model.joblib")
        encoders_path = os.path.join(model_dir, "label_encoders.pkl")

        print(f"🔹 Loading model from: {model_path}", flush=True)
        model = joblib.load(model_path)

        print(f"🔹 Loading label encoders from: {encoders_path}", flush=True)
        if os.path.exists(encoders_path):
            label_encoders = joblib.load(encoders_path)
        else:
            print("⚠️ No encoders found!", flush=True)

        print("✅ Model + encoders loaded successfully", flush=True)
        return model
    except Exception as e:
        print("❌ Model load failed:", e, flush=True)
        traceback.print_exc(file=sys.stdout)
        raise


# ---------------- PREPROCESSING (from preprocess.py) ---------------- #
def preprocess(input_json):
    df = pd.DataFrame([input_json])

    # --- Date features ---
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    # --- Promo2 features ---
    df["PromoInterval"] = df.get("PromoInterval", "NoPromo").fillna("NoPromo")
    df["MonthStr"] = df["Date"].dt.strftime("%b")
    df["IsPromoMonth"] = df.apply(
        lambda row: int(row["MonthStr"] in row["PromoInterval"].split(","))
        if row["PromoInterval"] != "NoPromo" else 0, axis=1
    )
    df["Promo2Active"] = ((df.get("Promo2", 0) == 1) & (df["IsPromoMonth"] == 1)).astype(int)

    # --- Competition distance ---
    df["CompetitionDistance"] = np.log1p(df["CompetitionDistance"])

    # --- Competition open months ---
    if "CompetitionOpenSinceYear" in df and "CompetitionOpenSinceMonth" in df:
        df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(0).astype(int)
        df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0).astype(int)

        def compute_months(row):
            if row["CompetitionOpenSinceYear"] > 0 and row["CompetitionOpenSinceMonth"] > 0:
                comp_date = datetime(year=row["CompetitionOpenSinceYear"],
                                     month=row["CompetitionOpenSinceMonth"], day=1)
                months = (row["Date"].year - comp_date.year) * 12 + (row["Date"].month - comp_date.month)
                return max(months, 0)
            else:
                return 0

        df["CompetitionOpenTimeMonths"] = df.apply(compute_months, axis=1)
        df["CompetitionOpenTimeMonths"] = np.log1p(df["CompetitionOpenTimeMonths"])
    else:
        df["CompetitionOpenTimeMonths"] = 0

    # --- Encode categorical ---
    if label_encoders:
        for col in ["StateHoliday", "Assortment", "StoreType", "Store"]:
            if col in df.columns and col in label_encoders:
                df[col] = label_encoders[col].transform(df[col].astype(str))

    # --- Drop unused ---
    drop_cols = ["Date", "Customers", "PromoInterval",
                 "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                 "MonthStr", "Promo2", "CompetitionOpenSinceDate"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # --- Ensure column order ---
    df = df[FEATURE_COLUMNS]

    return df.values


# ---------------- INPUT FN ---------------- #
def input_fn(request_body, request_content_type):
    try:
        print(f"🔹 input_fn received content_type={request_content_type}", flush=True)
        if request_content_type == "application/json":
            body = json.loads(request_body)
            data = preprocess(body)
        elif request_content_type == "text/csv":
            # for debugging or batch CSV input
            data = np.array([list(map(float, request_body.split(",")))])
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")

        print(f"✅ Parsed/preprocessed input shape: {data.shape}", flush=True)
        return data
    except Exception as e:
        print("❌ Error in input_fn:", e, flush=True)
        traceback.print_exc(file=sys.stdout)
        raise


# ---------------- PREDICT FN ---------------- #
def predict_fn(input_data, model):
    try:
        print(f"🔹 Running prediction on shape: {input_data.shape}", flush=True)
        prediction = model.predict(input_data)
        print(f"✅ Prediction done, shape: {prediction.shape}", flush=True)
        return prediction
    except Exception as e:
        print("❌ Error in predict_fn:", e, flush=True)
        traceback.print_exc(file=sys.stdout)
        raise


# ---------------- OUTPUT FN ----------------- #
def output_fn(prediction, content_type):
    try:
        print(f"🔹 Serializing prediction, content_type={content_type}", flush=True)
        if content_type == "application/json":
            return json.dumps(prediction.tolist())
        elif content_type == "text/csv":
            return ",".join(str(x) for x in prediction)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
    except Exception as e:
        print("❌ Error in output_fn:", e, flush=True)
        traceback.print_exc(file=sys.stdout)
        raise
