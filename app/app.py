import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

# ---------------- PATHS (update for your system) ---------------- #
MODEL_PATH = "/Users/abdullahhanjra/Downloads/model.joblib"
ENCODER_PATH = "/Users/abdullahhanjra/Downloads/label_encoders.pkl"

# ---------------- FEATURE ORDER ---------------- #
FEATURE_COLUMNS = [
    "Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday",
    "StoreType", "Assortment", "CompetitionDistance", "Year", "Month",
    "WeekOfYear", "Day", "IsWeekend", "IsPromoMonth", "Promo2Active",
    "CompetitionOpenTimeMonths"
]

# ---------------- LOAD ---------------- #
print("📥 Loading model + encoders...")
model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODER_PATH)
print("✅ Loaded successfully!")

# ---------------- PREPROCESS (copied from inference.py) ---------------- #
def preprocess(input_json):
    df = pd.DataFrame([input_json])

    # Date features
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    # Promo2 features
    df["PromoInterval"] = df.get("PromoInterval", "NoPromo").fillna("NoPromo")
    df["MonthStr"] = df["Date"].dt.strftime("%b")
    df["IsPromoMonth"] = df.apply(
        lambda row: int(row["MonthStr"] in row["PromoInterval"].split(","))
        if row["PromoInterval"] != "NoPromo" else 0, axis=1
    )
    df["Promo2Active"] = ((df.get("Promo2", 0) == 1) & (df["IsPromoMonth"] == 1)).astype(int)

    # Competition distance
    df["CompetitionDistance"] = np.log1p(df["CompetitionDistance"])

    # Competition open months
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

    # Encode categorical
    for col in ["StateHoliday", "Assortment", "StoreType", "Store"]:
        if col in df.columns and col in label_encoders:
            df[col] = label_encoders[col].transform(df[col].astype(str))

    # Drop unused
    drop_cols = ["Date", "Customers", "PromoInterval",
                 "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                 "MonthStr", "Promo2", "CompetitionOpenSinceDate"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Ensure order
    df = df[FEATURE_COLUMNS]

    return df.values

# ---------------- SAMPLE TEST ---------------- #
raw_input = {
    "Store": 1,
    "DayOfWeek": 1,
    "Date": "2015-07-27",
    "Promo": 1,
    "StateHoliday": "0",
    "SchoolHoliday": 1,
    "StoreType": "c",
    "Assortment": "a",
    "CompetitionDistance": 500.0,
    "CompetitionOpenSinceMonth": 9,
    "CompetitionOpenSinceYear": 2008,
    "Promo2": 1,
    "PromoInterval": "Jan,Apr,Jul,Oct"
}

print("📊 Raw input:", raw_input)
X = preprocess(raw_input)
print("✅ Preprocessed:", X)

pred = model.predict(X)
print("🎯 Prediction:", pred)
