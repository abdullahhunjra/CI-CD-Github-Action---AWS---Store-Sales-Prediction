import os, json, joblib, boto3
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# ---------------- CONFIG ---------------- #
BUCKET = "rossmann-sales-bucket"  # 🔁 your S3 bucket
RAW_KEYS = {
    "train": "rossmann-raw/train.csv",
    "test": "rossmann-raw/test.csv",
    "store": "rossmann-raw/store.csv"
}
PROC_PREFIX = "rossmann-processed/"
ART_PREFIX = "rossmann-artifacts/"

s3 = boto3.client("s3")

def read_csv_from_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj["Body"])

def upload_df(df, key):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=BUCKET, Key=key, Body=csv_buffer.getvalue())

# ---------------- LOAD DATA ---------------- #
print("📥 Loading data from S3...")
train_df = read_csv_from_s3(BUCKET, RAW_KEYS["train"])
test_df = read_csv_from_s3(BUCKET, RAW_KEYS["test"])
store_df = read_csv_from_s3(BUCKET, RAW_KEYS["store"])

# Merge store info
df_train = pd.merge(train_df, store_df, on="Store", how="left")
df_test = pd.merge(test_df, store_df, on="Store", how="left")

# ---------------- DATE FEATURES ---------------- #
def Date_column(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    return df

df_train = Date_column(df_train)
df_test = Date_column(df_test)

# ---------------- OPEN & PROMO2 ---------------- #
df_train = df_train[df_train["Open"] == 1].drop(columns=["Open"])
df_test = df_test[df_test["Open"] == 1].drop(columns=["Open"])

for df in [df_train, df_test]:
    df["PromoInterval"] = df["PromoInterval"].fillna("NoPromo")
    df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0).astype(int)
    df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0).astype(int)
    df["MonthStr"] = df["Date"].dt.strftime("%b")
    df["IsPromoMonth"] = df.apply(
        lambda row: int(row["MonthStr"] in row["PromoInterval"].split(","))
        if row["PromoInterval"] != "NoPromo" else 0, axis=1
    )
    df.drop(columns=["MonthStr"], inplace=True)
    df["Promo2Active"] = ((df["Promo2"] == 1) & (df["IsPromoMonth"] == 1)).astype(int)

# ---------------- COMPETITION DISTANCE ---------------- #
def preprocess_competition_distance(df):
    df = df[df["CompetitionDistance"].notna()]
    df["CompetitionDistance"] = np.log1p(df["CompetitionDistance"])

    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0).astype(int)
    df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(0).astype(int)

    df["CompetitionOpenSinceDate"] = df.apply(
        lambda row: datetime(
            year=row["CompetitionOpenSinceYear"],
            month=row["CompetitionOpenSinceMonth"],
            day=1
        ) if row["CompetitionOpenSinceYear"] > 0 and row["CompetitionOpenSinceMonth"] > 0
        else row["Date"], axis=1
    )

    df["CompetitionOpenTimeMonths"] = (
        (df["Date"].dt.year - df["CompetitionOpenSinceDate"].dt.year) * 12 +
        (df["Date"].dt.month - df["CompetitionOpenSinceDate"].dt.month)
    ).apply(lambda x: max(x, 0))

    df["CompetitionOpenTimeMonths"] = np.log1p(df["CompetitionOpenTimeMonths"])

    return df

df_train = preprocess_competition_distance(df_train)
df_test = preprocess_competition_distance(df_test)

# ---------------- DROP UNUSED COLUMNS ---------------- #
drop_cols = [
    "Date", "Customers",
    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
    "CompetitionOpenSinceDate"
]

df_train.drop(columns=drop_cols, inplace=True, errors="ignore")
df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], inplace=True)

# ---------------- ENCODING ---------------- #
df_train["StateHoliday"] = df_train["StateHoliday"].astype(str)
df_test["StateHoliday"] = df_test["StateHoliday"].astype(str)

label_encoders = {}
for col in ["StateHoliday", "Assortment", "StoreType", "Store"]:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])
    label_encoders[col] = le

# ---------------- SPLIT & UPLOAD ---------------- #
X_train = df_train.drop("Sales", axis=1)
y_train = df_train[["Sales"]]
X_test = df_test.copy()  # note: y_test is not available

print("📤 Uploading processed datasets to S3...")
upload_df(X_train, PROC_PREFIX + "X_train.csv")
upload_df(y_train, PROC_PREFIX + "y_train.csv")
upload_df(X_test, PROC_PREFIX + "X_test.csv")

# ---------------- SAVE ARTIFACTS ---------------- #
print("💾 Saving label encoders...")
os.makedirs("/tmp", exist_ok=True)
joblib.dump(label_encoders, "/tmp/label_encoders.pkl")
s3.upload_file("/tmp/label_encoders.pkl", BUCKET, ART_PREFIX + "label_encoders.pkl")

print("✅ Done preprocessing and uploading everything.")
