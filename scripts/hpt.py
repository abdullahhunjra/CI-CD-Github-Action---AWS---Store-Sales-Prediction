import os, json, joblib, boto3
import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ---- S3 Config ----
BUCKET = "rossmann-sales-bucket"
KEY = "rossmann-processed/df_train.csv"
s3 = boto3.client("s3")
obj = s3.get_object(Bucket=BUCKET, Key=KEY)
df_train = pd.read_csv(obj["Body"])

# ---- Parse SageMaker hyperparameters ----
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--min_samples_split', type=int, default=2)
parser.add_argument('--min_samples_leaf', type=int, default=1)
parser.add_argument('--max_features', type=str, default='auto')
args = parser.parse_args()

# ---- Prepare Data ----
X = df_train.drop(columns=['Sales'])
y = df_train['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Train Model ----
rf = RandomForestRegressor(
    n_estimators=args.n_estimators,
    min_samples_split=args.min_samples_split,
    min_samples_leaf=args.min_samples_leaf,
    max_features=args.max_features,
    random_state=42
)

rf.fit(X_train, y_train)
preds = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

# ---- Log Metric (SageMaker needs this!) ----
print(f"validation:RMSE {rmse:.4f}")

# ---- Save model ----
model_path = "/opt/ml/model"
os.makedirs(model_path, exist_ok=True)
joblib.dump(rf, os.path.join(model_path, "model.joblib"))


# Replace both print lines with:
print(f"validation:rmse {rmse:.4f}")


