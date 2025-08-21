import os, json, joblib, boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor

# ---------------- CONFIG ----------------
BUCKET = "rossmann-sales-bucket"
PROC_PREFIX = "rossmann-processed/"
MODEL_PREFIX = "rossmann-trained-models/"
RESULTS_PREFIX = "rossmann-model-results/"
FEATURES_PREFIX = "rossmann-selected-features/"

os.makedirs("/tmp", exist_ok=True)
s3 = boto3.client("s3")

# ---------------- Load Data ----------------
def load_csv(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(obj["Body"])

X = load_csv(PROC_PREFIX + "X_train.csv")
y = load_csv(PROC_PREFIX + "y_train.csv").values.ravel()

# ---------------- Train/Test Split ----------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- Feature Selection ----------------
X_sample, _, y_sample, _ = train_test_split(X_train, y_train, train_size=0.25, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_sample, y_sample)
rf_importances = rf_model.feature_importances_
rf_selector = SelectFromModel(rf_model, threshold="median", prefit=True)
rf_selected = X_sample.columns[rf_selector.get_support()]

xgb_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_model.fit(X_sample.to_numpy(), y_sample)
xgb_importances = xgb_model.feature_importances_
xgb_selector = SelectFromModel(xgb_model, threshold="median", prefit=True)
xgb_selected = X_sample.columns[xgb_selector.get_support()]

combined_features = list(set(rf_selected) | set(xgb_selected))
common_features = list(set(rf_selected) & set(xgb_selected))

# Save selected features txt
with open("/tmp/selected_features.txt", "w") as f:
    f.write("Features selected by Random Forest:\n")
    f.write("\n".join(rf_selected) + "\n\n")
    f.write("Features selected by XGBoost:\n")
    f.write("\n".join(xgb_selected) + "\n\n")
    f.write("Combined (Union) Features:\n")
    f.write("\n".join(combined_features) + "\n\n")
    f.write("Common (Intersection) Features:\n")
    f.write("\n".join(common_features) + "\n")
s3.upload_file("/tmp/selected_features.txt", BUCKET, FEATURES_PREFIX + "selected_features.txt")

# Save plots
def save_plot(importances, columns, title, filename):
    sorted_idx = np.argsort(importances)
    plt.figure(figsize=(10, 4))
    plt.barh(np.array(columns)[sorted_idx], importances[sorted_idx], color='skyblue')
    plt.axvline(np.median(importances), color='red', linestyle='--', label='Median')
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.legend()
    plt.tight_layout()
    path = f"/tmp/{filename}"
    plt.savefig(path)
    s3.upload_file(path, BUCKET, FEATURES_PREFIX + filename)

save_plot(rf_importances, X_sample.columns, "Feature Importances - Random Forest", "rf_feature_importances.png")
save_plot(xgb_importances, X_sample.columns, "Feature Importances - XGBoost", "xgb_feature_importances.png")

# Save selected features to JSON for later use
with open("/tmp/selected_features.json", "w") as f:
    json.dump(combined_features, f)
s3.upload_file("/tmp/selected_features.json", BUCKET, FEATURES_PREFIX + "selected_features.json")

# ---------------- Define Models ----------------
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

results = {}

# ---------------- Training Loop ----------------
for name, model in models.items():
    print(f"\n📦 Training model: {name}")

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    results[name + "_all"] = {
        "RMSE": float(np.sqrt(mean_squared_error(y_val, preds))),
        "MAE": float(mean_absolute_error(y_val, preds)),
        "R2": float(r2_score(y_val, preds))
    }

    joblib.dump(model, f"/tmp/{name}_all.pkl")
    s3.upload_file(f"/tmp/{name}_all.pkl", BUCKET, f"{MODEL_PREFIX}{name}_all.pkl")

    model.fit(X_train[combined_features], y_train)
    preds_sel = model.predict(X_val[combined_features])
    results[name + "_selected"] = {
        "RMSE": float(np.sqrt(mean_squared_error(y_val, preds_sel))),
        "MAE": float(mean_absolute_error(y_val, preds_sel)),
        "R2": float(r2_score(y_val, preds_sel))
    }

    joblib.dump(model, f"/tmp/{name}_selected.pkl")
    s3.upload_file(f"/tmp/{name}_selected.pkl", BUCKET, f"{MODEL_PREFIX}{name}_selected.pkl")

# ---------------- Save Evaluation Report ----------------
with open("/tmp/model_results.json", "w") as f:
    json.dump(results, f, indent=4)
s3.upload_file("/tmp/model_results.json", BUCKET, RESULTS_PREFIX + "model_results.json")

with open("/tmp/model_performance_report.txt", "w") as f:
    for model_name, metrics in results.items():
        f.write(f"===== {model_name} =====\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("\n")
s3.upload_file("/tmp/model_performance_report.txt", BUCKET, RESULTS_PREFIX + "model_performance_report.txt")

print("✅ Training complete. Models and artifacts uploaded to S3.")

