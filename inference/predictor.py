import os
import joblib
import json
import numpy as np
import pandas as pd

# ---- Load Model ----
def model_fn(model_dir):
    """Load model from model_dir (SageMaker expects this)."""
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    print("✅ Model loaded successfully.")
    return model

# ---- Input Handler ----
def input_fn(request_body, request_content_type):
    """Deserialize request body to DataFrame."""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

# ---- Predict ----
def predict_fn(input_data, model):
    """Make prediction using the trained model."""
    preds = model.predict(input_data)
    return preds

# ---- Output Formatter ----
def output_fn(prediction, content_type):
    """Serialize prediction output."""
    if content_type == "application/json":
        return json.dumps({"prediction": prediction.tolist()})
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
