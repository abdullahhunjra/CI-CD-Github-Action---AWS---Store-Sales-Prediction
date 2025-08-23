import os
import joblib
import json
import numpy as np
import pandas as pd

# -------------------------
# Load the trained model
# -------------------------
def model_fn(model_dir):
    """Loads the trained model from model_dir."""
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    print("✅ Model loaded successfully.")
    return model

# -------------------------
# Handle incoming request
# -------------------------
def input_fn(request_body, request_content_type):
    """Parses incoming request JSON into a DataFrame."""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        # Allow both dict & list inputs
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        return df
    else:
        raise ValueError(f"❌ Unsupported content type: {request_content_type}")

# -------------------------
# Make prediction
# -------------------------
def predict_fn(input_data, model):
    """Use the model to make predictions."""
    preds = model.predict(input_data)
    return preds

# -------------------------
# Format the output
# -------------------------
def output_fn(prediction, accept):
    """Formats prediction response into JSON."""
    if accept == "application/json":
        return json.dumps({"prediction": prediction.tolist()})
    else:
        raise ValueError(f"❌ Unsupported accept type: {accept}")
