import joblib
import os
import numpy as np
import json

def model_fn(model_dir):
    try:
        model_path = os.path.join(model_dir, "model.joblib")
        print(f"🔹 Attempting to load model from: {model_path}")
        model = joblib.load(model_path)
        print("✅ Model loaded successfully")
        return model
    except Exception as e:
        print("❌ Model load failed:", e)
        raise


def input_fn(request_body, request_content_type):
    """Deserialize input data"""
    if request_content_type == "text/csv":
        data = np.array([list(map(float, request_body.split(",")))])
        return data
    elif request_content_type == "application/json":
        return np.array(json.loads(request_body))
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make prediction using the loaded model"""
    return model.predict(input_data)

def output_fn(prediction, content_type):
    """Serialize prediction back"""
    if content_type == "text/csv":
        return ",".join(str(x) for x in prediction)
    elif content_type == "application/json":
        return json.dumps(prediction.tolist())
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
