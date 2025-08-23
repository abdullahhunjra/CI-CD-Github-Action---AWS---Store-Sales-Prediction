import joblib
import os
import numpy as np

def model_fn(model_dir):
    # Explicitly load model.joblib
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)

def input_fn(request_body, request_content_type):
    # Assume CSV input
    if request_content_type == "text/csv":
        return np.array([list(map(float, request_body.split(",")))])
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, content_type):
    return ",".join(str(x) for x in prediction)
