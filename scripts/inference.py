import joblib
import os
import numpy as np
import json
import sys
import traceback

def model_fn(model_dir):
    try:
        model_path = os.path.join(model_dir, "model.joblib")
        print(f"🔹 Attempting to load model from: {model_path}", flush=True)
        model = joblib.load(model_path)
        print("✅ Model loaded successfully", flush=True)
        return model
    except Exception as e:
        print("❌ Model load failed:", e, flush=True)
        traceback.print_exc(file=sys.stdout)
        raise

def input_fn(request_body, request_content_type):
    try:
        print(f"🔹 input_fn received content_type={request_content_type}", flush=True)
        if request_content_type == "text/csv":
            data = np.array([list(map(float, request_body.split(",")))])
        elif request_content_type == "application/json":
            data = np.array(json.loads(request_body))
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
        print(f"✅ Parsed input shape: {data.shape}", flush=True)
        return data
    except Exception as e:
        print("❌ Error in input_fn:", e, flush=True)
        traceback.print_exc(file=sys.stdout)
        raise

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

def output_fn(prediction, content_type):
    try:
        print(f"🔹 Serializing prediction, content_type={content_type}", flush=True)
        if content_type == "text/csv":
            return ",".join(str(x) for x in prediction)
        elif content_type == "application/json":
            return json.dumps(prediction.tolist())
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
    except Exception as e:
        print("❌ Error in output_fn:", e, flush=True)
        traceback.print_exc(file=sys.stdout)
        raise
