# cicd/deploy_model.py

import boto3
from sagemaker.sklearn.model import SKLearnModel
from sagemaker import Session
import sagemaker
from datetime import datetime

# --- Config ---
role = "arn:aws:iam::755283537318:role/telco-sagemaker-role"
bucket = "rossmann-sales-bucket"
region = "us-east-1"

# Replace with your best HPO job name
best_training_job_name = "rf-hpo-2025-08-21-19-05-05"

model_artifact = f"s3://{bucket}/{best_training_job_name}/output/model.tar.gz"
endpoint_name = f"rossmann-endpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# --- Create SKLearnModel from artifact ---
sklearn_model = SKLearnModel(
    model_data=model_artifact,
    role=role,
    entry_point="predictor.py",
    framework_version="0.23-1",
    source_dir="inference/",
    sagemaker_session=Session()
)

# --- Deploy to endpoint ---
predictor = sklearn_model.deploy(
    instance_type="ml.m5.large",  # Use a smaller one if needed
    initial_instance_count=1,
    endpoint_name=endpoint_name
)

print(f"✅ Model deployed at endpoint: {endpoint_name}")
