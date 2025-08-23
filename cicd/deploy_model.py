import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from datetime import datetime

# --- CONFIG ---
role = "arn:aws:iam::755283537318:role/telco-sagemaker-role"
region = "us-east-1"
tuning_job_name = "rf-hpo-2025-08-21-19-05-05"

session = sagemaker.Session()
sm_client = boto3.client("sagemaker", region_name=region)

# --- Step 1: Get best training job ---
tuning_info = sm_client.describe_hyper_parameter_tuning_job(
    HyperParameterTuningJobName=tuning_job_name
)
best_job_name = tuning_info["BestTrainingJob"]["TrainingJobName"]
print(f"✅ Best training job: {best_job_name}")

# --- Step 2: Get model artifact ---
job_info = sm_client.describe_training_job(TrainingJobName=best_job_name)
model_artifact = job_info["ModelArtifacts"]["S3ModelArtifacts"]
print(f"✅ Model artifact: {model_artifact}")

# --- Step 3: Use SKLearnModel for deployment ---
model = SKLearnModel(
    model_data=model_artifact,
    role=role,
    framework_version="0.23-1",  # ✅ Must match training version
    sagemaker_session=session
)

# --- Step 4: Deploy endpoint ---
endpoint_name = f"rossmann-endpoint-{datetime.now().strftime('%Y%m%d%H%M%S')}"
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name
)

print(f"✅ Model deployed at endpoint: {endpoint_name}")
