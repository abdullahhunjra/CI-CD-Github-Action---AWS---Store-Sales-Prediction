import boto3
import sagemaker
from sagemaker.model import Model
from datetime import datetime

# --- CONFIG ---
role = "arn:aws:iam::755283537318:role/telco-sagemaker-role"
region = "us-east-1"
bucket = "rossmann-sales-bucket"
output_prefix = "rf-hpo-output"
tuning_job_name = "rf-hpo-2025-08-21-19-05-05"

# --- SageMaker session and client ---
session = sagemaker.Session()
sm_client = boto3.client("sagemaker", region_name=region)

# --- Step 1: Get best training job ---
tuning_info = sm_client.describe_hyper_parameter_tuning_job(
    HyperParameterTuningJobName=tuning_job_name
)
best_job_name = tuning_info["BestTrainingJob"]["TrainingJobName"]
print(f"✅ Best training job: {best_job_name}")

# --- Step 2: Construct S3 model artifact path ---
model_artifact = f"s3://{bucket}/{output_prefix}/{best_job_name}/output/model.tar.gz"
print(f"✅ Constructed model path: {model_artifact}")

# --- Step 3: Get training image used by that job ---
training_info = sm_client.describe_training_job(TrainingJobName=best_job_name)
image_uri = training_info["AlgorithmSpecification"]["TrainingImage"]
print(f"✅ Training image URI: {image_uri}")

# --- Step 4: Register and deploy model ---
model = Model(
    image_uri=image_uri,
    model_data=model_artifact,
    role=role,
    sagemaker_session=session
)

endpoint_name = f"rossmann-store-sales-endpoints"
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name
)

print(f"✅ Model deployed to endpoint: {endpoint_name}")
