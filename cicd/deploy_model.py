import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from datetime import datetime

# Configuration
role = "arn:aws:iam::755283537318:role/telco-sagemaker-role"
region = "us-east-1"
tuning_job_name = "rf-hpo-2025-08-21-19-05-05"  # Replace with your actual HPO job name

session = sagemaker.Session()
sm_client = session.sagemaker_client

# Step 1: Get best training job from tuning job
hpo_result = sm_client.describe_hyper_parameter_tuning_job(
    HyperParameterTuningJobName=tuning_job_name
)
best_job_name = hpo_result["BestTrainingJob"]["TrainingJobName"]

# Step 2: Get model artifact path from that training job
training_job_info = sm_client.describe_training_job(TrainingJobName=best_job_name)
model_artifact = training_job_info["ModelArtifacts"]["S3ModelArtifacts"]

# Step 3: Deploy that model
sklearn_model = SKLearnModel(
    model_data=model_artifact,
    role=role,
    entry_point="inference/predictor.py",  # Make sure this exists
    framework_version="0.23-1",
    py_version="py3",
    sagemaker_session=session
)

endpoint_name = f"rossmann-endpoint-{datetime.now().strftime('%Y%m%d%H%M%S')}"

predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name
)

print(f"✅ Model deployed to endpoint: {endpoint_name}")
