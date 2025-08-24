import sagemaker
from sagemaker.sklearn.model import SKLearnModel

role = "arn:aws:iam::755283537318:role/telco-sagemaker-role"
sagemaker_session = sagemaker.Session()

# Best model artifact
model_artifact = (
    "s3://rossmann-sales-bucket/rf-hpo-output/"
    "rf-hpo-2025-08-21-19-05-05-007-496cb0b3/output/model.tar.gz"
)

endpoint_name = "rossmann-rf-endpoint-new-fully-final"

# Delete old endpoint if exists
try:
    sagemaker_session.delete_endpoint(endpoint_name)
except Exception as e:
    print("ℹ️ No existing endpoint to delete:", e)

# Create SageMaker model
model = SKLearnModel(
    entry_point="inference.py",
    source_dir="scripts",
    dependencies=["requirements.txt"],  # ✅ ensure right sklearn version
    model_data=model_artifact,
    role=role,
    framework_version="1.0-1",
    sagemaker_session=sagemaker_session
)

# Deploy
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large", 
    endpoint_name=endpoint_name
)

print(f"✅ Model deployed successfully! Endpoint name: {predictor.endpoint_name}")
