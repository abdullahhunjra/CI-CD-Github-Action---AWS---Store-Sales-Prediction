import sagemaker
from sagemaker.sklearn.model import SKLearnModel

role = "arn:aws:iam::755283537318:role/telco-sagemaker-role"
sagemaker_session = sagemaker.Session()

# Best model artifact
model_artifact = (
    "s3://rossmann-sales-bucket/rf-hpo-output/"
    "rf-hpo-2025-08-24-11-42-54-006-f2a2bddd/output/model.tar.gz"
)

endpoint_name = "rossmann-rf-endpoint-ASH"

# Delete old endpoint if exists
try:
    sagemaker_session.delete_endpoint(endpoint_name)
    print(f"ℹ️ Deleted old endpoint {endpoint_name}", flush=True)
except Exception as e:
    print(f"ℹ️ No existing endpoint to delete: {e}", flush=True)

# Create SageMaker model
model = SKLearnModel(
    entry_point="inference.py",
    source_dir="scripts",
    dependencies=["requirements.txt"],  # ensure sklearn 0.23 is installed
    model_data=model_artifact,
    role=role,
    framework_version="0.23-1",   # ✅ match training version
    py_version="py3",
    sagemaker_session=sagemaker_session
)

# Deploy
try:
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.2xlarge",
        endpoint_name=endpoint_name,
        wait=True
    )
    print(f"✅ Model deployed successfully! Endpoint name: {predictor.endpoint_name}", flush=True)
except Exception as e:
    print("❌ Deployment failed:", e, flush=True)
    raise
