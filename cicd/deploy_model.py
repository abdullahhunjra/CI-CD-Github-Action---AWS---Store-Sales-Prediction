import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# SageMaker execution role
role = "arn:aws:iam::755283537318:role/telco-sagemaker-role"

# S3 path to your trained model
model_artifact = "s3://rossmann-sales-bucket/rf-hpo-output/rf-hpo-2025-08-21-19-05-05-007-496cb0b3/output/model.tar.gz"

# Create a SageMaker session
sagemaker_session = sagemaker.Session()

# Create a SageMaker model object
model = SKLearnModel(
    model_data=model_artifact,
    role=role,
    framework_version="0.23-1",  # Match training version
    sagemaker_session=sagemaker_session
)

# Deploy to endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",  # You can also use ml.m5.xlarge if needed
    endpoint_name="rossmann-rf-endpoint"
)

print(f"✅ Model deployed successfully! Endpoint name: {predictor.endpoint_name}")
