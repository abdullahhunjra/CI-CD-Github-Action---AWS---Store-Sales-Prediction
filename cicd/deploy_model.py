import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# Role used for SageMaker execution
role = "arn:aws:iam::755283537318:role/telco-sagemaker-role"

# S3 path to your trained model
model_artifact = "s3://rossmann-sales-bucket/rf-hpo-output/rf-hpo-2025-08-21-19-05-05-007-496cb0b3/output/model.tar.gz"

# Create SageMaker session
sagemaker_session = sagemaker.Session()

# Create a SageMaker model
model = SKLearnModel(
    model_data=model_artifact,
    role=role,
    framework_version="0.23-1",  # Match the training version
    sagemaker_session=sagemaker_session
)

# Deploy the model to an endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="rossmann-rf-endpoint"  # Choose your endpoint name
)

print(f"✅ Model deployed successfully! Endpoint name: {predictor.endpoint_name}")
