import sagemaker
from sagemaker.sklearn.model import SKLearnModel

role = "arn:aws:iam::755283537318:role/telco-sagemaker-role"
sagemaker_session = sagemaker.Session()

# Model artifact from training
model_artifact = (
    "s3://rossmann-sales-bucket/rf-hpo-output/"
    "rf-hpo-2025-08-24-11-42-54-006-f2a2bddd/output/model.tar.gz"
)

# Preprocessed test set (from your preprocessing step)
test_input = "s3://rossmann-sales-bucket/rossmann-processed/X_test.csv"

# Output bucket location for predictions
output_path = "s3://rossmann-sales-bucket/rossmann-batch-predictions/"

# Create SageMaker model (using your inference.py entry point)
model = SKLearnModel(
    entry_point="inference.py",
    source_dir="scripts",
    dependencies=["requirements.txt"],
    model_data=model_artifact,
    role=role,
    framework_version="0.23-1",
    py_version="py3",
    sagemaker_session=sagemaker_session
)

# Create Transformer object
transformer = model.transformer(
    instance_count=1,
    instance_type="ml.m5.2xlarge",
    strategy="SingleRecord",
    assemble_with="Line",
    output_path=output_path
)

# Launch batch transform job
transformer.transform(
    data=test_input,
    content_type="text/csv",
    split_type="Line"
)

transformer.wait()

print(f"✅ Batch transform complete! Predictions saved at: {output_path}")
