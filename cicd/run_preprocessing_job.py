# cicd/run_preprocessing_job.py

import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

# Update these as needed
role = "arn:aws:iam::755283537318:role/telco-sagemaker-role"  
bucket = "rossmann-sales-bucket"  # 🔄 Replace this

sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1", 
    role=role,
    instance_type="ml.t3.medium",
    instance_count=1,
    base_job_name="rossmann-preprocessing",
)

sklearn_processor.run(
    code="scripts/preprocess.py",
    inputs=[
        ProcessingInput(
            source=f"s3://{bucket}/rossmann-raw",
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/processed",
            destination=f"s3://{bucket}/rossmann-processed"
        ),
        ProcessingOutput(
            source="/opt/ml/processing/artifacts",
            destination=f"s3://{bucket}/rossmann-artifacts"
        ),
    ],
)

print("✅ Rossmann preprocessing job launched on SageMaker")
