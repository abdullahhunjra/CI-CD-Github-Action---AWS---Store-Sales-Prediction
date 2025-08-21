import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from datetime import datetime

role = "arn:aws:iam::755283537318:role/telco-sagemaker-role"
bucket = "rossmann-sales-bucket"

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
job_name = f"rossmann-training-{timestamp}"


sklearn_estimator = SKLearn(
    entry_point="train.py",
    source_dir="scripts/",
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="0.23-1",
    py_version="py3",
    dependencies=["requirements.txt"]
)




# 🔁 Run Training Job
sklearn_estimator.fit(job_name=job_name)
