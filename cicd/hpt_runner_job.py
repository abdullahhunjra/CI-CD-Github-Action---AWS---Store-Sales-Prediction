import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, CategoricalParameter
from datetime import datetime

role = "arn:aws:iam::755283537318:role/telco-sagemaker-role"
bucket = "rossmann-sales-bucket"
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
job_name = f"rf-hpo-{timestamp}"

# Estimator
estimator = SKLearn(
    entry_point="hpt.py",                   
    source_dir="scripts/",
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    framework_version="0.23-1",
    py_version="py3",
    dependencies=["requirements.txt"],
    base_job_name="rf-hpo",
    output_path=f"s3://{bucket}/rf-hpo-output"
)

# Define HPO search space
hyperparameter_ranges = {
    "n_estimators": IntegerParameter(100, 200),
    "min_samples_split": IntegerParameter(2, 5),
    "min_samples_leaf": IntegerParameter(1, 2),
    "max_features": CategoricalParameter(["sqrt", "log2", "auto"]),
}

# Tuner with metric definition
tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name="validation:rmse",  # must match printed log line
    objective_type="Minimize",
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=10,
    max_parallel_jobs=2,
    metric_definitions=[
        {
            "Name": "validation:rmse",
            "Regex": "Best Random Forest RMSE: ([0-9\\.]+)"
        }
    ]
)

# Start tuning job
tuner.fit(job_name=job_name)
