# test_endpoints.py
import boto3

# SageMaker runtime client
runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

# Replace with your actual endpoint name
endpoint_name = "rossmann-rf-endpoint-by-abdullah-shahzad"

# A single row of features (must exactly match training feature count & order)
payload = "1,0,0,2,1,0,2,6.215,2015,7,27,15,0,0,0,2.302"

# Invoke endpoint
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="text/csv",
    Body=payload
)

# Read prediction
result = response["Body"].read().decode("utf-8")
print("🎯 Prediction:", result)
