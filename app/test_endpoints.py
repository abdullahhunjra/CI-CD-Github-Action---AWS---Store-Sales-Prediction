import boto3

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

endpoint_name = "rossmann-rf-endpoint-v1"

# A row with exactly 16 features (from your X_train)
payload = "1,0,0,2,1,0,2,6.215,2015,7,27,15,0,0,0,2.302"

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="text/csv",
    Body=payload  # 👈 no JSON, just plain CSV string
)

result = response["Body"].read().decode("utf-8")
print("Prediction:", result)
