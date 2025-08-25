import boto3
import json

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

endpoint_name = "rossmann-rf-endpoint-ASH"

# Raw input row BEFORE preprocessing
raw_input = {
    "Store": 1,
    "DayOfWeek": 1,
    "Date": "2015-07-27",   # 👈 important, preprocessing extracts features from this
    "Promo": 0,
    "StateHoliday": "0",
    "SchoolHoliday": 0,
    "StoreType": "a",
    "Assortment": "a",
    "CompetitionDistance": 500.0,
    "CompetitionOpenSinceYear": 2013,
    "CompetitionOpenSinceMonth": 9,
    "Promo2": 1,
    "PromoInterval": "Feb,May,Aug,Nov"
}

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",  # 👈 must be JSON now
    Body=json.dumps(raw_input)
)

result = response["Body"].read().decode("utf-8")
print("Prediction:", result)
