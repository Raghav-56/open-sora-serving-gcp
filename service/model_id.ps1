# Get the Endpoint ID
$env:ENDPOINT_ID = gcloud ai endpoints list --region=$env:REGION --filter="displayName:$($env:ENDPOINT_NAME)" --format="value(name)" --project=$env:PROJECT_ID | Select-Object -First 1
$env:ENDPOINT_ID = ($env:ENDPOINT_ID -split "/")[-1]

# Get the Model ID
$env:MODEL_ID = gcloud ai models list --region=$env:REGION --filter="displayName:$($env:MODEL_NAME)" --format="value(name)" --project=$env:PROJECT_ID | Select-Object -First 1
$env:MODEL_ID = ($env:MODEL_ID -split "/")[-1]

Write-Host "Deploying Model: $env:MODEL_ID to Endpoint: $env:ENDPOINT_ID"