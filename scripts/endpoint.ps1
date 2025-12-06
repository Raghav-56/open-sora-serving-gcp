$NewEndpoint = gcloud ai endpoints create --region=$env:REGION --display-name="opensora-v2-endpoint" --format="value(name)"
$NewEndpointId = $NewEndpoint.Split("/")[-1]

Write-Host "Created New Endpoint with ID: $NewEndpointId"