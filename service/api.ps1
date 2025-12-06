# Get your final API URL
$env:API_URL = "https://$($env:REGION)-aiplatform.googleapis.com/v1/projects/$($env:PROJECT_ID)/locations/$($env:REGION)/endpoints/$($env:ENDPOINT_ID):rawPredict"



Write-Host "Your API URL is: $env:API_URL"