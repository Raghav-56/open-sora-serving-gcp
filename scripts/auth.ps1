$env:ACCESS_TOKEN = gcloud auth print-access-token

Write-Host "Your access token is: $env:ACCESS_TOKEN"