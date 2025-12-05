$env:PROJECT_ID = "nannieai-website-stealth"
$env:REGION = "europe-west4"
$env:REPOSITORY = "opensora-serving-api"
$env:IMAGE_NAME = "opensora-api"
$env:TAG = "v1.0.0"
$env:IMAGE_URI = "$($env:REGION)-docker.pkg.dev/$($env:PROJECT_ID)/$($env:REPOSITORY)/$($env:IMAGE_NAME):$($env:TAG)"

$env:BUCKET_NAME = "nannie-opensora-weights-so"

# Verify it worked
Write-Output $env:IMAGE_URI