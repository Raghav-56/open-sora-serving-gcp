# =============================================================================
# Check Vertex AI Deployment Status
# =============================================================================

# Load configuration
. "$PSScriptRoot\myconfig.ps1"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Open-Sora Deployment Status" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Check Model
Write-Host "[Models]" -ForegroundColor Yellow
gcloud ai models list `
    --region=$env:REGION `
    --filter="displayName~opensora" `
    --format="table(displayName,name,createTime)" `
    --project=$env:PROJECT_ID

Write-Host ""

# Check Endpoints
Write-Host "[Endpoints]" -ForegroundColor Yellow
gcloud ai endpoints list `
    --region=$env:REGION `
    --filter="displayName~opensora" `
    --format="table(displayName,name,deployedModels)" `
    --project=$env:PROJECT_ID

Write-Host ""

# Get Endpoint ID and show URL
$env:ENDPOINT_ID = gcloud ai endpoints list `
    --region=$env:REGION `
    --filter="displayName:$($env:ENDPOINT_NAME)" `
    --format="value(name)" `
    --project=$env:PROJECT_ID | Select-Object -First 1

if ($env:ENDPOINT_ID) {
    $env:ENDPOINT_ID = ($env:ENDPOINT_ID -split "/")[-1]
    
    Write-Host "[API Information]" -ForegroundColor Yellow
    Write-Host "  Endpoint ID: $env:ENDPOINT_ID"
    Write-Host ""
    Write-Host "  API URL:" -ForegroundColor Green
    Write-Host "  https://$($env:REGION)-aiplatform.googleapis.com/v1/projects/$($env:PROJECT_ID)/locations/$($env:REGION)/endpoints/$($env:ENDPOINT_ID):rawPredict"
    Write-Host ""
    
    # Check deployed models on endpoint
    Write-Host "[Deployed Models on Endpoint]" -ForegroundColor Yellow
    gcloud ai endpoints describe $env:ENDPOINT_ID `
        --region=$env:REGION `
        --project=$env:PROJECT_ID `
        --format="yaml(deployedModels)"
}

Write-Host ""
Write-Host "[Recent Logs]" -ForegroundColor Yellow
Write-Host "  Run this to see logs:"
Write-Host "  gcloud logging read 'resource.type=aiplatform.googleapis.com/Endpoint' --limit=20 --project=$env:PROJECT_ID"

