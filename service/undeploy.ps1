# =============================================================================
# Undeploy and Cleanup Vertex AI Resources
# =============================================================================

param(
    [switch]$DeleteModel,
    [switch]$DeleteEndpoint,
    [switch]$Force
)

# Load configuration
. "$PSScriptRoot\myconfig.ps1"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Open-Sora Vertex AI Cleanup" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Get IDs
$env:ENDPOINT_ID = gcloud ai endpoints list `
    --region=$env:REGION `
    --filter="displayName:$($env:ENDPOINT_NAME)" `
    --format="value(name)" `
    --project=$env:PROJECT_ID 2>$null | Select-Object -First 1

$env:MODEL_ID = gcloud ai models list `
    --region=$env:REGION `
    --filter="displayName:$($env:MODEL_NAME)" `
    --format="value(name)" `
    --project=$env:PROJECT_ID 2>$null | Select-Object -First 1

if ($env:ENDPOINT_ID) {
    $env:ENDPOINT_ID = ($env:ENDPOINT_ID -split "/")[-1]
    Write-Host "  Found Endpoint: $env:ENDPOINT_ID"
}
if ($env:MODEL_ID) {
    $env:MODEL_ID = ($env:MODEL_ID -split "/")[-1]
    Write-Host "  Found Model: $env:MODEL_ID"
}

Write-Host ""

if (-not $Force) {
    Write-Host "This will:" -ForegroundColor Yellow
    Write-Host "  1. Undeploy the model from the endpoint"
    if ($DeleteEndpoint) { Write-Host "  2. Delete the endpoint" -ForegroundColor Red }
    if ($DeleteModel) { Write-Host "  3. Delete the model" -ForegroundColor Red }
    Write-Host ""
    $confirm = Read-Host "Proceed? (y/n)"
    if ($confirm -ne 'y') {
        Write-Host "Cancelled." -ForegroundColor Red
        exit 0
    }
}

# Step 1: Undeploy model from endpoint
if ($env:ENDPOINT_ID) {
    Write-Host ""
    Write-Host "[Step 1] Undeploying model from endpoint..." -ForegroundColor Green
    
    # Get deployed model ID
    $deployedModelId = gcloud ai endpoints describe $env:ENDPOINT_ID `
        --region=$env:REGION `
        --project=$env:PROJECT_ID `
        --format="value(deployedModels[0].id)" 2>$null
    
    if ($deployedModelId) {
        Write-Host "  Undeploying: $deployedModelId"
        gcloud ai endpoints undeploy-model $env:ENDPOINT_ID `
            --region=$env:REGION `
            --deployed-model-id=$deployedModelId `
            --project=$env:PROJECT_ID
        Write-Host "  ✓ Model undeployed" -ForegroundColor Green
    } else {
        Write-Host "  No deployed models found" -ForegroundColor Yellow
    }
}

# Step 2: Delete endpoint (optional)
if ($DeleteEndpoint -and $env:ENDPOINT_ID) {
    Write-Host ""
    Write-Host "[Step 2] Deleting endpoint..." -ForegroundColor Green
    gcloud ai endpoints delete $env:ENDPOINT_ID `
        --region=$env:REGION `
        --project=$env:PROJECT_ID `
        --quiet
    Write-Host "  ✓ Endpoint deleted" -ForegroundColor Green
}

# Step 3: Delete model (optional)
if ($DeleteModel -and $env:MODEL_ID) {
    Write-Host ""
    Write-Host "[Step 3] Deleting model..." -ForegroundColor Green
    gcloud ai models delete $env:MODEL_ID `
        --region=$env:REGION `
        --project=$env:PROJECT_ID `
        --quiet
    Write-Host "  ✓ Model deleted" -ForegroundColor Green
}

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Cleanup Complete" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Note: Docker images remain in Artifact Registry."
Write-Host "To delete: gcloud artifacts docker images delete $env:IMAGE_URI"

