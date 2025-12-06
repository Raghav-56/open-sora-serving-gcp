# =============================================================================
# Open-Sora Vertex AI Full Deployment Script
# =============================================================================
# This script handles the complete deployment of Open-Sora to Vertex AI
# Run from the service/ directory using PowerShell
# =============================================================================

param(
    [switch]$SkipBuild,
    [switch]$SkipUpload,
    [switch]$SkipEndpoint,
    [switch]$SkipDeploy
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Open-Sora Vertex AI Deployment" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Load configuration from .env file
. "$PSScriptRoot\myconfig.ps1"

# Display configuration
Write-Host "[Configuration]" -ForegroundColor Yellow
Write-Host "  PROJECT_ID     : $env:PROJECT_ID"
Write-Host "  REGION         : $env:REGION"
Write-Host "  IMAGE_URI      : $env:IMAGE_URI"
Write-Host "  MODEL_NAME     : $env:MODEL_NAME"
Write-Host "  ENDPOINT_NAME  : $env:ENDPOINT_NAME"
Write-Host "  MACHINE_TYPE   : $env:MACHINE_TYPE"
Write-Host "  ACCELERATOR    : $env:ACCELERATOR_TYPE"
Write-Host "  WEIGHT_BUCKET  : $env:WEIGHT_BUCKET"
Write-Host ""

# Confirm before proceeding
$confirm = Read-Host "Proceed with deployment? (y/n)"
if ($confirm -ne 'y') {
    Write-Host "Deployment cancelled." -ForegroundColor Red
    exit 0
}

# -----------------------------------------------------------------------------
# Step 0: Prerequisites Check
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "[Step 0] Prerequisites Check" -ForegroundColor Green
Write-Host "---------------------------------------------"

# Check gcloud is installed and authenticated
$account = $null
try {
    $account = gcloud auth list --filter="status:ACTIVE" --format="value(account)" 2>$null
}
catch {
    Write-Host "  [X] gcloud CLI not found" -ForegroundColor Red
    Write-Host "  Install: https://cloud.google.com/sdk/docs/install" -ForegroundColor Yellow
    exit 1
}

if ($account) {
    Write-Host "  [OK] Authenticated as: $account" -ForegroundColor Green
}
else {
    Write-Host "  [X] Not authenticated to gcloud" -ForegroundColor Red
    Write-Host "  Run: gcloud auth login" -ForegroundColor Yellow
    exit 1
}

# Set the project
Write-Host "  Setting project to $env:PROJECT_ID..."
gcloud config set project $env:PROJECT_ID

# Configure Docker authentication
Write-Host "  Configuring Docker authentication..."
gcloud auth configure-docker "$($env:REGION)-docker.pkg.dev" --quiet

# -----------------------------------------------------------------------------
# Step 1: Create Artifact Repository (if not exists)
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "[Step 1] Artifact Registry Setup" -ForegroundColor Green
Write-Host "---------------------------------------------"

$repoExists = gcloud artifacts repositories describe $env:REPOSITORY --location=$env:REGION --project=$env:PROJECT_ID 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Creating repository: $env:REPOSITORY..."
    gcloud artifacts repositories create $env:REPOSITORY `
        --repository-format=docker `
        --location=$env:REGION `
        --description="Open-Sora v2 API" `
        --project=$env:PROJECT_ID
    Write-Host "  [OK] Repository created" -ForegroundColor Green
}
else {
    Write-Host "  [OK] Repository already exists" -ForegroundColor Green
}

# -----------------------------------------------------------------------------
# Step 2: Build Docker Image (Cloud Build)
# -----------------------------------------------------------------------------
if (-not $SkipBuild) {
    Write-Host ""
    Write-Host "[Step 2] Building Docker Image" -ForegroundColor Green
    Write-Host "---------------------------------------------"
    Write-Host "  Using Cloud Build (this takes 15-20 minutes)..."
    Write-Host "  Image: $env:IMAGE_URI"
    Write-Host ""
    
    gcloud builds submit `
        --region=$env:REGION `
        --tag $env:IMAGE_URI `
        --timeout=3600s `
        --machine-type=e2-highcpu-32 `
        --project=$env:PROJECT_ID `
        .
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [X] Build failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "  [OK] Image built and pushed" -ForegroundColor Green
}
else {
    Write-Host ""
    Write-Host "[Step 2] Skipping build (SkipBuild flag set)" -ForegroundColor Yellow
}

# -----------------------------------------------------------------------------
# Step 3: Upload Model to Vertex AI
# -----------------------------------------------------------------------------
if (-not $SkipUpload) {
    Write-Host ""
    Write-Host "[Step 3] Uploading Model to Vertex AI" -ForegroundColor Green
    Write-Host "---------------------------------------------"
    
    # Check if model already exists
    $existingModel = gcloud ai models list `
        --region=$env:REGION `
        --filter="displayName:$($env:MODEL_NAME)" `
        --format="value(name)" `
        --project=$env:PROJECT_ID 2>$null | Select-Object -First 1
    
    $shouldUpload = $true
    
    if ($existingModel) {
        Write-Host "  Model '$env:MODEL_NAME' already exists." -ForegroundColor Yellow
        $overwrite = Read-Host "  Delete and re-upload? (y/n)"
        if ($overwrite -eq 'y') {
            Write-Host "  Deleting existing model..."
            gcloud ai models delete $existingModel --region=$env:REGION --project=$env:PROJECT_ID --quiet
        }
        else {
            Write-Host "  Using existing model." -ForegroundColor Yellow
            $shouldUpload = $false
        }
    }
    
    # Upload the model
    if ($shouldUpload) {
        Write-Host "  Uploading model..."
        gcloud ai models upload `
            --region=$env:REGION `
            --display-name=$env:MODEL_NAME `
            --container-image-uri=$env:IMAGE_URI `
            --container-health-route=/health `
            --container-predict-route=/predict `
            --container-ports=8080 `
            --container-env-vars="WEIGHT_BUCKET=$($env:WEIGHT_BUCKET),WEIGHT_PREFIX=$($env:WEIGHT_PREFIX),MODEL_PATH=$($env:MODEL_PATH),PORT=$($env:PORT),JOB_RETENTION_SECONDS=$($env:JOB_RETENTION_SECONDS),MAX_COMPLETED_JOBS=$($env:MAX_COMPLETED_JOBS),DEFAULT_RESOLUTION=$($env:DEFAULT_RESOLUTION),DEFAULT_NUM_FRAMES=$($env:DEFAULT_NUM_FRAMES),DEFAULT_ASPECT_RATIO=$($env:DEFAULT_ASPECT_RATIO),GENERATION_TIMEOUT=$($env:GENERATION_TIMEOUT)" `
            --project=$env:PROJECT_ID
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [X] Model upload failed!" -ForegroundColor Red
            exit 1
        }
        Write-Host "  [OK] Model uploaded" -ForegroundColor Green
    }
}
else {
    Write-Host ""
    Write-Host "[Step 3] Skipping model upload (SkipUpload flag set)" -ForegroundColor Yellow
}

# -----------------------------------------------------------------------------
# Step 4: Create Endpoint (if not exists)
# -----------------------------------------------------------------------------
if (-not $SkipEndpoint) {
    Write-Host ""
    Write-Host "[Step 4] Creating Endpoint" -ForegroundColor Green
    Write-Host "---------------------------------------------"
    
    $existingEndpoint = gcloud ai endpoints list `
        --region=$env:REGION `
        --filter="displayName:$($env:ENDPOINT_NAME)" `
        --format="value(name)" `
        --project=$env:PROJECT_ID 2>$null | Select-Object -First 1
    
    if ($existingEndpoint) {
        Write-Host "  [OK] Endpoint already exists" -ForegroundColor Green
        $env:ENDPOINT_ID = ($existingEndpoint -split "/")[-1]
    }
    else {
        Write-Host "  Creating endpoint: $env:ENDPOINT_NAME..."
        gcloud ai endpoints create `
            --region=$env:REGION `
            --display-name=$env:ENDPOINT_NAME `
            --project=$env:PROJECT_ID
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [X] Endpoint creation failed!" -ForegroundColor Red
            exit 1
        }
        
        # Get the endpoint ID
        $env:ENDPOINT_ID = gcloud ai endpoints list `
            --region=$env:REGION `
            --filter="displayName:$($env:ENDPOINT_NAME)" `
            --format="value(name)" `
            --project=$env:PROJECT_ID | Select-Object -First 1
        $env:ENDPOINT_ID = ($env:ENDPOINT_ID -split "/")[-1]
        
        Write-Host "  [OK] Endpoint created: $env:ENDPOINT_ID" -ForegroundColor Green
    }
}
else {
    Write-Host ""
    Write-Host "[Step 4] Skipping endpoint creation (SkipEndpoint flag set)" -ForegroundColor Yellow
    
    # Still need to get endpoint ID
    $existingEndpoint = gcloud ai endpoints list `
        --region=$env:REGION `
        --filter="displayName:$($env:ENDPOINT_NAME)" `
        --format="value(name)" `
        --project=$env:PROJECT_ID 2>$null | Select-Object -First 1
    $env:ENDPOINT_ID = ($existingEndpoint -split "/")[-1]
}

# -----------------------------------------------------------------------------
# Step 5: Deploy Model to Endpoint
# -----------------------------------------------------------------------------
if (-not $SkipDeploy) {
    Write-Host ""
    Write-Host "[Step 5] Deploying Model to Endpoint" -ForegroundColor Green
    Write-Host "---------------------------------------------"
    
    # Get model ID
    $env:MODEL_ID = gcloud ai models list `
        --region=$env:REGION `
        --filter="displayName:$($env:MODEL_NAME)" `
        --format="value(name)" `
        --project=$env:PROJECT_ID | Select-Object -First 1
    $env:MODEL_ID = ($env:MODEL_ID -split "/")[-1]
    
    Write-Host "  Model ID   : $env:MODEL_ID"
    Write-Host "  Endpoint ID: $env:ENDPOINT_ID"
    Write-Host ""
    Write-Host "  Deploying with:"
    Write-Host "    - Machine: $env:MACHINE_TYPE"
    Write-Host "    - GPU: $env:ACCELERATOR_TYPE"
    Write-Host "    - Replicas: 1"
    Write-Host ""
    Write-Host "  This will take 15-30 minutes..." -ForegroundColor Yellow
    
    gcloud ai endpoints deploy-model $env:ENDPOINT_ID `
        --region=$env:REGION `
        --model=$env:MODEL_ID `
        --display-name="opensora-deployment-$($env:TAG)" `
        --machine-type=$env:MACHINE_TYPE `
        --accelerator="type=$($env:ACCELERATOR_TYPE),count=1" `
        --service-account=$env:SERVICE_ACCOUNT `
        --min-replica-count=1 `
        --max-replica-count=1 `
        --project=$env:PROJECT_ID
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [X] Deployment failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "  [OK] Model deployed successfully!" -ForegroundColor Green
}
else {
    Write-Host ""
    Write-Host "[Step 5] Skipping deployment (SkipDeploy flag set)" -ForegroundColor Yellow
}

# -----------------------------------------------------------------------------
# Deployment Complete
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Deployment Complete!" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Endpoint ID: $env:ENDPOINT_ID"
Write-Host ""
Write-Host "  API URL:" -ForegroundColor Yellow
$apiUrl = "https://$($env:REGION)-aiplatform.googleapis.com/v1/projects/$($env:PROJECT_ID)/locations/$($env:REGION)/endpoints/$($env:ENDPOINT_ID):rawPredict"
Write-Host "  $apiUrl"
Write-Host ""
Write-Host "  Test command:" -ForegroundColor Yellow
Write-Host "  gcloud auth print-access-token | Set-Variable token"
Write-Host "  Invoke-RestMethod -Uri '$apiUrl' -Method POST -Headers @{Authorization='Bearer ' + `$token} -ContentType 'application/json' -Body '{`"prompt`":`"A cat playing piano`",`"resolution`":`"256px`",`"num_frames`":49}'"
Write-Host ""
Write-Host "  Monitor logs:" -ForegroundColor Yellow
Write-Host "  gcloud logging read 'resource.type=aiplatform.googleapis.com/Endpoint AND resource.labels.endpoint_id=$($env:ENDPOINT_ID)' --limit=50"
Write-Host ""
