# Load .env file from parent directory
$envFile = Join-Path (Split-Path $PSScriptRoot -Parent) ".env"
if (-not (Test-Path $envFile)) {
    Write-Error "Missing .env file! Copy .env.example to .env and fill in your values."
    exit 1
}
Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]*?)\s*=\s*(.*)$') {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim().Trim('"').Trim("'")
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

# Validate required variables
$required = @("PROJECT_ID", "REGION", "WEIGHT_BUCKET")
foreach ($var in $required) {
    if (-not [Environment]::GetEnvironmentVariable($var, "Process")) {
        Write-Error "Missing required variable: $var in .env file"
        exit 1
    }
}

# GCP Project Configuration
if (-not $env:REPOSITORY) { $env:REPOSITORY = "opensora-serving-api" }
if (-not $env:IMAGE_NAME) { $env:IMAGE_NAME = "opensora-api" }
if (-not $env:TAG) { $env:TAG = "v1.0.0" }

$env:IMAGE_URI = "$($env:REGION)-docker.pkg.dev/$($env:PROJECT_ID)/$($env:REPOSITORY)/$($env:IMAGE_NAME):$($env:TAG)"

# Model Weights Configuration
if (-not $env:WEIGHT_PREFIX) { $env:WEIGHT_PREFIX = "ckpts/" }
if (-not $env:MODEL_PATH) { $env:MODEL_PATH = "/app/ckpts" }
if (-not $env:FORCE_DOWNLOAD) { $env:FORCE_DOWNLOAD = "false" }

# API Server Configuration (defaults, overridden by .env)
if (-not $env:PORT) { $env:PORT = "8080" }

# Job Manager Configuration (defaults, overridden by .env)
if (-not $env:JOB_RETENTION_SECONDS) { $env:JOB_RETENTION_SECONDS = "3600" }
if (-not $env:MAX_COMPLETED_JOBS) { $env:MAX_COMPLETED_JOBS = "100" }

# Video Generation Defaults (defaults, overridden by .env)
if (-not $env:DEFAULT_RESOLUTION) { $env:DEFAULT_RESOLUTION = "256px" }
if (-not $env:DEFAULT_NUM_FRAMES) { $env:DEFAULT_NUM_FRAMES = "49" }
if (-not $env:DEFAULT_ASPECT_RATIO) { $env:DEFAULT_ASPECT_RATIO = "16:9" }
if (-not $env:GENERATION_TIMEOUT) { $env:GENERATION_TIMEOUT = "1800" }

# Vertex AI Deployment (computed values)
$env:MODEL_NAME = "opensora-video-$($env:TAG)"
$env:ENDPOINT_NAME = "opensora-video-endpoint"
if (-not $env:MACHINE_TYPE) { $env:MACHINE_TYPE = "a2-ultragpu-1g" }
if (-not $env:ACCELERATOR_TYPE) { $env:ACCELERATOR_TYPE = "nvidia-a100-80gb" }
$env:SERVICE_ACCOUNT = "ml-model-serving@$($env:PROJECT_ID).iam.gserviceaccount.com"


# Verify it worked
# Write-Output $env:IMAGE_URI
Write-Host $env:IMAGE_URI