# Open-Sora v2 Deployment Guide

Complete guide for deploying Open-Sora v2 to Google Cloud Vertex AI.

Nannie AI - Proprietary System

!!! info "Configuration Reference"
    See [Configuration Guide](configuration.md) for detailed environment variable documentation.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Configuration](#configuration)
3. [Deployment Steps](#deployment-steps)
4. [Using the API](#using-the-api)

## Prerequisites

### Required Access

- Google Cloud Project with billing enabled
- Service Account with required permissions
- GCS Bucket for model weights (~50GB required)

### Upload Model Weights

```bash
# Clone Open-Sora repository
git clone https://github.com/Raghav-56/Open-Sora
cd Open-Sora

# Download weights from Hugging Face
uv sync
uv run huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir ./ckpts

# Upload to GCS
gsutil -m cp -r ./ckpts/* gs://YOUR_BUCKET/ckpts/
```

### Required IAM Permissions

The service account needs:

- `Storage Object Viewer` - Read model weights
- `Storage Object Creator` - Upload generated videos
- `Vertex AI User` - Deploy and manage models

### Local Requirements

- Linux x86_64 system (for Docker build)
- Docker with NVIDIA runtime
- gcloud CLI configured
- ~50GB disk space

## Configuration

Create a `.myconfig` file with your project settings:

```bash
# .myconfig - Project Configuration
export PROJECT_ID="nannieai-website-stealth"
export REGION="europe-west4"
export REPOSITORY="opensora-serving-api"
export IMAGE_NAME="opensora-api"
export TAG="v1.0.0"

# Derived values
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${TAG}"
export BUCKET_NAME="nannie-opensora-weights-so"

# Vertex AI
export MODEL_NAME="opensora-video-${TAG}"
export ENDPOINT_NAME="opensora-video-endpoint"
export MACHINE_TYPE="a3-highgpu-1g"
export SERVICE_ACCOUNT="ml-model-serving@${PROJECT_ID}.iam.gserviceaccount.com"
```

Load the configuration:

```bash
source .myconfig
```

## Deployment Steps

### Step 1: Authenticate

```bash
gcloud config set project $PROJECT_ID
gcloud auth configure-docker ${REGION}-docker.pkg.dev
```

### Step 2: Build Docker Image

```bash
cd service
docker build -t ${IMAGE_NAME}:${TAG} .
```

### Step 3: Push to Artifact Registry

```bash
# Create repository if needed
gcloud artifacts repositories create ${REPOSITORY} \
  --repository-format=docker \
  --location=${REGION} \
  --description="Open-Sora v2 API"

# Tag and push
docker tag ${IMAGE_NAME}:${TAG} ${IMAGE_URI}
docker push ${IMAGE_URI}
```

### Step 4: Upload Model to Vertex AI

```bash
gcloud ai models upload \
  --region=${REGION} \
  --display-name=${MODEL_NAME} \
  --container-image-uri=${IMAGE_URI} \
  --container-health-route=/health \
  --container-predict-route=/predict \
  --container-ports=8080 \
  --container-env-vars="WEIGHT_BUCKET=${BUCKET_NAME}"
```

### Step 5: Create Endpoint

```bash
gcloud ai endpoints create \
  --region=${REGION} \
  --display-name=${ENDPOINT_NAME}

# Get endpoint ID
ENDPOINT_ID=$(gcloud ai endpoints list \
  --region=${REGION} \
  --filter="displayName:${ENDPOINT_NAME}" \
  --format="value(name)" | head -1)
```

### Step 6: Deploy Model

```bash
MODEL_ID=$(gcloud ai models list \
  --region=${REGION} \
  --filter="displayName:${MODEL_NAME}" \
  --format="value(name)" | head -1)

gcloud ai endpoints deploy-model ${ENDPOINT_ID} \
  --region=${REGION} \
  --model=${MODEL_ID} \
  --display-name=opensora-deployment-${TAG} \
  --machine-type=${MACHINE_TYPE} \
  --min-replica-count=1 \
  --max-replica-count=1 \
  --accelerator=type=nvidia-h100-80gb,count=1 \
  --service-account=${SERVICE_ACCOUNT}
```

## Using the API

### Generate Video

```bash
ENDPOINT_URL="https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_ID}:rawPredict"

curl -X POST ${ENDPOINT_URL} \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing piano in a jazz club, cinematic lighting",
    "resolution": "256px",
    "num_frames": 49,
    "aspect_ratio": "16:9",
    "seed": 42,
    "output_bucket": "your-output-bucket"
  }'
```

### API Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `prompt` | string | required | Text description (up to 2000 chars) |
| `resolution` | `256px`, `768px` | `256px` | Video resolution |
| `num_frames` | 17, 33, 49, 65, 81, 97, 113, 129 | 49 | Frame count (4k+1 format) |
| `aspect_ratio` | `16:9`, `9:16`, `1:1`, `2.39:1` | `16:9` | Video aspect ratio |
| `motion_score` | 1-10 | `4` | Motion intensity (4-6 natural) |
| `seed` | integer | auto | For reproducibility |
| `output_bucket` | string | required | GCS bucket for output |

### Expected Generation Times (H100)

| Resolution | 49 frames | 97 frames |
|------------|-----------|-----------|
| 256px | ~2-3 min | ~4-6 min |
| 768px | ~10-15 min | ~20-30 min |

### Check Logs

```bash
gcloud logging read \
  "resource.type=aiplatform.googleapis.com/Endpoint AND resource.labels.endpoint_id=${ENDPOINT_ID}" \
  --limit=30 \
  --format="table(timestamp,jsonPayload.message)"
```
