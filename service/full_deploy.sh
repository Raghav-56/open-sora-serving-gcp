#!/bin/bash
# =============================================================================
# Open-Sora Vertex AI Full Deployment Script (Bash Version)
# =============================================================================
# This script handles the complete deployment of Open-Sora to Vertex AI
# Run from the service/ directory
# Usage: ./full_deploy.sh
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# Load Configuration from .env file
# -----------------------------------------------------------------------------
echo -e "${CYAN}=============================================${NC}"
echo -e "${CYAN}  Open-Sora Vertex AI Deployment${NC}"
echo -e "${CYAN}=============================================${NC}"
echo ""

# Find and load .env file (in parent directory)
ENV_FILE="../.env"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}ERROR: Missing .env file!${NC}"
    echo "Create .env file in the project root with your configuration."
    echo "Example: copy env.example.txt to .env"
    exit 1
fi

# Load .env file (handle Windows CRLF line endings)
while IFS='=' read -r key value; do
    # Skip comments and empty lines
    [[ "$key" =~ ^#.*$ ]] && continue
    [[ -z "$key" ]] && continue
    # Remove carriage returns and whitespace
    key=$(echo "$key" | tr -d '\r' | xargs)
    value=$(echo "$value" | tr -d '\r' | xargs)
    # Export the variable
    if [ -n "$key" ]; then
        export "$key=$value"
    fi
done < "$ENV_FILE"

# Set defaults if not provided
REPOSITORY=${REPOSITORY:-opensora-serving-api}
IMAGE_NAME=${IMAGE_NAME:-opensora-api}
TAG=${TAG:-v1.0.0}
WEIGHT_PREFIX=${WEIGHT_PREFIX:-ckpts/}
MODEL_PATH=${MODEL_PATH:-/app/ckpts}
FORCE_DOWNLOAD=${FORCE_DOWNLOAD:-false}
PORT=${PORT:-8080}
JOB_RETENTION_SECONDS=${JOB_RETENTION_SECONDS:-3600}
MAX_COMPLETED_JOBS=${MAX_COMPLETED_JOBS:-100}
DEFAULT_RESOLUTION=${DEFAULT_RESOLUTION:-256px}
DEFAULT_NUM_FRAMES=${DEFAULT_NUM_FRAMES:-49}
DEFAULT_ASPECT_RATIO=${DEFAULT_ASPECT_RATIO:-16:9}
GENERATION_TIMEOUT=${GENERATION_TIMEOUT:-1800}
MACHINE_TYPE=${MACHINE_TYPE:-a2-ultragpu-1g}
ACCELERATOR_TYPE=${ACCELERATOR_TYPE:-nvidia-a100-80gb}

# Computed values
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${TAG}"
MODEL_NAME="opensora-video-${TAG}"
ENDPOINT_NAME="opensora-video-endpoint"
SERVICE_ACCOUNT="ml-model-serving@${PROJECT_ID}.iam.gserviceaccount.com"

# Container environment variables
CONTAINER_ENV_VARS="WEIGHT_BUCKET=${WEIGHT_BUCKET},WEIGHT_PREFIX=${WEIGHT_PREFIX},MODEL_PATH=${MODEL_PATH},PORT=${PORT},JOB_RETENTION_SECONDS=${JOB_RETENTION_SECONDS},MAX_COMPLETED_JOBS=${MAX_COMPLETED_JOBS},DEFAULT_RESOLUTION=${DEFAULT_RESOLUTION},DEFAULT_NUM_FRAMES=${DEFAULT_NUM_FRAMES},DEFAULT_ASPECT_RATIO=${DEFAULT_ASPECT_RATIO},GENERATION_TIMEOUT=${GENERATION_TIMEOUT}"

# Display configuration
echo -e "${YELLOW}[Configuration]${NC}"
echo "  PROJECT_ID     : ${PROJECT_ID}"
echo "  REGION         : ${REGION}"
echo "  IMAGE_URI      : ${IMAGE_URI}"
echo "  MODEL_NAME     : ${MODEL_NAME}"
echo "  ENDPOINT_NAME  : ${ENDPOINT_NAME}"
echo "  MACHINE_TYPE   : ${MACHINE_TYPE}"
echo "  ACCELERATOR    : ${ACCELERATOR_TYPE}"
echo "  WEIGHT_BUCKET  : ${WEIGHT_BUCKET}"
echo ""

# Validate required variables
if [ -z "$PROJECT_ID" ] || [ -z "$REGION" ] || [ -z "$WEIGHT_BUCKET" ]; then
    echo -e "${RED}ERROR: Missing required variables (PROJECT_ID, REGION, WEIGHT_BUCKET)${NC}"
    exit 1
fi

# Confirm before proceeding
read -p "Proceed with deployment? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo -e "${RED}Deployment cancelled.${NC}"
    exit 0
fi

# -----------------------------------------------------------------------------
# Step 0: Prerequisites Check
# -----------------------------------------------------------------------------
echo ""
echo -e "${GREEN}[Step 0] Prerequisites Check${NC}"
echo "---------------------------------------------"

# Check gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}  [X] gcloud CLI not found${NC}"
    echo "  Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

ACCOUNT=$(gcloud auth list --filter="status:ACTIVE" --format="value(account)" 2>/dev/null | head -1)
if [ -n "$ACCOUNT" ]; then
    echo -e "${GREEN}  [OK] Authenticated as: $ACCOUNT${NC}"
else
    echo -e "${RED}  [X] Not authenticated to gcloud${NC}"
    echo "  Run: gcloud auth login"
    exit 1
fi

# Set the project
echo "  Setting project to ${PROJECT_ID}..."
gcloud config set project "$PROJECT_ID"

# Configure Docker authentication
echo "  Configuring Docker authentication..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# -----------------------------------------------------------------------------
# Step 1: Create Artifact Repository (if not exists)
# -----------------------------------------------------------------------------
echo ""
echo -e "${GREEN}[Step 1] Artifact Registry Setup${NC}"
echo "---------------------------------------------"

if ! gcloud artifacts repositories describe "$REPOSITORY" --location="$REGION" --project="$PROJECT_ID" &>/dev/null; then
    echo "  Creating repository: ${REPOSITORY}..."
    gcloud artifacts repositories create "$REPOSITORY" \
        --repository-format=docker \
        --location="$REGION" \
        --description="Open-Sora v2 API" \
        --project="$PROJECT_ID"
    echo -e "${GREEN}  [OK] Repository created${NC}"
else
    echo -e "${GREEN}  [OK] Repository already exists${NC}"
fi

# -----------------------------------------------------------------------------
# Step 2: Build Docker Image (Cloud Build)
# -----------------------------------------------------------------------------
# echo ""
# echo -e "${GREEN}[Step 2] Building Docker Image${NC}"
# echo "---------------------------------------------"
# echo "  Using Cloud Build (this takes 15-20 minutes)..."
# echo "  Image: ${IMAGE_URI}"
# echo ""
echo -e "${YELLOW}  [SKIPPED] Image already exists: ${IMAGE_URI}${NC}"

# gcloud builds submit \
#     --region="$REGION" \
#     --tag "$IMAGE_URI" \
#     --timeout=3600s \
#     --machine-type=e2-highcpu-32 \
#     --project="$PROJECT_ID" \
#     .

# if [ $? -ne 0 ]; then
#     echo -e "${RED}  [X] Build failed!${NC}"
#     exit 1
# fi
# echo -e "${GREEN}  [OK] Image built and pushed${NC}"

# -----------------------------------------------------------------------------
# Step 3: Upload Model to Vertex AI
# -----------------------------------------------------------------------------
echo ""
echo -e "${GREEN}[Step 3] Uploading Model to Vertex AI${NC}"
echo "---------------------------------------------"

# Check if model already exists
EXISTING_MODEL=$(gcloud ai models list \
    --region="$REGION" \
    --filter="displayName:${MODEL_NAME}" \
    --format="value(name)" \
    --project="$PROJECT_ID" 2>/dev/null | head -1)

SHOULD_UPLOAD=true

if [ -n "$EXISTING_MODEL" ]; then
    echo -e "${YELLOW}  Model '${MODEL_NAME}' already exists.${NC}"
    read -p "  Delete and re-upload? (y/n): " overwrite
    if [ "$overwrite" = "y" ]; then
        echo "  Deleting existing model..."
        gcloud ai models delete "$EXISTING_MODEL" --region="$REGION" --project="$PROJECT_ID" --quiet
    else
        echo -e "${YELLOW}  Using existing model.${NC}"
        SHOULD_UPLOAD=false
    fi
fi

if [ "$SHOULD_UPLOAD" = true ]; then
    echo "  Uploading model..."
    gcloud ai models upload \
        --region="$REGION" \
        --display-name="$MODEL_NAME" \
        --container-image-uri="$IMAGE_URI" \
        --container-health-route=/health \
        --container-predict-route=/predict \
        --container-ports=8080 \
        --container-env-vars="$CONTAINER_ENV_VARS" \
        --project="$PROJECT_ID"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}  [X] Model upload failed!${NC}"
        exit 1
    fi
    echo -e "${GREEN}  [OK] Model uploaded${NC}"
fi

# -----------------------------------------------------------------------------
# Step 4: Create Endpoint (if not exists)
# -----------------------------------------------------------------------------
echo ""
echo -e "${GREEN}[Step 4] Creating Endpoint${NC}"
echo "---------------------------------------------"

EXISTING_ENDPOINT=$(gcloud ai endpoints list \
    --region="$REGION" \
    --filter="displayName:${ENDPOINT_NAME}" \
    --format="value(name)" \
    --project="$PROJECT_ID" 2>/dev/null | head -1)

if [ -n "$EXISTING_ENDPOINT" ]; then
    echo -e "${GREEN}  [OK] Endpoint already exists${NC}"
    ENDPOINT_ID=$(echo "$EXISTING_ENDPOINT" | awk -F'/' '{print $NF}')
else
    echo "  Creating endpoint: ${ENDPOINT_NAME}..."
    gcloud ai endpoints create \
        --region="$REGION" \
        --display-name="$ENDPOINT_NAME" \
        --project="$PROJECT_ID"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}  [X] Endpoint creation failed!${NC}"
        exit 1
    fi
    
    # Get the endpoint ID
    ENDPOINT_ID=$(gcloud ai endpoints list \
        --region="$REGION" \
        --filter="displayName:${ENDPOINT_NAME}" \
        --format="value(name)" \
        --project="$PROJECT_ID" | head -1 | awk -F'/' '{print $NF}')
    
    echo -e "${GREEN}  [OK] Endpoint created: ${ENDPOINT_ID}${NC}"
fi

# -----------------------------------------------------------------------------
# Step 5: Deploy Model to Endpoint
# -----------------------------------------------------------------------------
echo ""
echo -e "${GREEN}[Step 5] Deploying Model to Endpoint${NC}"
echo "---------------------------------------------"

# Get model ID
MODEL_ID=$(gcloud ai models list \
    --region="$REGION" \
    --filter="displayName:${MODEL_NAME}" \
    --format="value(name)" \
    --project="$PROJECT_ID" | head -1 | awk -F'/' '{print $NF}')

echo "  Model ID   : ${MODEL_ID}"
echo "  Endpoint ID: ${ENDPOINT_ID}"
echo ""
echo "  Deploying with:"
echo "    - Machine: ${MACHINE_TYPE}"
echo "    - GPU: ${ACCELERATOR_TYPE}"
echo "    - Replicas: 1"
echo ""
echo -e "${YELLOW}  This will take 15-30 minutes...${NC}"

gcloud ai endpoints deploy-model "$ENDPOINT_ID" \
    --region="$REGION" \
    --model="$MODEL_ID" \
    --display-name="opensora-deployment-${TAG}" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=${ACCELERATOR_TYPE},count=1" \
    --service-account="$SERVICE_ACCOUNT" \
    --min-replica-count=1 \
    --max-replica-count=1 \
    --project="$PROJECT_ID"

if [ $? -ne 0 ]; then
    echo -e "${RED}  [X] Deployment failed!${NC}"
    exit 1
fi
echo -e "${GREEN}  [OK] Model deployed successfully!${NC}"

# -----------------------------------------------------------------------------
# Deployment Complete
# -----------------------------------------------------------------------------
echo ""
echo -e "${CYAN}=============================================${NC}"
echo -e "${CYAN}  Deployment Complete!${NC}"
echo -e "${CYAN}=============================================${NC}"
echo ""
echo "  Endpoint ID: ${ENDPOINT_ID}"
echo ""
echo -e "${YELLOW}  API URL:${NC}"
API_URL="https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_ID}:rawPredict"
echo "  ${API_URL}"
echo ""
echo -e "${YELLOW}  Test with curl:${NC}"
echo '  TOKEN=$(gcloud auth print-access-token)'
echo "  curl -X POST \"${API_URL}\" \\"
echo '    -H "Authorization: Bearer $TOKEN" \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"prompt": "A cat playing piano", "resolution": "256px", "num_frames": 49}'"'"
echo ""
echo -e "${YELLOW}  Monitor logs:${NC}"
echo "  gcloud logging read 'resource.type=aiplatform.googleapis.com/Endpoint AND resource.labels.endpoint_id=${ENDPOINT_ID}' --limit=50"
echo ""

