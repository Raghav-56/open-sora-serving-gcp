gcloud ai models upload `
  --region=$env:REGION `
  --display-name=$env:MODEL_NAME `
  --container-image-uri=$env:IMAGE_URI `
  --container-health-route=/health `
  --container-predict-route=/predict `
  --container-ports=8080 `
  --container-env-vars=$env:CONTAINER_ENV_VARS `
  --project=$env:PROJECT_ID