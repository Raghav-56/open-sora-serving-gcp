gcloud ai endpoints deploy-model 254130122527866880 `
  --region=$env:REGION `
  --model=$env:MODEL_ID `
  --display-name="opensora-deployment-v1" `
  --machine-type=$env:MACHINE_TYPE `
  --accelerator="type=$($env:ACCELERATOR_TYPE),count=1" `
  --service-account=$env:SERVICE_ACCOUNT `
  --min-replica-count=1 `
  --max-replica-count=1 `
  --project=$env:PROJECT_ID