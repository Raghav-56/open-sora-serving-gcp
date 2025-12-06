# 1. Define your payload
$body = @{
    prompt = "A dog and cat playing together in a park"
    resolution = "256px"
    num_frames = 49
    aspect_ratio = "16:9"
    output_bucket = "nannie-opensora-weights-so"
    output_prefix = "test-videos/"
} | ConvertTo-Json

# 2. Send request to Vertex AI
$response = Invoke-RestMethod `
    -Uri "https://europe-west4-aiplatform.googleapis.com/v1/projects/nannieai-website-stealth/locations/europe-west4/endpoints/9111584689658789888:rawPredict" `
    -Method Post `
    -Headers @{ "Authorization" = "Bearer $env:ACCESS_TOKEN"; "Content-Type" = "application/json" } `
    -Body $body

# 3. Print the result
$response