# 1. Define your payload (direct format for rawPredict)
$body = @{
    prompt = "A dog and cat playing together in a park"
    resolution = "256px"
    num_frames = 17
    aspect_ratio = "16:9"
    mode = "t2v"
    output_bucket = "nannie-opensora-weights-so"
    output_prefix = "test-videos/"
} | ConvertTo-Json -Depth 10

# 2. Send request to Vertex AI using rawPredict
$response = Invoke-RestMethod `
    -Uri "https://europe-west4-aiplatform.googleapis.com/v1/projects/nannieai-website-stealth/locations/europe-west4/endpoints/7157022451379994624:rawPredict" `
    -Method Post `
    -Headers @{ "Authorization" = "Bearer $env:ACCESS_TOKEN"; "Content-Type" = "application/json" } `
    -Body $body

# 3. Print the result
$response