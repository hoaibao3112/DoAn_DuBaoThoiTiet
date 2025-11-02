# Test script - Send test requests to verify the system is working

Write-Host "🧪 Testing AI Zalo Assistant..." -ForegroundColor Cyan
Write-Host ""

# Test 1: FastAPI health check
Write-Host "Test 1: FastAPI health check" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Method Get -Uri "http://localhost:8000/docs" -ErrorAction Stop
    Write-Host "✅ FastAPI is responding" -ForegroundColor Green
} catch {
    Write-Host "❌ FastAPI is not responding" -ForegroundColor Red
    Write-Host "   Make sure services are running: .\start.ps1" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# Test 2: Call /generate endpoint directly
Write-Host "Test 2: Testing /generate endpoint" -ForegroundColor Yellow
try {
    $body = @{
        prompt = "Tôi muốn mua một chiếc xe máy giá rẻ"
        user_id = "test_user_123"
    } | ConvertTo-Json

    $response = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/generate" `
        -ContentType "application/json" `
        -Body $body `
        -ErrorAction Stop

    Write-Host "✅ /generate endpoint working" -ForegroundColor Green
    Write-Host "   Ad text: $($response.ad_text)" -ForegroundColor White
    if ($response.drive_file_id) {
        Write-Host "   Drive file: $($response.drive_file_id)" -ForegroundColor White
    }
} catch {
    Write-Host "❌ Failed to call /generate" -ForegroundColor Red
    Write-Host "   Error: $_" -ForegroundColor Red
}
Write-Host ""

# Test 3: n8n webhook (if workflow is active)
Write-Host "Test 3: Testing n8n webhook" -ForegroundColor Yellow
Write-Host "   (This requires workflow to be active in n8n)" -ForegroundColor Gray
try {
    $body = @{
        message = "Tôi muốn mua laptop gaming"
        user_id = "zalo_test_456"
    } | ConvertTo-Json

    $response = Invoke-RestMethod -Method Post -Uri "http://localhost:5678/webhook/zalo-webhook" `
        -ContentType "application/json" `
        -Body $body `
        -ErrorAction Stop

    Write-Host "✅ n8n webhook responding" -ForegroundColor Green
    Write-Host "   Response: $($response | ConvertTo-Json -Compress)" -ForegroundColor White
} catch {
    if ($_.Exception.Response.StatusCode -eq 404) {
        Write-Host "⚠️  Workflow not found or not active" -ForegroundColor Yellow
        Write-Host "   Go to http://localhost:5678 and activate the workflow" -ForegroundColor Yellow
    } else {
        Write-Host "⚠️  n8n webhook error: $_" -ForegroundColor Yellow
    }
}
Write-Host ""

# Summary
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Test Summary" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps to complete setup:" -ForegroundColor Yellow
Write-Host "1. Open http://localhost:5678" -ForegroundColor White
Write-Host "2. Login with credentials from .env" -ForegroundColor White
Write-Host "3. Import n8n-workflows/Zalo_AI_Assistant.json" -ForegroundColor White
Write-Host "4. Configure Google Drive credential" -ForegroundColor White
Write-Host "5. Activate the workflow (toggle Active)" -ForegroundColor White
Write-Host "6. Setup ngrok for public webhook: ngrok http 5678" -ForegroundColor White
Write-Host "7. Configure Zalo webhook URL in Zalo Developer Console" -ForegroundColor White
Write-Host ""
