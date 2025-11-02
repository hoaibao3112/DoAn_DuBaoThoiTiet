# Start script - Launch all services with docker-compose

Write-Host "🚀 Starting AI Zalo Assistant services..." -ForegroundColor Cyan
Write-Host ""

# Check if .env exists
if (-not (Test-Path ".env")) {
    Write-Host "❌ .env file not found. Run .\setup.ps1 first!" -ForegroundColor Red
    exit 1
}

# Build and start services
Write-Host "Building and starting containers..." -ForegroundColor Yellow
docker-compose up -d --build

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Services started successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Services are running at:" -ForegroundColor Cyan
    Write-Host "  - FastAPI (AI Service): http://localhost:8000" -ForegroundColor White
    Write-Host "  - API Docs: http://localhost:8000/docs" -ForegroundColor White
    Write-Host "  - n8n (Workflow): http://localhost:5678" -ForegroundColor White
    Write-Host ""
    Write-Host "To view logs:" -ForegroundColor Yellow
    Write-Host "  docker-compose logs -f" -ForegroundColor White
    Write-Host ""
    Write-Host "To check status:" -ForegroundColor Yellow
    Write-Host "  docker-compose ps" -ForegroundColor White
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Open http://localhost:5678 and login to n8n" -ForegroundColor White
    Write-Host "2. Import workflow from n8n-workflows/Zalo_AI_Assistant.json" -ForegroundColor White
    Write-Host "3. Configure Google Drive credentials in n8n" -ForegroundColor White
    Write-Host "4. Activate the workflow" -ForegroundColor White
    Write-Host "5. Test with: .\test.ps1" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "❌ Failed to start services. Check logs with:" -ForegroundColor Red
    Write-Host "  docker-compose logs" -ForegroundColor White
    Write-Host ""
    exit 1
}
