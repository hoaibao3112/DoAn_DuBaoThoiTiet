# Stop script - Gracefully shutdown all services

Write-Host "🛑 Stopping AI Zalo Assistant services..." -ForegroundColor Cyan
Write-Host ""

docker-compose down

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Services stopped successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To start again: .\start.ps1" -ForegroundColor Yellow
    Write-Host "To remove volumes (delete data): docker-compose down -v" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "❌ Failed to stop services" -ForegroundColor Red
    exit 1
}
