# Setup script - Initialize folders and check prerequisites
# Run this once before first docker-compose up

Write-Host "🚀 AI Zalo Assistant - Setup Script" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "Checking Docker..." -ForegroundColor Yellow
try {
    $dockerVersion = docker version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker not found. Please install Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check if docker-compose is available
Write-Host "Checking docker-compose..." -ForegroundColor Yellow
try {
    $composeVersion = docker-compose version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ docker-compose not found" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ docker-compose is available" -ForegroundColor Green
} catch {
    Write-Host "❌ docker-compose not found" -ForegroundColor Red
    exit 1
}

# Create necessary directories
Write-Host ""
Write-Host "Creating directories..." -ForegroundColor Yellow

$dirs = @("secrets", "n8n-workflows")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✅ Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "✅ Already exists: $dir" -ForegroundColor Green
    }
}

# Check if .env exists
Write-Host ""
Write-Host "Checking environment file..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "✅ Created .env from .env.example" -ForegroundColor Green
        Write-Host "⚠️  Please edit .env and fill in your credentials!" -ForegroundColor Yellow
    } else {
        Write-Host "❌ .env.example not found" -ForegroundColor Red
    }
} else {
    Write-Host "✅ .env exists" -ForegroundColor Green
}

# Check if service account exists
Write-Host ""
Write-Host "Checking Google service account..." -ForegroundColor Yellow
if (-not (Test-Path "secrets/service_account.json")) {
    Write-Host "⚠️  secrets/service_account.json not found" -ForegroundColor Yellow
    Write-Host "   Please download your Google Cloud service account JSON" -ForegroundColor Yellow
    Write-Host "   and save it as secrets/service_account.json" -ForegroundColor Yellow
} else {
    Write-Host "✅ Service account found" -ForegroundColor Green
}

# Summary
Write-Host ""
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Edit .env and fill in:" -ForegroundColor White
Write-Host "   - GDRIVE_FOLDER_ID" -ForegroundColor White
Write-Host "   - ZALO_ACCESS_TOKEN" -ForegroundColor White
Write-Host "   - OPENAI_API_KEY (optional)" -ForegroundColor White
Write-Host "   - N8N_PASSWORD" -ForegroundColor White
Write-Host ""
Write-Host "2. Place service_account.json in secrets/" -ForegroundColor White
Write-Host ""
Write-Host "3. Run: .\start.ps1" -ForegroundColor White
Write-Host ""
