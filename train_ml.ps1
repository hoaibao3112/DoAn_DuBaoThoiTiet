# Quick Start Script - Run ML Training Pipeline
# This script runs the complete ML pipeline: ETL -> Train -> Verify

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Weather Forecasting ML Pipeline - Quick Start  " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python environment
Write-Host "🔍 Checking Python environment..." -ForegroundColor Yellow
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "✅ Virtual environment found" -ForegroundColor Green
    & .venv\Scripts\Activate.ps1
} else {
    Write-Host "⚠️  Virtual environment not found. Creating..." -ForegroundColor Yellow
    python -m venv .venv
    & .venv\Scripts\Activate.ps1
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Install dependencies
Write-Host ""
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt -q
Write-Host "✅ Dependencies installed" -ForegroundColor Green

# Check if station_day.csv exists
Write-Host ""
Write-Host "📊 Checking for dataset..." -ForegroundColor Yellow
if (Test-Path "station_day.csv") {
    Write-Host "✅ Dataset found: station_day.csv" -ForegroundColor Green
} else {
    Write-Host "❌ Dataset not found! Please ensure station_day.csv is in the root directory." -ForegroundColor Red
    Write-Host "   Download it or place your dataset here." -ForegroundColor Yellow
    exit 1
}

# Step 1: ETL Pipeline
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  STEP 1: ETL Pipeline (Data Cleaning)  " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Running ETL pipeline..." -ForegroundColor Yellow
python scripts/etl_pipeline.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ ETL pipeline failed! Check error messages above." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ ETL pipeline completed successfully!" -ForegroundColor Green

# Step 2: Train ML Models
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  STEP 2: Train ML Models (RandomForest)  " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Training models (this may take 2-5 minutes)..." -ForegroundColor Yellow
python scripts/train_model.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Model training failed! Check error messages above." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ Model training completed successfully!" -ForegroundColor Green

# Step 3: Verify Models
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  STEP 3: Verify Models  " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

if ((Test-Path "models/pm25_forecast.pkl") -and (Test-Path "models/aqi_forecast.pkl")) {
    Write-Host "✅ Models saved successfully:" -ForegroundColor Green
    Write-Host "   - models/pm25_forecast.pkl" -ForegroundColor White
    Write-Host "   - models/aqi_forecast.pkl" -ForegroundColor White
    Write-Host "   - models/feature_columns.pkl" -ForegroundColor White
    Write-Host "   - models/metadata.json" -ForegroundColor White
    
    # Show model metadata
    if (Test-Path "models/metadata.json") {
        Write-Host ""
        Write-Host "📊 Model Performance:" -ForegroundColor Yellow
        $metadata = Get-Content "models/metadata.json" | ConvertFrom-Json
        Write-Host "   PM2.5 R²: $($metadata.pm25_test_r2)" -ForegroundColor White
        Write-Host "   AQI R²: $($metadata.aqi_test_r2)" -ForegroundColor White
        Write-Host "   Trained at: $($metadata.trained_at)" -ForegroundColor White
    }
} else {
    Write-Host "⚠️  Some model files are missing!" -ForegroundColor Yellow
}

# Final message
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  🎉 ML Pipeline Completed Successfully!  " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Start the application: .\start.ps1" -ForegroundColor White
Write-Host "  2. Test ML endpoint: http://localhost:8000/weather/forecast-ml" -ForegroundColor White
Write-Host "  3. View API docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "Happy Forecasting! 🌤️📊" -ForegroundColor Green
