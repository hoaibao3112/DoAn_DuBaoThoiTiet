# Test Script - Verify Weather Forecasting System

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Testing Weather Forecasting System  " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if server is running
Write-Host "🔍 Checking if FastAPI server is running..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/weather/health" -Method GET
    Write-Host "✅ Server is running!" -ForegroundColor Green
    Write-Host "   APIs configured:" -ForegroundColor White
    Write-Host "   - OpenWeatherMap: $($health.apis.openweather)" -ForegroundColor White
    Write-Host "   - WAQI: $($health.apis.waqi)" -ForegroundColor White
} catch {
    Write-Host "❌ Server not running! Start with .\start.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  TEST 1: Real-time Weather API  " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

try {
    $current = Invoke-RestMethod -Uri "http://localhost:8000/weather/current?city=Hanoi" -Method GET
    Write-Host "✅ Real-time API works!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📍 City: $($current.city)" -ForegroundColor White
    Write-Host "🌡️  Temperature: $($current.temperature)°C (feels like $($current.feels_like)°C)" -ForegroundColor White
    Write-Host "💧 Humidity: $($current.humidity)%" -ForegroundColor White
    Write-Host "☁️  Description: $($current.description)" -ForegroundColor White
    Write-Host "💨 PM2.5: $($current.pm25) µg/m³" -ForegroundColor White
    Write-Host "🌫️  AQI: $($current.aqi) ($($current.aqi_category))" -ForegroundColor White
    Write-Host "📡 Source: $($current.source)" -ForegroundColor White
} catch {
    Write-Host "⚠️  Real-time API error: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  TEST 2: ML Forecast API (Batch)  " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if models exist
if (Test-Path "models/pm25_forecast.pkl") {
    Write-Host "✅ ML models found!" -ForegroundColor Green
    
    try {
        $forecast = Invoke-RestMethod -Uri "http://localhost:8000/weather/forecast-ml/batch/3" -Method GET
        
        if ($forecast.success) {
            Write-Host "✅ ML forecast works!" -ForegroundColor Green
            Write-Host ""
            Write-Host "📊 Model Info:" -ForegroundColor Yellow
            Write-Host "   - PM2.5 R²: $($forecast.model_info.pm25_r2)" -ForegroundColor White
            Write-Host "   - AQI R²: $($forecast.model_info.aqi_r2)" -ForegroundColor White
            Write-Host "   - Trained: $($forecast.model_info.trained_at)" -ForegroundColor White
            Write-Host ""
            Write-Host "🔮 3-Day Forecast:" -ForegroundColor Yellow
            
            foreach ($day in $forecast.forecasts) {
                Write-Host ""
                Write-Host "   📅 $($day.date)" -ForegroundColor Cyan
                Write-Host "      PM2.5: $($day.pm25) µg/m³" -ForegroundColor White
                Write-Host "      AQI: $($day.aqi) ($($day.category))" -ForegroundColor White
                Write-Host "      Confidence: $($day.confidence)" -ForegroundColor White
            }
        } else {
            Write-Host "❌ ML forecast failed: $($forecast.error)" -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ ML forecast error: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "⚠️  ML models not found! Train models with:" -ForegroundColor Yellow
    Write-Host "   .\train_ml.ps1" -ForegroundColor White
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  TEST 3: ML Forecast API (Custom Data)  " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path "models/pm25_forecast.pkl") {
    # Prepare test data
    $testData = @{
        historical_data = @(
            @{date="2024-01-01"; pm25=45.2; pm10=89.1; aqi=102; no=12; no2=34},
            @{date="2024-01-02"; pm25=50.3; pm10=95.2; aqi=115; no=15; no2=38},
            @{date="2024-01-03"; pm25=42.1; pm10=85.3; aqi=98; no=11; no2=32},
            @{date="2024-01-04"; pm25=48.5; pm10=92.4; aqi=110; no=14; no2=36},
            @{date="2024-01-05"; pm25=55.2; pm10=102.1; aqi=125; no=18; no2=42},
            @{date="2024-01-06"; pm25=51.8; pm10=98.5; aqi=118; no=16; no2=40},
            @{date="2024-01-07"; pm25=46.3; pm10=88.2; aqi=105; no=13; no2=35}
        )
        days_ahead = 1
    }
    
    $jsonBody = $testData | ConvertTo-Json -Depth 10
    
    try {
        $customForecast = Invoke-RestMethod -Uri "http://localhost:8000/weather/forecast-ml" `
            -Method POST `
            -Body $jsonBody `
            -ContentType "application/json"
        
        if ($customForecast.success) {
            Write-Host "✅ Custom ML forecast works!" -ForegroundColor Green
            Write-Host ""
            Write-Host "📅 Forecast Date: $($customForecast.forecast_date)" -ForegroundColor Cyan
            Write-Host "💨 PM2.5: $($customForecast.pm25_forecast) µg/m³" -ForegroundColor White
            Write-Host "🌫️  AQI: $($customForecast.aqi_forecast) ($($customForecast.aqi_category))" -ForegroundColor White
            Write-Host "📊 Confidence: $($customForecast.confidence)" -ForegroundColor White
            Write-Host ""
            Write-Host "💡 Recommendation:" -ForegroundColor Yellow
            Write-Host "   $($customForecast.recommendation)" -ForegroundColor White
        } else {
            Write-Host "❌ Custom forecast failed: $($customForecast.error)" -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ Custom forecast error: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "   Response: $($_.ErrorDetails.Message)" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  TEST 4: n8n Workflow  " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

try {
    $n8nHealth = Invoke-RestMethod -Uri "http://localhost:5678/healthz" -Method GET -ErrorAction SilentlyContinue
    Write-Host "✅ n8n is running!" -ForegroundColor Green
    Write-Host "   Access: http://localhost:5678" -ForegroundColor White
} catch {
    Write-Host "⚠️  n8n not accessible. Check Docker container:" -ForegroundColor Yellow
    Write-Host "   docker-compose logs n8n" -ForegroundColor White
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  📋 Summary  " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

$testsRun = 4
$testsPassed = 0

if ($current) { $testsPassed++ }
if ($forecast -and $forecast.success) { $testsPassed++ }
if ($customForecast -and $customForecast.success) { $testsPassed++ }
if ($n8nHealth) { $testsPassed++ }

Write-Host "Tests Passed: $testsPassed / $testsRun" -ForegroundColor $(if ($testsPassed -eq $testsRun) { "Green" } else { "Yellow" })
Write-Host ""

if ($testsPassed -eq $testsRun) {
    Write-Host "🎉 All tests passed! System ready for production." -ForegroundColor Green
} else {
    Write-Host "⚠️  Some tests failed. Check logs above." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor White
    Write-Host "  1. If ML models missing: .\train_ml.ps1" -ForegroundColor Gray
    Write-Host "  2. If APIs not configured: Edit .env with API keys" -ForegroundColor Gray
    Write-Host "  3. If n8n down: docker-compose up -d n8n" -ForegroundColor Gray
}

Write-Host ""
Write-Host "📚 Documentation:" -ForegroundColor Yellow
Write-Host "  - README.md - Project overview" -ForegroundColor White
Write-Host "  - ML_FORECASTING_GUIDE.md - ML training & usage" -ForegroundColor White
Write-Host "  - API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
