# apply_ngrok_url_and_test.ps1
param(
    [string]$NgrokUrl = 'https://36ae8e21c8dd.ngrok-free.app/'
)

function Write-Ok($m){ Write-Host "[OK] $m" -ForegroundColor Green }
function Write-Err($m){ Write-Host "[ERR] $m" -ForegroundColor Red }

$ProjectDir = (Get-Location).Path
$envFile = Join-Path $ProjectDir '.env'
if (-not (Test-Path $envFile)){
    Write-Err ".env not found at $envFile"
    exit 1
}

try{
    $content = (Get-Content $envFile) -join "`n"
    $new = [regex]::Replace($content,'(?m)^\s*WEBHOOK_URL\s*=.*',"WEBHOOK_URL=$NgrokUrl")
    Set-Content -Path $envFile -Value $new
    Write-Ok ".env updated with $NgrokUrl"
} catch {
    Write-Err "Failed to update .env: $_"
    exit 1
}

try{
    Write-Host "Recreating n8n container..."
    docker-compose up -d --force-recreate n8n
    Write-Ok "n8n recreated"
} catch {
    Write-Err "Failed to recreate n8n: $_"
    exit 1
}

try{
    Write-Host "Setting Telegram webhook..."
    powershell -ExecutionPolicy Bypass -File "$ProjectDir\scripts\set-telegram-webhook.ps1" -UseEnv -DeleteFirst
    Write-Ok "Telegram webhook set (script finished)"
} catch {
    Write-Err "Setting webhook failed: $_"
}

# Run simulate
$simUrl = $NgrokUrl + 'webhook/telegram-webhook/telegram-ai-assistant'
Write-Host "Running simulate against: $simUrl"
try{
    python "$ProjectDir\scripts\simulate_telegram_webhook.py" $simUrl
} catch {
    Write-Err "Simulate failed: $_"
}

Write-Host "Tailing n8n logs (Ctrl+C to stop)..."
docker-compose logs -f n8n
