param(
    [string]$NgrokPath = "C:\\Users\\PC\\Downloads\\ngrok-v3-stable-windows-amd64\\ngrok.exe",
    [string]$ProjectDir = "${PWD}",
    [switch]$RestartN8n,
    [switch]$SetWebhook,
    [switch]$Simulate,
    [int]$WaitSeconds = 30
)

function Write-Ok($m){ Write-Host "[OK] $m" -ForegroundColor Green }
function Write-Err($m){ Write-Host "[ERR] $m" -ForegroundColor Red }

# 1. Ensure ngrok exists
if (-not (Test-Path $NgrokPath)){
    Write-Err "ngrok not found at: $NgrokPath"
    Write-Host "Pass -NgrokPath with the full path to ngrok.exe or place ngrok in PATH."
    exit 1
}

# 2. Start ngrok if not already running
$existing = Get-Process -Name ngrok -ErrorAction SilentlyContinue
if ($existing){
    Write-Host "ngrok already running (PID: $($existing.Id)). Will try to use existing tunnel." -ForegroundColor Yellow
} else {
    Write-Host "Starting ngrok..."
    Start-Process -FilePath $NgrokPath -ArgumentList 'http','5678' -WindowStyle Hidden
    Start-Sleep -Seconds 1
}

# 3. Wait for ngrok local API to be available and return public_url
$publicUrl = $null
$start = Get-Date
while ((Get-Date) - $start -lt (New-TimeSpan -Seconds $WaitSeconds)){
    try{
        $t = Invoke-RestMethod -Uri 'http://127.0.0.1:4040/api/tunnels' -ErrorAction Stop
        if ($t -and $t.tunnels -and $t.tunnels.Count -gt 0){
            $publicUrl = $t.tunnels[0].public_url
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $publicUrl){
    Write-Err "Failed to obtain ngrok public URL after $WaitSeconds seconds. Check ngrok process and logs (http://127.0.0.1:4040)."
    exit 1
}

Write-Ok "ngrok public URL: $publicUrl"

# Ensure trailing slash
if ($publicUrl[-1] -ne '/') { $publicUrl = $publicUrl + '/' }

# 4. Update .env in project directory
$envFile = Join-Path $ProjectDir '.env'
if (-not (Test-Path $envFile)){
    Write-Err ".env not found in $ProjectDir"
    exit 1
}

$content = (Get-Content $envFile) -join "`n"
$new = [regex]::Replace($content,'(?m)^\s*WEBHOOK_URL\s*=.*','WEBHOOK_URL=' + $publicUrl)
Set-Content -Path $envFile -Value $new
Write-Ok ".env updated with WEBHOOK_URL=$publicUrl"

# 5. Optionally restart n8n so it picks up new .env
if ($RestartN8n){
    Write-Host "Recreating n8n container..."
    Push-Location $ProjectDir
    try{
        docker-compose up -d --force-recreate n8n
        Write-Ok "n8n restarted"
    } catch {
        Write-Err "Failed to restart n8n: $_"
        Pop-Location
        exit 1
    }
    Pop-Location
}

# 6. Optionally run the repo script to set Telegram webhook (reads .env)
if ($SetWebhook){
    Write-Host "Setting Telegram webhook using scripts/set-telegram-webhook.ps1..."
    try{
        powershell -ExecutionPolicy Bypass -File (Join-Path $ProjectDir 'scripts\set-telegram-webhook.ps1') -UseEnv -DeleteFirst
        Write-Ok "Telegram webhook set"
    } catch {
        Write-Err "Failed to set Telegram webhook: $_"
        exit 1
    }
}

# 7. Optionally run simulate against local or public URL
if ($Simulate){
    $simUrl = "$publicUrl`webhook/telegram-webhook/telegram-ai-assistant"
    Write-Host "Running simulate against: $simUrl"
    try{
        python (Join-Path $ProjectDir 'scripts\simulate_telegram_webhook.py') $simUrl
    } catch {
        Write-Err "Simulate failed: $_"
    }
}

Write-Ok "All done. If you opened n8n Editor, open it at: $publicUrl" 
