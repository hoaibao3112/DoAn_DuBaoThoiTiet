<#
.SYNOPSIS
  Set or update Telegram webhook for the Telegram AI Assistant n8n workflow.

.DESCRIPTION
  This script sets Telegram webhook to point at the n8n webhook path used in the
  repository: /webhook/telegram-webhook/telegram-ai-assistant

  It can read `TELEGRAM_BOT_TOKEN` and `WEBHOOK_URL` from a `.env` file in the
  repository root if you pass `-UseEnv`. Otherwise pass `-Token` and `-WebhookBase`.

.EXAMPLE
  # Use values from .env
  .\scripts\set-telegram-webhook.ps1 -UseEnv

  # Provide token and localtunnel url directly
  .\scripts\set-telegram-webhook.ps1 -Token "123:ABC" -WebhookBase "https://xxxx.loca.lt"

  # Delete existing webhook first, then set new one
  .\scripts\set-telegram-webhook.ps1 -UseEnv -DeleteFirst

#>

param(
    [string]$Token,
    [string]$WebhookBase,
    [switch]$UseEnv,
    [switch]$DeleteFirst
)

function Read-DotEnv($path) {
    if (-not (Test-Path $path)) { return @{} }
    $lines = Get-Content $path | ForEach-Object { $_.Trim() } | Where-Object { $_ -and -not ($_.StartsWith('#')) }
    $hash = @{}
    foreach ($l in $lines) {
        if ($l -match '^(.*?)=(.*)$') {
            $k = $matches[1].Trim()
            $v = $matches[2].Trim()
            # remove surrounding quotes
            if ($v.StartsWith('"') -and $v.EndsWith('"')) { $v = $v.Substring(1,$v.Length-2) }
            if ($v.StartsWith("'") -and $v.EndsWith("'")) { $v = $v.Substring(1,$v.Length-2) }
            $hash[$k] = $v
        }
    }
    return $hash
}

Push-Location (Split-Path -Path $MyInvocation.MyCommand.Definition -Parent) | Out-Null
# repo root is one level up from scripts
$repoRoot = Resolve-Path '..' | Select-Object -ExpandProperty Path
Pop-Location | Out-Null

if ($UseEnv) {
    $envFile = Join-Path $repoRoot '.env'
    $vals = Read-DotEnv $envFile
    if (-not $Token) { $Token = $vals['TELEGRAM_BOT_TOKEN'] }
    if (-not $WebhookBase) { $WebhookBase = $vals['WEBHOOK_URL'] }
}

if (-not $Token) {
    $Token = Read-Host -Prompt 'Enter your Telegram bot token (from BotFather)'
}

if (-not $WebhookBase) {
    $WebhookBase = Read-Host -Prompt 'Enter public webhook base URL (https://...)'
}

if (-not $WebhookBase.EndsWith('/')) { $WebhookBase = $WebhookBase + '/' }

$webhookPath = "$WebhookBase`webhook/telegram-webhook/telegram-ai-assistant"

Write-Host "Using webhook URL: $webhookPath" -ForegroundColor Cyan

try {
    if ($DeleteFirst) {
        Write-Host 'Deleting existing webhook (if any)...' -ForegroundColor Yellow
        $delUri = "https://api.telegram.org/bot$Token/deleteWebhook"
        $delResp = Invoke-RestMethod -Method Get -Uri $delUri -ErrorAction Stop
        Write-Host "Delete response: $($delResp | ConvertTo-Json -Depth 2)"
    }

    Write-Host 'Setting new webhook...' -ForegroundColor Yellow
    $setUri = "https://api.telegram.org/bot$Token/setWebhook?url=$webhookPath"
    $setResp = Invoke-RestMethod -Method Get -Uri $setUri -ErrorAction Stop
    Write-Host 'Set webhook response:' -ForegroundColor Green
    Write-Host ($setResp | ConvertTo-Json -Depth 3) -ForegroundColor Green
    if ($setResp.ok) {
        Write-Host "Webhook configured successfully." -ForegroundColor Green
    } else {
        Write-Host "Telegram returned ok=false. Response: $($setResp | ConvertTo-Json -Depth 3)" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host 'Error while setting webhook:' -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    if ($_.Exception.Response) {
        try { $body = $_.Exception.Response.GetResponseStream() | New-Object System.IO.StreamReader | ForEach-Object { $_.ReadToEnd() } ; Write-Host $body } catch { }
    }
    exit 1
}

Write-Host 'Done.' -ForegroundColor Cyan
