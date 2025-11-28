# PowerShell wrapper to run the Python send_test_email.py script
# Use this wrapper when creating a Scheduled Task on Windows so the correct working directory and virtualenv are used.

param(
    [string]$RepoRoot = "C:\Users\PC\Desktop\DoAn_PTDL"
)

Set-Location -Path $RepoRoot

# If you use a virtualenv at .\venv, activate it; otherwise it will use system python
$venvActivate = Join-Path $RepoRoot "venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    try {
        & $venvActivate
    } catch {
        Write-Host "Failed to activate virtualenv: $_"
    }
}

$scriptPath = Join-Path $RepoRoot "scripts\send_test_email.py"
Write-Host "Running: python $scriptPath"
python $scriptPath

if ($LASTEXITCODE -ne 0) {
    Write-Host "send_test_email.py exited with code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "send_test_email.py completed successfully"
