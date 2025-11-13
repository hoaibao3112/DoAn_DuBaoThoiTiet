# Import and activate an n8n workflow using REST API
# Reads N8N_USER and N8N_PASSWORD from .env if present, otherwise uses defaults

$envPath = Join-Path $PSScriptRoot '..\.env'
$envVars = @{}
if (Test-Path $envPath) {
    Get-Content $envPath | ForEach-Object {
        if ($_ -match '^\s*([^#=]+)=(.*)$') {
            $k = $matches[1].Trim()
            $v = $matches[2].Trim()
            # strip surrounding single or double quotes if present
            if ($v.Length -ge 2) {
                if (($v[0] -eq '"') -and ($v[$v.Length-1] -eq '"')) { $v = $v.Substring(1,$v.Length-2) }
                elseif (($v[0] -eq "'") -and ($v[$v.Length-1] -eq "'")) { $v = $v.Substring(1,$v.Length-2) }
            }
            $envVars[$k] = $v
        }
    }
}

$user = $envVars['N8N_USER']  
$pass = $envVars['N8N_PASSWORD']
if (-not $user) { $user = 'admin' }
if (-not $pass) { $pass = 'admin' }

$workflowPath = Join-Path $PSScriptRoot '..\n8n-workflows\Zalo_AI_Assistant.json'
if (-not (Test-Path $workflowPath)) {
    Write-Error "Workflow file not found: $workflowPath"
    exit 2
}

$wfJson = Get-Content -Raw -Path $workflowPath

$pair = "$user`:$pass"
$b = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes($pair))
$headers = @{ Authorization = "Basic $b" }

$baseUrl = $envVars['WEBHOOK_URL'] -replace '/$',''
if (-not $baseUrl) { $baseUrl = 'http://localhost:5678' }

Write-Host "Using n8n admin user: $user"
Write-Host "Posting workflow to $baseUrl/rest/workflows ..."
try {
    $resp = Invoke-RestMethod -Uri "$baseUrl/rest/workflows" -Method Post -Body $wfJson -ContentType 'application/json' -Headers $headers -ErrorAction Stop
} catch {
    Write-Error "Failed to POST workflow: $($_.Exception.Message)"
    exit 3
}

Write-Host "Created workflow id: $($resp.id)"
Write-Host "Activating workflow..."
try {
    Invoke-RestMethod -Uri "$baseUrl/rest/workflows/$($resp.id)/activate" -Method Post -Headers $headers -ErrorAction Stop
    Write-Host "Workflow activated successfully."
} catch {
    Write-Error "Failed to activate workflow: $($_.Exception.Message)"
    exit 4
}

# Print webhook info from returned workflow (if available)
if ($resp.nodes) {
    $webhooks = $resp.nodes | Where-Object { $_.type -like '*webhook*' }
    if ($webhooks) {
        Write-Host "Detected webhook nodes in imported workflow:"
        foreach ($n in $webhooks) {
            Write-Host " - node id:$($n.id) name:$($n.name)"
        }
    }
}

Write-Host "Done."