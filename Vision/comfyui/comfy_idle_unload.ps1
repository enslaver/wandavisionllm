# Unloads ComfyUI models after it has been idle for $IdleMinutes.
# Run every few minutes by the "ComfyUI Idle Unload" scheduled task.
param([int]$IdleMinutes = 15, [string]$Url = 'http://127.0.0.1:8188')

$log = Join-Path $PSScriptRoot 'comfy_idle_unload.log'
try {
    $q = Invoke-RestMethod "$Url/queue" -TimeoutSec 5
    if ($q.queue_running.Count -or $q.queue_pending.Count) { exit }

    # Timestamp of the most recent job event (ms since epoch)
    $h = Invoke-RestMethod "$Url/history?max_items=1" -TimeoutSec 5
    $last = 0
    foreach ($p in $h.PSObject.Properties) {
        foreach ($m in $p.Value.status.messages) { if ($m[1].timestamp -gt $last) { $last = $m[1].timestamp } }
    }
    $idle = ([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() - $last) / 60000
    if ($idle -lt $IdleMinutes) { exit }

    # Skip if already unloaded (process private memory under 7 GB)
    $pid_ = (Get-NetTCPConnection -LocalPort ([uri]$Url).Port -State Listen -ErrorAction Stop).OwningProcess | Select-Object -First 1
    $gb = (Get-Process -Id $pid_).PrivateMemorySize64 / 1GB
    if ($gb -lt 7) { exit }

    Invoke-RestMethod -Method Post "$Url/free" -ContentType 'application/json' -Body '{"unload_models":true,"free_memory":true}' -TimeoutSec 10 | Out-Null
    Add-Content $log ("{0}  idle {1:N0} min, {2:N1} GB -> unloaded" -f (Get-Date -Format s), $idle, $gb)
} catch {
    # ComfyUI not running or unreachable: nothing to do
}
