<#
  Stop the server started by start.ps1. The counterpart of stop.sh.
#>
[CmdletBinding()]
param([int] $Port = 8000)

$ErrorActionPreference = 'SilentlyContinue'
Set-Location -LiteralPath $PSScriptRoot

$stopped = $false

if (Test-Path '.server.pid') {
    $serverPid = Get-Content '.server.pid'
    if ($serverPid -and (Get-Process -Id $serverPid -ErrorAction SilentlyContinue)) {
        Stop-Process -Id $serverPid -Force
        Write-Host "Stopped PID $serverPid." -ForegroundColor Green
        $stopped = $true
    }
    Remove-Item '.server.pid' -Force
}

# The pid file can be stale -- a crash, or a server started by hand. Fall back
# to whatever is actually holding the port, so "stop" means stopped.
if (-not $stopped) {
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        foreach ($owner in ($conn.OwningProcess | Select-Object -Unique)) {
            Stop-Process -Id $owner -Force
            Write-Host "Stopped PID $owner (held port $Port)." -ForegroundColor Green
            $stopped = $true
        }
    }
}

if (-not $stopped) { Write-Host "Nothing was running." }
