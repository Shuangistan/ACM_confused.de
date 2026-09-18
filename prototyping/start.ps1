<#
  Start the server and open a browser. The PowerShell counterpart of start.sh.

  Usage:
      .\start.ps1                 # whatever LLM_PROVIDER says, or Anthropic
      .\start.ps1 -Provider openai
      .\start.ps1 -Provider openai -Port 8080 -NoBrowser
#>
[CmdletBinding()]
param(
    [ValidateSet('anthropic','openai')] [string] $Provider,
    [int]    $Port = 8000,
    [switch] $NoBrowser
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if ($Provider) { $env:LLM_PROVIDER = $Provider }
if (-not $env:LLM_PROVIDER) { $env:LLM_PROVIDER = 'anthropic' }

# Windows buffers stdout when it is redirected, so the log stays empty until the
# process ends -- exactly when you no longer need it. This turns that off.
$env:PYTHONUNBUFFERED = '1'

$python = if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe' }
          elseif (Test-Path '.venv\Scripts\python.exe') { '.venv\Scripts\python.exe' }
          else { 'python' }

if (Test-Path '.server.pid') {
    $old = Get-Content '.server.pid' -ErrorAction SilentlyContinue
    if ($old -and (Get-Process -Id $old -ErrorAction SilentlyContinue)) {
        Write-Host "A server is already running (PID $old). Run .\stop.ps1 first." -ForegroundColor Yellow
        exit 1
    }
    Remove-Item '.server.pid' -Force -ErrorAction SilentlyContinue
}

$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    Write-Host "Port $Port is already in use. Pass -Port to choose another." -ForegroundColor Yellow
    exit 1
}

Write-Host "Starting confused.de  (provider: $($env:LLM_PROVIDER), port: $Port)" -ForegroundColor Cyan

$proc = Start-Process -FilePath $python `
    -ArgumentList @('-m','uvicorn','server:app','--host','127.0.0.1','--port',"$Port") `
    -RedirectStandardOutput '.server.log' -RedirectStandardError '.server.err' `
    -NoNewWindow -PassThru
$proc.Id | Out-File '.server.pid' -Encoding ascii

# Poll rather than sleep a fixed time: a cold import of langgraph takes a few
# seconds on Windows, and opening the browser before the port answers shows the
# reader a connection error for a server that is about to work.
$url = "http://127.0.0.1:$Port"
$ready = $false
foreach ($i in 1..40) {
    Start-Sleep -Milliseconds 500
    if ($proc.HasExited) { break }
    try {
        Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 | Out-Null
        $ready = $true; break
    } catch { }
}

if (-not $ready) {
    Write-Host "The server did not come up. Last lines of .server.err:" -ForegroundColor Red
    if (Test-Path '.server.err') { Get-Content '.server.err' -Tail 20 }
    if (Test-Path '.server.log') { Get-Content '.server.log' -Tail 20 }
    exit 1
}

Write-Host "Ready: $url" -ForegroundColor Green
Write-Host "Logs:  .server.log / .server.err      Stop: .\stop.ps1"
if (-not $NoBrowser) { Start-Process $url }
