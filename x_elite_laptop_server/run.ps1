#Requires -Version 5.1
<#
.SYNOPSIS
    Dependency bootstrap helper -- installs required packages via uv.

.DESCRIPTION
    Creates a local .venv and installs every package listed in
    requirements.txt using uv. Run this once before running the sample.

    NPU note: this sample needs a NATIVE ARM64 Python 3.10-3.12 (the
    geniex wheel is win-arm64 only and will not load on an emulated
    x86-64 Python). The default -Python 3.12 targets that.

.EXAMPLE
    # One-time setup
    .\run.ps1

    # Start the server (binds 0.0.0.0:8000 by default)
    .venv\Scripts\python.exe serve_qwen_vl.py

    # Custom port
    .venv\Scripts\python.exe serve_qwen_vl.py --port 8090
#>

[CmdletBinding()]
param(
    [string]$Python = "3.12"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Ok($msg)   { Write-Host "  [OK]    $msg" -ForegroundColor Green }
function Write-Info($msg) { Write-Host "  [INFO]  $msg" }

$ReqFile = Join-Path $PSScriptRoot "requirements.txt"
$VenvDir = Join-Path $PSScriptRoot ".venv"

# 1. Verify requirements.txt exists
if (-not (Test-Path $ReqFile)) {
    Write-Error "requirements.txt not found: $ReqFile"
    exit 1
}

# 2. Verify uv -- auto-install if missing
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Info "uv not found -- installing ..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","User") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH","Machine")
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Error "uv install failed. Install manually: https://docs.astral.sh/uv/"
        exit 1
    }
}
Write-Ok "uv $(uv --version)"

# 3. Create venv if absent
if (-not (Test-Path $VenvDir)) {
    Write-Info "Creating .venv (Python $Python) ..."
    uv venv "$VenvDir" --python $Python
    if ($LASTEXITCODE -ne 0) { Write-Error "uv venv failed."; exit 1 }
    Write-Ok ".venv created"
} else {
    Write-Ok ".venv already exists -- skipping creation"
}

# 4. Install dependencies
Write-Info "Installing dependencies from requirements.txt ..."
uv pip install --system-certs --python "$VenvDir\Scripts\python.exe" -r "$ReqFile"
if ($LASTEXITCODE -ne 0) { Write-Error "uv pip install failed."; exit 1 }
Write-Ok "All dependencies installed"

Write-Host ""
Write-Host "  Setup complete. Start the server:" -ForegroundColor Cyan
Write-Host ""
Write-Host "    .venv\Scripts\python.exe serve_qwen_vl.py"
Write-Host ""
