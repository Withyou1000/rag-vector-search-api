$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

# Start from project venv to avoid using a Python without dependencies.
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
& $pythonPath -m uvicorn app.main:app --host 127.0.0.1 --port 8000
