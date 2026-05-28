$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

# Config is loaded by app/config.py from code defaults and .env.
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
& $pythonPath -m uvicorn app.asgi:app --host 127.0.0.1 --port 8000 --reload
