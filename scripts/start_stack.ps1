# Start all IR services + Streamlit UI (run from project root).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "Missing venv Python at $Python"
}

# Load .env into process if present
$EnvFile = Join-Path $Root ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*#' -or $_ -notmatch '=') { return }
        $k, $v = $_ -split '=', 2
        Set-Item -Path "env:$($k.Trim())" -Value $v.Trim()
    }
}

function Stop-Port($port) {
    Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique |
        ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
}

8000, 8002, 8003, 8004, 8005, 8501 | ForEach-Object { Stop-Port $_ }
Start-Sleep -Seconds 1

Get-ChildItem -Path $Root -Recurse -Directory -Filter __pycache__ |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Start-Process -FilePath $Python -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000" -WorkingDirectory (Join-Path $Root "preprocessing_service")
Start-Process -FilePath $Python -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8002" -WorkingDirectory (Join-Path $Root "retrieval_service")
Start-Process -FilePath $Python -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8003" -WorkingDirectory (Join-Path $Root "query_refinement_service")
Start-Process -FilePath $Python -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8004" -WorkingDirectory (Join-Path $Root "personalization_service")
Start-Process -FilePath $Python -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8005" -WorkingDirectory (Join-Path $Root "clustering_service")

Start-Sleep -Seconds 3
Start-Process -FilePath $Python -ArgumentList "-m", "streamlit", "run", "app_ui.py", "--server.port", "8501" -WorkingDirectory $Root

Write-Host "Started: preprocessing:8000 retrieval:8002 refinement:8003 personalization:8004 clustering:8005 streamlit:8501"
