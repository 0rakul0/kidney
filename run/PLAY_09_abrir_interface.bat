@echo off
setlocal
cd /d "%~dp0.."
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo Ambiente Python nao encontrado: %PY%
  pause
  exit /b 1
)

echo Reiniciando interface local na porta 8765...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$port=8765; $connections=Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue; foreach ($conn in $connections) { Stop-Process -Id $conn.OwningProcess -Force }; Start-Sleep -Seconds 1; Start-Process -FilePath '.\.venv\Scripts\python.exe' -ArgumentList 'curadoria_web\app.py','--port','8765' -WorkingDirectory (Get-Location).Path -WindowStyle Hidden"

start "" "http://127.0.0.1:8765/"

echo.
echo Interface aberta em http://127.0.0.1:8765/
pause
