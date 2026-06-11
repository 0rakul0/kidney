@echo off
setlocal
cd /d "%~dp0.."
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo Ambiente Python nao encontrado: %PY%
  pause
  exit /b 1
)

echo Limpando mascaras renais: manter apenas maior componente...
"%PY%" src\segmentation\tools\post\clean_kidney.py
if errorlevel 1 goto erro

echo Recortando mascaras internas pela ROI renal limpa...
"%PY%" src\segmentation\tools\post\constrain_inner.py
if errorlevel 1 goto erro

echo Reconstruindo miniaturas da curadoria...
"%PY%" engenharia_dataset\build_curation_thumbnails.py
if errorlevel 1 goto erro

echo Reiniciando interface local na porta 8765...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$port=8765; $connections=Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue; foreach ($conn in $connections) { Stop-Process -Id $conn.OwningProcess -Force }; Start-Sleep -Seconds 1; Start-Process -FilePath '.\.venv\Scripts\python.exe' -ArgumentList 'curadoria_web\app.py','--port','8765' -WorkingDirectory (Get-Location).Path -WindowStyle Hidden"
if errorlevel 1 goto erro

start "" "http://127.0.0.1:8765/"

echo.
echo Finalizado. Interface aberta em http://127.0.0.1:8765/
pause
exit /b 0

:erro
echo.
echo Ocorreu erro durante a execucao.
pause
exit /b 1
