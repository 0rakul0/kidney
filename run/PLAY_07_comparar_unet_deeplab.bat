@echo off
setlocal
cd /d "%~dp0.."
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo Ambiente Python nao encontrado: %PY%
  pause
  exit /b 1
)

echo Comparando divergencias U-Net x DeepLab...
"%PY%" src\segmentation\tools\compare\inner_models.py

echo.
echo Finalizado.
pause
