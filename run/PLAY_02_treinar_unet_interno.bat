@echo off
setlocal
cd /d "%~dp0.."
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo Ambiente Python nao encontrado: %PY%
  pause
  exit /b 1
)

echo Treinando U-Net intrarrenal: cortex, medulla e CEC...
"%PY%" src\segmentation\experiments\train_inner_unet.py

echo.
echo Finalizado.
pause
