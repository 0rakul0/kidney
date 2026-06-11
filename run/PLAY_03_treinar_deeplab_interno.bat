@echo off
setlocal
cd /d "%~dp0.."
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo Ambiente Python nao encontrado: %PY%
  pause
  exit /b 1
)

echo Treinando DeepLab intrarrenal: cortex, medulla e CEC...
"%PY%" src\segmentation\experiments\train_inner_deeplab.py

echo.
echo Finalizado.
pause
