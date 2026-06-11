@echo off
setlocal
cd /d "%~dp0.."
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo Ambiente Python nao encontrado: %PY%
  pause
  exit /b 1
)

echo Gerando mascaras internas com DeepLab...
"%PY%" src\segmentation\tools\predict\inner_deeplab.py

echo.
echo Finalizado.
pause
