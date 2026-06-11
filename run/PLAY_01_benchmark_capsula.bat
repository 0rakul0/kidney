@echo off
setlocal
cd /d "%~dp0.."
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo Ambiente Python nao encontrado: %PY%
  pause
  exit /b 1
)

echo Rodando benchmark da capsula renal...
"%PY%" src\segmentation\tools\bench\capsule.py

echo.
echo Finalizado.
pause
