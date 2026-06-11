@echo off
setlocal
cd /d "%~dp0.."
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo Ambiente Python nao encontrado: %PY%
  pause
  exit /b 1
)

echo Gerando mascaras renais com U-Net, Lanczos, CLAHE e limiar 0.90...
"%PY%" src\segmentation\tools\predict\missing_kidney.py ^
  --dataset-root dataset_aumentado\dataset_geral ^
  --model unet ^
  --checkpoint models\kidneyus_capsule_unet.pth ^
  --confidence-threshold 0.90 ^
  --super-resolution lanczos ^
  --clahe ^
  --refresh-all

echo.
echo Finalizado.
pause
