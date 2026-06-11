@echo off
setlocal
cd /d "%~dp0.."
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo Ambiente Python nao encontrado: %PY%
  pause
  exit /b 1
)

echo ============================================================
echo Pipeline de curadoria: rim U-Net, interno U-Net/DeepLab,
echo comparacao, limpeza, miniaturas e interface.
echo ============================================================
echo.

echo [1/7] Gerando mascaras renais com U-Net, Lanczos, CLAHE e limiar 0.90...
"%PY%" src\segmentation\tools\predict\missing_kidney.py ^
  --dataset-root dataset_aumentado\dataset_geral ^
  --model unet ^
  --checkpoint models\kidneyus_capsule_unet.pth ^
  --confidence-threshold 0.90 ^
  --super-resolution lanczos ^
  --clahe ^
  --refresh-all
if errorlevel 1 goto erro

echo [2/7] Gerando mascaras internas com U-Net...
"%PY%" src\segmentation\tools\predict\inner_unet.py
if errorlevel 1 goto erro

echo [3/7] Gerando mascaras internas com DeepLab...
"%PY%" src\segmentation\tools\predict\inner_deeplab.py
if errorlevel 1 goto erro

echo [4/7] Comparando U-Net x DeepLab...
"%PY%" src\segmentation\tools\compare\inner_models.py
if errorlevel 1 goto erro

echo [5/7] Limpando rim e recortando mascaras internas...
"%PY%" src\segmentation\tools\post\clean_kidney.py
if errorlevel 1 goto erro
"%PY%" src\segmentation\tools\post\constrain_inner.py
if errorlevel 1 goto erro

echo [6/7] Reconstruindo miniaturas...
"%PY%" engenharia_dataset\build_curation_thumbnails.py
if errorlevel 1 goto erro

echo [7/7] Reiniciando interface...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$port=8765; $connections=Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue; foreach ($conn in $connections) { Stop-Process -Id $conn.OwningProcess -Force }; Start-Sleep -Seconds 1; Start-Process -FilePath '.\.venv\Scripts\python.exe' -ArgumentList 'curadoria_web\app.py','--port','8765' -WorkingDirectory (Get-Location).Path -WindowStyle Hidden"
if errorlevel 1 goto erro

start "" "http://127.0.0.1:8765/"

echo.
echo Pipeline finalizado com sucesso.
pause
exit /b 0

:erro
echo.
echo Pipeline interrompido por erro.
pause
exit /b 1
