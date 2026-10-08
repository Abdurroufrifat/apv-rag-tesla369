@echo off
setlocal
cd /d "%~dp0.." || exit /b 1
if not exist ".venv\Scripts\python.exe" (
  echo Use the existing project environment; .venv is missing.
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install --no-deps -e . || exit /b 1
if not exist "artifacts\text_stress_v1\output_manifest.json" (
  ".venv\Scripts\python.exe" scripts\run_text_stress.py || exit /b 1
)
".venv\Scripts\python.exe" scripts\verify_text_stress.py || exit /b 1
powershell -NoProfile -Command "$items = @('input_manifest.json','contexts.json','execution_progress.json','feature_cache.json','predictions.json','responses.json','summary.json','output_manifest.json') | ForEach-Object { Join-Path 'artifacts\text_stress_v1' $_ }; $items += 'artifacts\text_stress_verification_v1'; Compress-Archive -LiteralPath $items -DestinationPath 'text_stress_outputs.zip' -Force" || exit /b 1
echo Send text_stress_outputs.zip when complete. Interrupted runs resume with the same identity.
exit /b 0
