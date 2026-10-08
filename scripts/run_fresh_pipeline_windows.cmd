@echo off
setlocal
cd /d "%~dp0.." || exit /b 1
if not exist ".venv\Scripts\python.exe" (
  echo Existing .venv is missing. Use the original project environment.
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install --no-deps -e . || exit /b 1
if not exist "artifacts\fresh_pipeline_v1\output_manifest.json" (
  ".venv\Scripts\python.exe" scripts\run_fresh_pipeline.py || exit /b 1
)
".venv\Scripts\python.exe" scripts\verify_fresh_pipeline.py || exit /b 1
powershell -NoProfile -Command "$items = @('input_manifest.json','feature_cache.json','execution_progress.json','predictions.json','responses.json','summary.json','output_manifest.json') | ForEach-Object { Join-Path 'artifacts\fresh_pipeline_v1' $_ }; $items += 'artifacts\fresh_pipeline_verification_v1'; Compress-Archive -LiteralPath $items -DestinationPath 'fresh_pipeline_outputs.zip' -Force" || exit /b 1
echo Send fresh_pipeline_outputs.zip. Keep the existing local model and SQLite caches.
exit /b 0
