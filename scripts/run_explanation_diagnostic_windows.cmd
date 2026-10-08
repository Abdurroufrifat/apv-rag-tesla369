@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv\Scripts\python.exe in this project folder.
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install --no-deps -e . || exit /b 1
if not exist "artifacts\explanation_diagnostic_v1\output_manifest.json" (
  ".venv\Scripts\python.exe" scripts\run_explanation_diagnostic.py || exit /b 1
)
".venv\Scripts\python.exe" scripts\verify_explanation_diagnostic.py || exit /b 1
powershell -NoProfile -Command "Compress-Archive -LiteralPath 'artifacts\explanation_diagnostic_v1\input_manifest.json','artifacts\explanation_diagnostic_v1\pairs.json','artifacts\explanation_diagnostic_v1\summary.json','artifacts\explanation_diagnostic_v1\output_manifest.json','artifacts\explanation_diagnostic_verification_v1\verification.json','artifacts\explanation_diagnostic_verification_v1\RESULTS.md','artifacts\explanation_diagnostic_verification_v1\audit_manifest.json' -DestinationPath 'explanation_diagnostic_outputs.zip' -Force" || exit /b 1
echo Explanation diagnostic verified. Upload explanation_diagnostic_outputs.zip.
