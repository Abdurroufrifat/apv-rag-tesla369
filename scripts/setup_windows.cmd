@echo off
setlocal
cd /d "%~dp0\.."

echo [1/5] Creating Python environment...
py -3 -m venv .venv || exit /b 1

echo [2/5] Activating environment...
call .venv\Scripts\activate.bat || exit /b 1

echo [3/5] Installing local dependencies...
python -m pip install --upgrade pip || exit /b 1
python -m pip install -r requirements-local.txt || exit /b 1
python -m pip install --no-deps -e . || exit /b 1

echo [4/5] Checking the environment and tests...
python scripts\check_environment.py || exit /b 1
python -m pytest -q || exit /b 1

echo [5/5] Validating starter records...
python scripts\validate_claims.py data\templates\claims_template.csv || exit /b 1

echo.
echo APV-RAG setup completed successfully.
endlocal
