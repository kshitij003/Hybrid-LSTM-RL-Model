@echo off
REM Setup script for ML Service Phase 1
REM This script installs dependencies and runs basic tests

echo ================================================
echo  ML Service Setup - Phase 1
echo ================================================
echo.

echo [1/3] Installing Python dependencies...
echo.
cd ml_service
pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to install dependencies
    echo Please check that Python and pip are installed
    pause
    exit /b 1
)

echo.
echo [2/3] Creating directory structure...
mkdir models\saved_models 2>nul
mkdir data\cache 2>nul

echo.
echo [3/3] Setup complete!
echo.
echo ================================================
echo  Ready to start Flask server
echo ================================================
echo.
echo To start the server:
echo   cd ml_service
echo   python app.py
echo.
echo Then test with:
echo   python test_service.py
echo.
echo Or visit: http://localhost:8000/health
echo.

pause
