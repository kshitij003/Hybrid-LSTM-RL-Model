@echo off
TITLE Hybrid AI Trading System - Launcher
COLOR 0A

echo ======================================================
echo    🚀 HYBRID LSTM+RL TRADING SYSTEM - DEMO MODE
echo ======================================================
echo.

:: 1. Start ML Service (Flask)
echo [1/3] Starting ML Service (Flask on Port 8000)...
cd ml_service
start "Flask ML Service" cmd /k "python app.py"
echo Done.
echo.

:: 2. Start Trading Backend (Spring Boot)
echo [2/3] Starting Trading Backend (Spring Boot on Port 8080)...
cd ..\trading_backend\trading_backend
start "Spring Boot Backend" cmd /k "mvn spring-boot:run"
echo Done.
echo.

:: 3. Launch Dashboard
echo [3/3] Opening Dashboard in 10 seconds...
timeout /t 10 /nobreak > nul
start http://localhost:8080/index.html

echo.
echo ======================================================
echo    ✅ SYSTEM STARTED SUCCESSFULLY
echo ======================================================
echo.
echo Keep the other two windows open! 
echo Dashboard: http://localhost:8080/index.html
echo.
pause
