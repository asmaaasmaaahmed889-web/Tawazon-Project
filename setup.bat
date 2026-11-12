@echo off
chcp 65001 >nul
echo ============================================
echo   Tawazon - One-time setup
echo ============================================

for /f "delims=" %%i in ('where python') do set PYPATH=%%i
echo %PYPATH% | find /i "WindowsApps" >nul
if not errorlevel 1 (
    echo [WARNING] You are using the Microsoft Store Python.
    echo It is blocked on some laptops. Install Python 3.13 from python.org instead.
    pause
)
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not installed or not on PATH.
    echo Install Python 3.13 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment (.venv)...
python -m venv .venv
if errorlevel 1 (
    echo [ERROR] Could not create the virtual environment.
    pause
    exit /b 1
)

echo [2/3] Installing requirements...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Installing requirements failed. Copy the error and send it to the team.
    pause
    exit /b 1
)

echo [3/3] Setup finished successfully.
echo Next time, just double-click run.bat
pause
