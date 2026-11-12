@echo off
chcp 65001 >nul
if not exist .venv\Scripts\activate.bat (
    echo [ERROR] Setup not done yet. Double-click setup.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
set DISABLE_SQLALCHEMY_CEXT_RUNTIME=1
echo ============================================
echo   Tawazon server is starting...
echo   Open frontend\login.html in your browser
echo   Press CTRL+C here to stop the server
echo ============================================
python -m uvicorn backend.main:app --reload
pause
