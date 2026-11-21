@echo off
chcp 65001 > nul
title Tawazon AI Platform
cd /d "%~dp0"
echo ===================================================
echo   Starting Tawazon Platform - FastAPI Backend...
echo ===================================================
start "" "frontend\login.html"
py -3.13 -m uvicorn backend.main:app --reload --port 8000
pause
