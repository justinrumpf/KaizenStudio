@echo off
echo Starting Background Remover...
echo.
echo Server will be available at: http://localhost:5000
echo.
cd /d "%~dp0"
call venv\Scripts\activate
python app.py
pause
