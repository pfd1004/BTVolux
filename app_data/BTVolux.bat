@echo off
cd /d "%~dp0"

set APP_ROOT=%~dp0
set PATH=%APP_ROOT%runtime\env_app;%APP_ROOT%runtime\env_app\Scripts;%PATH%

set TF_CPP_MIN_LOG_LEVEL=2
REM set TF_ENABLE_ONEDNN_OPTS=0

echo Iniciando BTVolux...
echo.

"%APP_ROOT%runtime\env_app\python.exe" -m shiny run --host 127.0.0.1 --port 8000 --launch-browser app.py

pause