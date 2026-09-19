@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE="

where python >nul 2>nul
if not errorlevel 1 set "PYTHON_EXE=python"

if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python313\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python313\python.exe"

if not defined PYTHON_EXE (
    echo Python was not found.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    "%PYTHON_EXE%" -m venv .venv
    if errorlevel 1 goto error
)

set "PYTHON=.venv\Scripts\python.exe"

"%PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto error

"%PYTHON%" -m pip uninstall -y opencv-python-headless opencv-contrib-python-headless
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto error

"%PYTHON%" -c "import cv2, flygym_gymnasium; print('All requirements installed.')"
if errorlevel 1 goto error

"%PYTHON%" main.py
pause
exit /b 0

:error
echo Setup failed. Check the error above.
pause
exit /b 1