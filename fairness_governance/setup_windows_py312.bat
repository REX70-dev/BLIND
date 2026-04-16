@echo off
setlocal

cd /d C:\BLIND

where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher "py" was not found.
    echo Install Python 3.12 from https://www.python.org/downloads/windows/
    exit /b 1
)

py -3.12 --version >nul 2>nul
if errorlevel 1 (
    echo Python 3.12 was not found.
    echo Install Python 3.12 from https://www.python.org/downloads/windows/
    echo During install, enable "Add python.exe to PATH" if offered.
    exit /b 1
)

if exist .venv (
    echo Existing .venv found. Reusing it.
) else (
    py -3.12 -m venv .venv
)

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r fairness_governance\requirements.txt

echo.
echo Setup complete. Start the app with:
echo fairness_governance\run_app.bat

