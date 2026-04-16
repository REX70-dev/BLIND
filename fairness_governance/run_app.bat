@echo off
setlocal

cd /d C:\BLIND

if not exist .venv\Scripts\python.exe (
    echo Virtual environment not found.
    echo Run fairness_governance\setup_windows_py312.bat first.
    exit /b 1
)

.\.venv\Scripts\python.exe -m streamlit run fairness_governance\app.py

