@echo off
setlocal
echo ==========================================
echo    AgroScan AI Dashboard Starter
echo ==========================================

if exist .venv\Scripts\python.exe (
    echo Using Virtual Environment (.venv)...
    .\.venv\Scripts\python.exe -m streamlit run scripts/app.py
) else (
    echo Using System Python...
    python -m streamlit run scripts/app.py
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Failed to start the dashboard. 
    echo Please make sure streamlit is installed.
    echo Run: .\.venv\Scripts\python.exe -m pip install streamlit pandas plotly
)

pause
