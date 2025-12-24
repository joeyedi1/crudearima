@echo off
cd /d "%~dp0"
:: Paste your Gemini API Key below (remove ":: " if you want to use it here instead of env vars)
set GEMINI_API_KEY=AIzaSyCdvYtdC3CI3jd1FwDNFGvDteu05dEosU8
call .venv\Scripts\activate.bat
echo Starting Daily Report Automation: %date% %time%
python src/fetch_data.py
if %errorlevel% neq 0 (
    echo Fetch Data Failed!
    pause
    exit /b %errorlevel%
)
python src/generate_report.py
if %errorlevel% neq 0 (
    echo Generate Report Failed!
    pause
    exit /b %errorlevel%
)
echo Report Generated Successfully!
pause
