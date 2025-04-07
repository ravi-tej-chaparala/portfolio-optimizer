@echo off
echo ===================================================
echo STARTING PORTFOLIO RECOMMENDATION APP
echo ===================================================

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Setting environment variables...
set PYTHONWARNINGS=ignore::DeprecationWarning
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1
set PYTHONHTTPSVERIFY=0
set PORTFOLIO_APP_ENV=production
set PYTHONPATH=%CD%\src

echo Checking for dependencies...
pip freeze | findstr "yfinance" >nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing required dependencies...
    pip install yfinance pandas numpy scikit-learn flask huggingface_hub python-dotenv
)

echo Creating cache directory...
if not exist "src\cache" mkdir src\cache

echo Starting the app...
cd src\web
python -W ignore::UserWarning -W ignore::FutureWarning -W ignore::DeprecationWarning app.py

echo.
echo App stopped.

call deactivate 