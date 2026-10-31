@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist "art\icon.png" (
  echo Missing art\icon.png
  exit /b 1
)
if not exist "art\cover.png" (
  echo Missing art\cover.png
  exit /b 1
)
where py >nul 2>&1 && (set "PY=py -3") || (set "PY=python")
if not exist ".venv\Scripts\python.exe" (
  %PY% -m venv .venv
  if errorlevel 1 exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
echo LoraDatasetGate uses art\icon.png and art\cover.png
echo Opening http://127.0.0.1:8765/
start "Open LoraDatasetGate" /MIN .venv\Scripts\python.exe scripts\open_browser.py 8765
python -m loradatasetgate --host 127.0.0.1 --port 8765 --no-browser
