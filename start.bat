@echo off
REM ===================================================================
REM  Ollama Voice Assistant - launcher (double-click to run)
REM
REM  This script:
REM   1. Switches to the project directory (so config.ini / models/ resolve)
REM   2. Picks a working Python interpreter
REM   3. Verifies Ollama is reachable on http://localhost:11434
REM   4. Starts run.py
REM   5. Pauses on exit so you can read errors before the window closes
REM ===================================================================

setlocal enableextensions

REM --- 1. cd to the directory this .bat lives in --------------------
cd /d "%~dp0"

title Ollama Voice Assistant

echo.
echo  =====================================================
echo   Ollama Voice Assistant
echo  =====================================================
echo.

REM --- 2. Find a Python interpreter ---------------------------------
set "PYTHON_EXE="

REM Prefer a project-local venv if one exists.
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
)

REM Otherwise fall back to whatever `python` is on PATH.
if "%PYTHON_EXE%"=="" (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set "PYTHON_EXE=python"
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] No Python interpreter found.
    echo         Install Python 3.10+ or create a venv in this folder, then re-run.
    goto :end
)

echo  [info] Using Python: %PYTHON_EXE%

REM --- 3. Probe Ollama ---------------------------------------------
echo  [info] Checking Ollama at http://localhost:11434 ...
%PYTHON_EXE% -c "import urllib.request,sys;urllib.request.urlopen('http://localhost:11434/api/tags',timeout=3)" 2>nul
if errorlevel 1 (
    echo.
    echo  [WARN] Ollama is not responding on http://localhost:11434.
    echo         Open the Ollama desktop app or run `ollama serve` in another
    echo         terminal, then come back. The assistant will still launch but
    echo         will not be able to answer until Ollama is up.
    echo.
    timeout /t 3 >nul
)

REM --- 4. Run --------------------------------------------------------
echo.
echo  [info] Starting voice assistant... say your wake word when you see
echo         'Ready! Listening for ...'
echo.
echo  Tip: speak the word "goodbye" or press Ctrl+C to stop.
echo.

%PYTHON_EXE% run.py %*
set "EXIT_CODE=%errorlevel%"

:end
echo.
if not "%EXIT_CODE%"=="" (
    echo  Exit code: %EXIT_CODE%
)
echo.
echo  Press any key to close this window...
pause >nul
endlocal
