@echo off
REM Debug launcher - same as start.bat but adds --debug for verbose logging.
REM Use this when tuning wake-word threshold, mic gain, or VAD settings.
call "%~dp0start.bat" --debug
