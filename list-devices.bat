@echo off
REM Show audio input/output devices and their indices, then pause so you can
REM copy them into config.ini (device_index, piper_output_device_index).
call "%~dp0start.bat" --list-devices --list-output-devices
