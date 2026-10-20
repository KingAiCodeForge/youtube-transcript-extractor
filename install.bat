@echo off
REM Use the same interpreter for pip and resolve paths from this installer.
python -m pip install "%~dp0." %*
exit /b %errorlevel%
