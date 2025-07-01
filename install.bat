@echo off
REM This batch script installs the necessary dependencies for the YouTube Transcript Extractor.

REM Ensure pip is up-to-date
python -m pip install --upgrade pip

REM Install required packages
pip install -r requirements.txt

echo Installation complete. You can now run the YouTube Transcript Extractor.