#!/bin/bash

# This script installs the necessary dependencies for the YouTube Transcript Extractor project.

# Ensure pip is up-to-date
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

echo "Installation complete. You can now run the YouTube Transcript Extractor."