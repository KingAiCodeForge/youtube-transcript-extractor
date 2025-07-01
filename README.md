# YouTube Transcript Extractor

## Overview
The YouTube Transcript Extractor is a Python application that allows users to fetch transcripts from YouTube videos using the YouTube Transcript API. The application features a user-friendly interface built with PySide6, enabling users to easily input video links and retrieve transcripts.

## 📷 Application Screenshot

This is the working YouTube Transcript Extractor in action:

![YouTube Transcript Extractor UI](https://github.com/KingAiCodeForge/youtube-transcript-extractor/blob/main/screenshot-app-ui.png?raw=true)

## Project Structure
```
youtube-transcript-extractor
├── youtubescraper.py       # Main application code
├── requirements.txt        # List of dependencies
├── install.sh              # Cross-platform installation script
├── install.bat             # Windows installation script
└── README.md               # Project documentation
```

## Prerequisites
- Python 3.6 or higher
- pip (Python package installer)

## Installation Instructions

### Clone the Repository
First, clone the repository to your local machine using the following command:
```bash
git clone https://github.com/yourusername/youtube-transcript-extractor.git
cd youtube-transcript-extractor
```

### Install Dependencies
You can install the required dependencies using one of the following methods:

#### Method 1: Using `install.sh` (Cross-platform)
1. Open a terminal.
2. Navigate to the project directory.
3. Run the installation script:
   ```bash
   bash install.sh
   ```

#### Method 2: Using `install.bat` (Windows)
1. Right-click on `install.bat` and select "Run as administrator" or double-click it to execute the script. This will ensure that pip is up-to-date and install the necessary dependencies.

### Manual Installation
Alternatively, you can manually install the dependencies by running:
```bash
pip install -r requirements.txt
```

## Running the Application
After installing the dependencies, you can run the application using the following command:
```bash
python src/youtubescraper.py
```

## Usage
1. Paste YouTube video links into the input area.
2. Click on "Extract Transcripts" to fetch the transcripts.
3. Once the transcripts are fetched, you can save them as a Markdown file.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api) for providing the transcript fetching functionality.
- [PySide6](https://pyside.org/) for the GUI framework.
