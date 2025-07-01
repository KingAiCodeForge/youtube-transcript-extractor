# filepath: youtube-transcript-extractor/youtube-transcript-extractor/src/youtubescraper.py
import sys
import os
import logging

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, QRunnable, QThreadPool, Signal, QObject

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

time_format = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s — %(message)s',
    datefmt=time_format
)
logger = logging.getLogger(__name__)

class WorkerSignals(QObject):
    progress = Signal(str, str)
    result = Signal(str, list)
    error = Signal(str, str)

class TranscriptWorker(QRunnable):
    def __init__(self, video_id):
        super().__init__()
        self.video_id = video_id
        self.signals = WorkerSignals()

    def run(self):
        logger.info(f"[{self.video_id}] Worker started")
        self.signals.progress.emit(self.video_id, "Fetching")
        try:
            transcript = YouTubeTranscriptApi.get_transcript(self.video_id)
            logger.info(f"[{self.video_id}] Transcript fetched ({len(transcript)} entries)")
            self.signals.result.emit(self.video_id, transcript)
            self.signals.progress.emit(self.video_id, "Done")
        except TranscriptsDisabled:
            msg = "Transcripts disabled"
            logger.warning(f"[{self.video_id}] {msg}")
            self.signals.error.emit(self.video_id, msg)
            self.signals.progress.emit(self.video_id, "Failed")
        except NoTranscriptFound:
            msg = "No transcript found"
            logger.warning(f"[{self.video_id}] {msg}")
            self.signals.error.emit(self.video_id, msg)
            self.signals.progress.emit(self.video_id, "Failed")
        except Exception as e:
            logger.exception(f"[{self.video_id}] Unexpected error: {e}")
            self.signals.error.emit(self.video_id, str(e))
            self.signals.progress.emit(self.video_id, "Failed")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube Transcript Extractor")
        self.resize(800, 600)

        self.threadpool = QThreadPool()
        self.transcripts = {}

        self.input_edit = QPlainTextEdit(placeholderText="Paste YouTube links here, one per line...")
        self.extract_button = QPushButton("🔍 Extract Transcripts")
        self.save_button = QPushButton("💾 Save as .md")
        self.save_button.setEnabled(False)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Video ID", "Status"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.extract_button)
        controls_layout.addWidget(self.save_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.input_edit)
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.table)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.extract_button.clicked.connect(self.start_extraction)
        self.save_button.clicked.connect(self.save_markdown)

    def start_extraction(self):
        raw_text = self.input_edit.toPlainText().strip()
        if not raw_text:
            QMessageBox.warning(self, "Input Error", "Please paste at least one YouTube link.")
            return

        links = [line.strip() for line in raw_text.splitlines() if line.strip()]
        video_ids = []
        for link in links:
            if "youtube.com/watch" in link or "youtu.be/" in link:
                vid = link.split("v=")[-1].split("&")[0] if "v=" in link else link.split('/')[-1]
                video_ids.append(vid)

        if not video_ids:
            QMessageBox.warning(self, "Input Error", "No valid YouTube URLs found.")
            return

        logger.info(f"Starting extraction for {len(video_ids)} video(s): {video_ids}")

        self.table.setRowCount(0)
        self.transcripts.clear()
        self.save_button.setEnabled(False)

        for vid in video_ids:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(vid))
            self.table.setItem(row, 1, QTableWidgetItem("⏳ Queued"))

            worker = TranscriptWorker(vid)
            worker.signals.progress.connect(self.update_status)
            worker.signals.result.connect(self.store_transcript)
            worker.signals.error.connect(self.handle_error)
            self.threadpool.start(worker)
            logger.info(f"[{vid}] Worker dispatched")

    def update_status(self, video_id, status):
        emoji_status = {
            "Queued": "⏳ Queued",
            "Fetching": "🔄 Fetching",
            "Done": "✅ Done",
            "Failed": "❌ Failed"
        }
        logger.info(f"Status update for {video_id}: {status}")
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == video_id:
                self.table.setItem(row, 1, QTableWidgetItem(emoji_status.get(status, status)))
                break

        all_done = all(
            self.table.item(r,1).text() in ("✅ Done", "❌ Failed")
            for r in range(self.table.rowCount())
        )
        if all_done:
            logger.info("All tasks completed, enabling Save button")
            self.save_button.setEnabled(True)

    def store_transcript(self, video_id, transcript):
        self.transcripts[video_id] = transcript
        logger.info(f"Transcript stored for {video_id} ({len(transcript)} entries)")

    def handle_error(self, video_id, error_msg):
        logger.error(f"Error for {video_id}: {error_msg}")

    def save_markdown(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Markdown", os.getenv('HOME'), "Markdown Files (*.md)"
        )
        if not path:
            logger.info("Save operation cancelled by user")
            return

        logger.info(f"Saving transcripts to {path}")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                for vid, transcript in self.transcripts.items():
                    f.write(f"## {vid}\n")
                    f.write(f"https://www.youtube.com/watch?v={vid}\n\n")
                    f.write("Transcript:\n")
                    for entry in transcript:
                        ts = entry.get('start', 0)
                        minutes = int(ts // 60)
                        seconds = int(ts % 60)
                        text = entry.get('text', '')
                        f.write(f"- {minutes:02d}:{seconds:02d}: {text}\n")
                    f.write("\n---\n\n")
            logger.info("Transcripts successfully saved")
            QMessageBox.information(self, "Saved", f"Transcripts saved to {path}")
        except Exception as e:
            logger.exception(f"Failed to save transcripts: {e}")
            QMessageBox.critical(self, "Save Error", str(e))

if __name__ == '__main__':
    logger.info("Application starting")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())