#!/usr/bin/env python3
"""
YouTube Transcript Extractor - FIXED VERSION
=============================================
Updated for youtube-transcript-api v1.2.3+ (new API)

Author: KingAI
Date: November 26, 2025
"""

import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Optional

# Core dependencies with auto-install
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTextEdit, QPushButton, QTableWidget, QTableWidgetItem,
        QFileDialog, QMessageBox, QLabel, QProgressBar,
        QGroupBox, QTabWidget, QCheckBox, QComboBox
    )
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtGui import QFont
except ImportError:
    print("Installing PySide6...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PySide6"])
    from PySide6.QtWidgets import *
    from PySide6.QtCore import *
    from PySide6.QtGui import *

# YouTube transcript API with auto-install
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
except ImportError:
    print("Installing youtube-transcript-api...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "youtube-transcript-api"])
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound


class TranscriptWorker(QThread):
    """Worker thread for transcript extraction"""
    progress = Signal(str, str)  # video_id, status
    result = Signal(str, list)    # video_id, transcript
    error = Signal(str, str)      # video_id, error_message
    
    def __init__(self, video_id: str):
        super().__init__()
        self.video_id = video_id
    
    def run(self):
        """Extract transcript"""
        try:
            self.progress.emit(self.video_id, "Fetching")
            
            # NEW API v1.2.3+: Create instance and fetch
            api = YouTubeTranscriptApi()
            fetched = api.fetch(self.video_id)
            
            # Convert snippets to dict format
            transcript = [
                {
                    'text': snippet.text,
                    'start': snippet.start,
                    'duration': snippet.duration
                }
                for snippet in fetched.snippets
            ]
            
            self.result.emit(self.video_id, transcript)
            self.progress.emit(self.video_id, "Done")
            
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            error_msg = f"No transcript available: {str(e)}"
            self.error.emit(self.video_id, error_msg)
            self.progress.emit(self.video_id, "Failed")
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.error.emit(self.video_id, error_msg)
            self.progress.emit(self.video_id, "Failed")


class YouTubeTranscriptExtractor(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube Transcript Extractor - KingAI Edition")
        self.resize(1000, 700)
        self.transcripts = {}
        self.workers = []
        
        self.init_ui()
        self.apply_style()
    
    def init_ui(self):
        """Initialize UI"""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Input group
        input_group = QGroupBox("YouTube URLs")
        input_layout = QVBoxLayout()
        
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText(
            "Paste YouTube URLs here (one per line)...\n\n"
            "Supports formats:\n"
            "  • https://www.youtube.com/watch?v=VIDEO_ID\n"
            "  • https://youtu.be/VIDEO_ID\n"
            "  • Just the VIDEO_ID"
        )
        self.input_text.setMaximumHeight(150)
        input_layout.addWidget(self.input_text)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        self.extract_btn = QPushButton("🔍 Extract Transcripts")
        self.extract_btn.clicked.connect(self.extract_transcripts)
        btn_layout.addWidget(self.extract_btn)
        
        self.save_md_btn = QPushButton("💾 Save as Markdown")
        self.save_md_btn.setEnabled(False)
        self.save_md_btn.clicked.connect(self.save_markdown)
        btn_layout.addWidget(self.save_md_btn)
        
        self.save_json_btn = QPushButton("📄 Save as JSON")
        self.save_json_btn.setEnabled(False)
        self.save_json_btn.clicked.connect(self.save_json)
        btn_layout.addWidget(self.save_json_btn)
        
        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.clicked.connect(self.clear_all)
        btn_layout.addWidget(self.clear_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Results table
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Video ID", "Status", "Segments"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.itemDoubleClicked.connect(self.show_transcript)
        results_layout.addWidget(self.table)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Status bar
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
    
    def apply_style(self):
        """Apply dark theme"""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #2d2d2d;
                color: #ffffff;
            }
            QTextEdit, QTableWidget {
                background-color: #3c3c3c;
                border: 1px solid #555;
                color: #ffffff;
            }
            QPushButton {
                background-color: #0e639c;
                border: none;
                padding: 8px;
                border-radius: 3px;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:disabled {
                background-color: #555;
            }
            QGroupBox {
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QTableWidget {
                gridline-color: #555;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #0e639c;
            }
        """)
    
    def extract_video_id(self, text: str) -> Optional[str]:
        """Extract video ID from URL or return if already an ID"""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'^([0-9A-Za-z_-]{11})$',  # Direct ID
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.strip())
            if match:
                return match.group(1)
        return None
    
    def extract_transcripts(self):
        """Start extraction process"""
        raw_text = self.input_text.toPlainText().strip()
        if not raw_text:
            QMessageBox.warning(self, "No Input", "Please paste YouTube URLs first!")
            return
        
        # Extract video IDs
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        video_ids = []
        for line in lines:
            vid = self.extract_video_id(line)
            if vid:
                video_ids.append(vid)
        
        if not video_ids:
            QMessageBox.warning(self, "Invalid URLs", "No valid YouTube URLs found!")
            return
        
        # Remove duplicates
        video_ids = list(dict.fromkeys(video_ids))
        
        # Clear previous results
        self.table.setRowCount(0)
        self.transcripts.clear()
        self.workers.clear()
        
        # Setup progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(video_ids))
        self.progress_bar.setValue(0)
        
        # Disable buttons
        self.extract_btn.setEnabled(False)
        self.save_md_btn.setEnabled(False)
        self.save_json_btn.setEnabled(False)
        
        self.status_label.setText(f"Extracting {len(video_ids)} video(s)...")
        
        # Start workers
        for vid in video_ids:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(vid))
            self.table.setItem(row, 1, QTableWidgetItem("⏳ Queued"))
            self.table.setItem(row, 2, QTableWidgetItem(""))
            
            worker = TranscriptWorker(vid)
            worker.progress.connect(self.update_progress)
            worker.result.connect(self.store_transcript)
            worker.error.connect(self.handle_error)
            worker.finished.connect(lambda: self.check_completion())
            
            self.workers.append(worker)
            worker.start()
    
    def update_progress(self, video_id: str, status: str):
        """Update progress in table"""
        status_emoji = {
            "Queued": "⏳ Queued",
            "Fetching": "🔄 Fetching",
            "Done": "✅ Done",
            "Failed": "❌ Failed"
        }
        
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == video_id:
                self.table.setItem(row, 1, QTableWidgetItem(status_emoji.get(status, status)))
                break
    
    def store_transcript(self, video_id: str, transcript: List[Dict]):
        """Store extracted transcript"""
        self.transcripts[video_id] = transcript
        
        # Update segment count
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == video_id:
                self.table.setItem(row, 2, QTableWidgetItem(str(len(transcript))))
                break
    
    def handle_error(self, video_id: str, error_msg: str):
        """Handle extraction error"""
        print(f"Error for {video_id}: {error_msg}")
    
    def check_completion(self):
        """Check if all extractions are complete"""
        completed = sum(
            1 for row in range(self.table.rowCount())
            if self.table.item(row, 1).text() in ("✅ Done", "❌ Failed")
        )
        
        self.progress_bar.setValue(completed)
        
        if completed == self.table.rowCount():
            success_count = sum(
                1 for row in range(self.table.rowCount())
                if self.table.item(row, 1).text() == "✅ Done"
            )
            
            self.status_label.setText(
                f"Complete: {success_count}/{self.table.rowCount()} successful"
            )
            self.progress_bar.setVisible(False)
            self.extract_btn.setEnabled(True)
            
            if self.transcripts:
                self.save_md_btn.setEnabled(True)
                self.save_json_btn.setEnabled(True)
    
    def show_transcript(self, item):
        """Show transcript when double-clicked"""
        row = item.row()
        video_id = self.table.item(row, 0).text()
        
        if video_id not in self.transcripts:
            return
        
        transcript = self.transcripts[video_id]
        
        # Create dialog
        dialog = QMessageBox(self)
        dialog.setWindowTitle(f"Transcript: {video_id}")
        dialog.setIcon(QMessageBox.Information)
        
        # Format transcript
        text = f"Video ID: {video_id}\n"
        text += f"Segments: {len(transcript)}\n"
        text += "=" * 60 + "\n\n"
        
        for t in transcript[:50]:  # Show first 50
            mins = int(t['start'] // 60)
            secs = int(t['start'] % 60)
            text += f"[{mins:02d}:{secs:02d}] {t['text']}\n"
        
        if len(transcript) > 50:
            text += f"\n... and {len(transcript) - 50} more segments"
        
        dialog.setText(text)
        dialog.exec()
    
    def save_markdown(self):
        """Save transcripts as Markdown"""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Markdown", "", "Markdown Files (*.md)"
        )
        if not path:
            return
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write("# YouTube Transcript Export\n\n")
                f.write(f"**Extracted**: {len(self.transcripts)} video(s)\n\n")
                f.write("---\n\n")
                
                for video_id, transcript in self.transcripts.items():
                    f.write(f"## Video: {video_id}\n\n")
                    f.write(f"**URL**: https://www.youtube.com/watch?v={video_id}\n")
                    f.write(f"**Segments**: {len(transcript)}\n\n")
                    
                    for t in transcript:
                        mins = int(t['start'] // 60)
                        secs = int(t['start'] % 60)
                        f.write(f"[{mins:02d}:{secs:02d}] {t['text']}\n")
                    
                    f.write("\n---\n\n")
            
            QMessageBox.information(self, "Success", f"Saved to: {path}")
            self.status_label.setText(f"Saved to: {Path(path).name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")
    
    def save_json(self):
        """Save transcripts as JSON"""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save JSON", "", "JSON Files (*.json)"
        )
        if not path:
            return
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.transcripts, f, indent=2, ensure_ascii=False)
            
            QMessageBox.information(self, "Success", f"Saved to: {path}")
            self.status_label.setText(f"Saved to: {Path(path).name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")
    
    def clear_all(self):
        """Clear all data"""
        self.input_text.clear()
        self.table.setRowCount(0)
        self.transcripts.clear()
        self.workers.clear()
        self.save_md_btn.setEnabled(False)
        self.save_json_btn.setEnabled(False)
        self.status_label.setText("Ready")


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("YouTube Transcript Extractor")
    app.setOrganizationName("KingAI")
    
    window = YouTubeTranscriptExtractor()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
