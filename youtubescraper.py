# filepath: youtube-transcript-extractor/youtube-transcript-extractor/src/youtubescraper.py
import sys
import os
import logging
import json
import re
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import timedelta

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QComboBox, QLabel, QCheckBox,
    QGroupBox, QProgressBar, QTabWidget, QTextEdit, QSpinBox
)
from PySide6.QtCore import Qt, QRunnable, QThreadPool, Signal, QObject, QThread
from PySide6.QtGui import QFont

# Import dependencies with graceful fallback
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
except ImportError:
    print("youtube-transcript-api not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "youtube-transcript-api"])
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

# Audio processing for fallback methods
try:
    import yt_dlp
    import speech_recognition as sr
    from pydub import AudioSegment
    from pydub.utils import make_chunks
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Audio processing libraries not available (optional)")

# Whisper for local transcription
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

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
    detailed_result = Signal(str, dict)  # For enhanced results with confidence

class EnhancedTranscriptWorker(QRunnable):
    """Enhanced worker with multiple transcription methods"""
    
    def __init__(self, video_id, methods=None, use_fallback=True):
        super().__init__()
        self.video_id = video_id
        self.signals = WorkerSignals()
        self.methods = methods or ['youtube']
        self.use_fallback = use_fallback
        self.recognizer = sr.Recognizer() if AUDIO_AVAILABLE else None
        
    def run(self):
        logger.info(f"[{self.video_id}] Enhanced worker started with methods: {self.methods}")
        self.signals.progress.emit(self.video_id, "Fetching")
        
        transcript = None
        confidence = 1.0
        method_used = None
        
        # Try YouTube transcript first
        if 'youtube' in self.methods:
            transcript = self._get_youtube_transcript()
            if transcript:
                method_used = 'youtube'
                confidence = 0.95  # YouTube transcripts are generally reliable
        
        # Try audio extraction methods if needed
        if not transcript and self.use_fallback and AUDIO_AVAILABLE:
            audio_path = self._download_audio()
            if audio_path:
                # Try Whisper if available
                if 'whisper' in self.methods and WHISPER_AVAILABLE:
                    transcript, conf = self._transcribe_whisper(audio_path)
                    if transcript:
                        method_used = 'whisper'
                        confidence = conf
                
                # Try Google Speech Recognition
                if not transcript and 'google' in self.methods:
                    transcript, conf = self._transcribe_google(audio_path)
                    if transcript:
                        method_used = 'google'
                        confidence = conf
                
                # Cleanup audio file
                if os.path.exists(audio_path):
                    os.remove(audio_path)
        
        if transcript:
            logger.info(f"[{self.video_id}] Transcript fetched using {method_used} ({len(transcript)} entries, confidence: {confidence:.2f})")
            
            # Emit both simple and detailed results
            self.signals.result.emit(self.video_id, transcript)
            self.signals.detailed_result.emit(self.video_id, {
                'transcript': transcript,
                'method': method_used,
                'confidence': confidence
            })
            self.signals.progress.emit(self.video_id, f"Done ({method_used})")
        else:
            msg = "No transcript available with any method"
            logger.warning(f"[{self.video_id}] {msg}")
            self.signals.error.emit(self.video_id, msg)
            self.signals.progress.emit(self.video_id, "Failed")
    
    def _get_youtube_transcript(self) -> Optional[List[Dict]]:
        """Get YouTube transcript with multiple language fallbacks"""
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(self.video_id)
            
            # Try manually created English first
            try:
                transcript = transcript_list.find_manually_created_transcript(['en']).fetch()
                logger.info(f"[{self.video_id}] Found manually created English transcript")
                return transcript
            except:
                pass
            
            # Try generated English
            try:
                transcript = transcript_list.find_generated_transcript(['en']).fetch()
                logger.info(f"[{self.video_id}] Found generated English transcript")
                return transcript
            except:
                pass
            
            # Try any English variant
            for t in transcript_list:
                if t.language_code.startswith('en'):
                    transcript = t.fetch()
                    logger.info(f"[{self.video_id}] Found transcript in {t.language}")
                    return transcript
            
            # Get first available transcript
            for t in transcript_list:
                transcript = t.fetch()
                logger.info(f"[{self.video_id}] Using transcript in language: {t.language}")
                return transcript
                
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            logger.warning(f"[{self.video_id}] No YouTube transcript: {e}")
        except Exception as e:
            logger.error(f"[{self.video_id}] Error getting YouTube transcript: {e}")
        
        return None
    
    def _download_audio(self) -> Optional[str]:
        """Download audio from YouTube video"""
        if not AUDIO_AVAILABLE:
            return None
            
        try:
            logger.info(f"[{self.video_id}] Downloading audio...")
            output_path = Path("temp_audio")
            output_path.mkdir(exist_ok=True)
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(output_path / f'{self.video_id}.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
            }
            
            url = f"https://www.youtube.com/watch?v={self.video_id}"
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                wav_file = str(output_path / f"{self.video_id}.wav")
                
                if os.path.exists(wav_file):
                    logger.info(f"[{self.video_id}] Audio downloaded successfully")
                    return wav_file
                    
        except Exception as e:
            logger.error(f"[{self.video_id}] Error downloading audio: {e}")
        
        return None
    
    def _transcribe_whisper(self, audio_path: str) -> tuple[Optional[List[Dict]], float]:
        """Transcribe using Whisper"""
        if not WHISPER_AVAILABLE:
            return None, 0.0
            
        try:
            logger.info(f"[{self.video_id}] Transcribing with Whisper...")
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            
            transcript = []
            for segment in result['segments']:
                transcript.append({
                    'text': segment['text'].strip(),
                    'start': segment['start'],
                    'duration': segment['end'] - segment['start']
                })
            
            # Calculate average confidence
            avg_confidence = 0.85  # Whisper is generally reliable
            return transcript, avg_confidence
            
        except Exception as e:
            logger.error(f"[{self.video_id}] Whisper transcription error: {e}")
        
        return None, 0.0
    
    def _transcribe_google(self, audio_path: str) -> tuple[Optional[List[Dict]], float]:
        """Transcribe using Google Speech Recognition"""
        if not AUDIO_AVAILABLE:
            return None, 0.0
            
        try:
            logger.info(f"[{self.video_id}] Transcribing with Google Speech Recognition...")
            audio = AudioSegment.from_wav(audio_path)
            chunks = make_chunks(audio, 30000)  # 30-second chunks
            
            transcript = []
            total_confidence = 0
            
            for i, chunk in enumerate(chunks):
                chunk_path = f"temp_chunk_{self.video_id}_{i}.wav"
                chunk.export(chunk_path, format="wav")
                
                try:
                    with sr.AudioFile(chunk_path) as source:
                        audio_data = self.recognizer.record(source)
                        # Get detailed results with confidence
                        result = self.recognizer.recognize_google(
                            audio_data, 
                            show_all=True
                        )
                        
                        if result and 'alternative' in result:
                            best = result['alternative'][0]
                            text = best.get('transcript', '')
                            confidence = best.get('confidence', 0.5)
                            
                            transcript.append({
                                'text': text,
                                'start': i * 30,
                                'duration': len(chunk) / 1000.0
                            })
                            total_confidence += confidence
                            
                except sr.UnknownValueError:
                    logger.warning(f"[{self.video_id}] Could not understand chunk {i}")
                except Exception as e:
                    logger.error(f"[{self.video_id}] Error in chunk {i}: {e}")
                finally:
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
            
            avg_confidence = total_confidence / len(chunks) if chunks else 0.5
            return transcript if transcript else None, avg_confidence
            
        except Exception as e:
            logger.error(f"[{self.video_id}] Google transcription error: {e}")
        
        return None, 0.0

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube Transcript Extractor Pro")
        self.resize(900, 700)
        
        self.threadpool = QThreadPool()
        self.transcripts = {}
        self.detailed_results = {}
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Main extraction tab
        extract_tab = QWidget()
        self._setup_extract_tab(extract_tab)
        tabs.addTab(extract_tab, "Extract")
        
        # Settings tab
        settings_tab = QWidget()
        self._setup_settings_tab(settings_tab)
        tabs.addTab(settings_tab, "Settings")
        
        self.setCentralWidget(tabs)
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2d2d2d;
            }
            QWidget {
                background-color: #2d2d2d;
                color: #ffffff;
            }
            QPlainTextEdit, QTextEdit, QTableWidget {
                background-color: #3c3c3c;
                border: 1px solid #555;
            }
            QPushButton {
                background-color: #0e639c;
                border: none;
                padding: 8px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:disabled {
                background-color: #555;
            }
            QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555;
                padding: 5px;
            }
            QGroupBox {
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
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
            }
            QProgressBar::chunk {
                background-color: #0e639c;
            }
        """)
    
    def _setup_extract_tab(self, parent):
        """Setup main extraction interface"""
        layout = QVBoxLayout(parent)
        
        # Input area
        input_group = QGroupBox("YouTube Videos")
        input_layout = QVBoxLayout()
        
        self.input_edit = QPlainTextEdit()
        self.input_edit.setPlaceholderText("Paste YouTube links here, one per line...")
        self.input_edit.setMaximumHeight(150)
        input_layout.addWidget(self.input_edit)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.extract_button = QPushButton("🔍 Extract Transcripts")
        self.extract_button.clicked.connect(self.start_extraction)
        controls_layout.addWidget(self.extract_button)
        
        self.save_button = QPushButton("💾 Save as Markdown")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_markdown)
        controls_layout.addWidget(self.save_button)
        
        self.save_json_button = QPushButton("📄 Save as JSON")
        self.save_json_button.setEnabled(False)
        self.save_json_button.clicked.connect(self.save_json)
        controls_layout.addWidget(self.save_json_button)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Results table
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Video ID", "Status", "Method", "Confidence"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        results_layout.addWidget(self.table)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
    
    def _setup_settings_tab(self, parent):
        """Setup settings interface"""
        layout = QVBoxLayout(parent)
        
        # Method selection
        method_group = QGroupBox("Transcription Methods")
        method_layout = QVBoxLayout()
        
        self.use_youtube = QCheckBox("YouTube Closed Captions")
        self.use_youtube.setChecked(True)
        method_layout.addWidget(self.use_youtube)
        
        self.use_whisper = QCheckBox("Whisper (Local AI)")
        self.use_whisper.setEnabled(WHISPER_AVAILABLE)
        if not WHISPER_AVAILABLE:
            self.use_whisper.setText("Whisper (Not Available - Install: pip install openai-whisper)")
        method_layout.addWidget(self.use_whisper)
        
        self.use_google = QCheckBox("Google Speech Recognition")
        self.use_google.setEnabled(AUDIO_AVAILABLE)
        if not AUDIO_AVAILABLE:
            self.use_google.setText("Google Speech (Not Available - Install: pip install SpeechRecognition pydub yt-dlp)")
        method_layout.addWidget(self.use_google)
        
        self.use_fallback = QCheckBox("Use fallback methods if YouTube transcript unavailable")
        self.use_fallback.setChecked(True)
        method_layout.addWidget(self.use_fallback)
        
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)
        
        # Output options
        output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout()
        
        self.include_timestamps = QCheckBox("Include timestamps")
        self.include_timestamps.setChecked(True)
        output_layout.addWidget(self.include_timestamps)
        
        self.include_confidence = QCheckBox("Include confidence scores")
        self.include_confidence.setChecked(False)
        output_layout.addWidget(self.include_confidence)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        layout.addStretch()
    
    def start_extraction(self):
        raw_text = self.input_edit.toPlainText().strip()
        if not raw_text:
            QMessageBox.warning(self, "Input Error", "Please paste at least one YouTube link.")
            return
        
        # Extract video IDs
        video_ids = self._extract_video_ids(raw_text)
        
        if not video_ids:
            QMessageBox.warning(self, "Input Error", "No valid YouTube URLs found.")
            return
        
        logger.info(f"Starting extraction for {len(video_ids)} video(s)")
        
        # Clear previous results
        self.table.setRowCount(0)
        self.transcripts.clear()
        self.detailed_results.clear()
        self.save_button.setEnabled(False)
        self.save_json_button.setEnabled(False)
        
        # Setup progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(video_ids))
        self.progress_bar.setValue(0)
        
        # Get selected methods
        methods = []
        if self.use_youtube.isChecked():
            methods.append('youtube')
        if self.use_whisper.isChecked() and WHISPER_AVAILABLE:
            methods.append('whisper')
        if self.use_google.isChecked() and AUDIO_AVAILABLE:
            methods.append('google')
        
        if not methods:
            methods = ['youtube']  # Default to YouTube
        
        # Start workers
        for vid in video_ids:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(vid))
            self.table.setItem(row, 1, QTableWidgetItem("⏳ Queued"))
            self.table.setItem(row, 2, QTableWidgetItem("-"))
            self.table.setItem(row, 3, QTableWidgetItem("-"))
            
            worker = EnhancedTranscriptWorker(
                vid, 
                methods=methods,
                use_fallback=self.use_fallback.isChecked()
            )
            worker.signals.progress.connect(self.update_status)
            worker.signals.result.connect(self.store_transcript)
            worker.signals.detailed_result.connect(self.store_detailed_result)
            worker.signals.error.connect(self.handle_error)
            self.threadpool.start(worker)
    
    def _extract_video_ids(self, text: str) -> List[str]:
        """Extract video IDs from various YouTube URL formats"""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        video_ids = []
        
        for link in lines:
            # Support various YouTube URL formats
            if "youtube.com" in link or "youtu.be" in link or "m.youtube.com" in link:
                # Extract video ID from various URL formats
                if "v=" in link:
                    vid = link.split("v=")[1].split("&")[0].split("#")[0]
                elif "youtu.be/" in link:
                    vid = link.split("youtu.be/")[1].split("?")[0].split("#")[0]
                elif "/embed/" in link:
                    vid = link.split("/embed/")[1].split("?")[0].split("#")[0]
                elif "/v/" in link:
                    vid = link.split("/v/")[1].split("?")[0].split("#")[0]
                else:
                    continue
                
                # Validate video ID format (11 characters)
                if vid and len(vid) == 11:
                    video_ids.append(vid)
            elif len(link) == 11 and link.replace("-", "").replace("_", "").isalnum():
                # Direct video ID
                video_ids.append(link)
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for vid in video_ids:
            if vid not in seen:
                seen.add(vid)
                unique.append(vid)
        
        return unique
    
    def update_status(self, video_id, status):
        """Update status in table"""
        emoji_status = {
            "Queued": "⏳ Queued",
            "Fetching": "🔄 Fetching",
            "Done": "✅ Done",
            "Failed": "❌ Failed"
        }
        
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == video_id:
                # Update status
                if "(" in status:  # Contains method info
                    base_status = status.split("(")[0].strip()
                    method = status.split("(")[1].rstrip(")")
                    self.table.setItem(row, 1, QTableWidgetItem(emoji_status.get(base_status, status)))
                    self.table.setItem(row, 2, QTableWidgetItem(method))
                else:
                    self.table.setItem(row, 1, QTableWidgetItem(emoji_status.get(status, status)))
                break
        
        # Update progress bar
        completed = sum(
            1 for r in range(self.table.rowCount())
            if self.table.item(r, 1).text() in ("✅ Done", "❌ Failed")
        )
        self.progress_bar.setValue(completed)
        
        # Check if all done
        if completed == self.table.rowCount():
            self.progress_bar.setVisible(False)
            self.save_button.setEnabled(bool(self.transcripts))
            self.save_json_button.setEnabled(bool(self.transcripts))
            logger.info("All tasks completed")
    
    def store_transcript(self, video_id, transcript):
        """Store simple transcript"""
        self.transcripts[video_id] = transcript
        logger.info(f"Transcript stored for {video_id} ({len(transcript)} entries)")
    
    def store_detailed_result(self, video_id, result):
        """Store detailed result with metadata"""
        self.detailed_results[video_id] = result
        
        # Update confidence in table
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == video_id:
                confidence = result.get('confidence', 0)
                self.table.setItem(row, 3, QTableWidgetItem(f"{confidence:.1%}"))
                break
    
    def handle_error(self, video_id, error_msg):
        """Handle extraction error"""
        logger.error(f"Error for {video_id}: {error_msg}")
        # Error status is already updated via progress signal
    
    def save_markdown(self):
        """Save transcripts as Markdown"""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Markdown", os.path.expanduser("~"), "Markdown Files (*.md)"
        )
        if not path:
            return
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write("# YouTube Transcripts\n\n")
                
                for vid, transcript in self.transcripts.items():
                    f.write(f"## Video: {vid}\n")
                    f.write(f"🔗 https://www.youtube.com/watch?v={vid}\n\n")
                    
                    # Add metadata if available
                    if vid in self.detailed_results:
                        details = self.detailed_results[vid]
                        f.write(f"**Method:** {details.get('method', 'Unknown')}\n")
                        if self.include_confidence.isChecked():
                            f.write(f"**Confidence:** {details.get('confidence', 0):.1%}\n")
                        f.write("\n")
                    
                    f.write("### Transcript\n\n")
                    
                    for entry in transcript:
                        if self.include_timestamps.isChecked():
                            ts = entry.get('start', 0)
                            minutes = int(ts // 60)
                            seconds = int(ts % 60)
                            text = entry.get('text', '')
                            f.write(f"**[{minutes:02d}:{seconds:02d}]** {text}\n\n")
                        else:
                            f.write(f"{entry.get('text', '')}\n\n")
                    
                    f.write("\n---\n\n")
            
            logger.info(f"Transcripts saved to {path}")
            QMessageBox.information(self, "Saved", f"Transcripts saved to {path}")
            
        except Exception as e:
            logger.exception(f"Failed to save transcripts: {e}")
            QMessageBox.critical(self, "Save Error", str(e))
    
    def save_json(self):
        """Save transcripts as JSON with full metadata"""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save JSON", os.path.expanduser("~"), "JSON Files (*.json)"
        )
        if not path:
            return
        
        try:
            output = {}
            for vid in self.transcripts:
                output[vid] = {
                    'url': f"https://www.youtube.com/watch?v={vid}",
                    'transcript': self.transcripts[vid]
                }
                
                if vid in self.detailed_results:
                    output[vid].update(self.detailed_results[vid])
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Transcripts saved to {path}")
            QMessageBox.information(self, "Saved", f"Transcripts saved to {path}")
            
        except Exception as e:
            logger.exception(f"Failed to save JSON: {e}")
            QMessageBox.critical(self, "Save Error", str(e))

if __name__ == '__main__':
    logger.info("YouTube Transcript Extractor Pro starting")
    app = QApplication(sys.argv)
    app.setApplicationName("YouTube Transcript Extractor Pro")
    app.setOrganizationName("KingAI")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())