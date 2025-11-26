#!/usr/bin/env python3
"""
YouTube Transcript Extractor with Multi-Method Speech Recognition
Enhanced with confidence scoring, word-level timestamps, and multiple fallbacks
"""

import sys
import os
import json
import re
from typing import Optional, List, Dict, Tuple, Any
from datetime import timedelta
import threading
from pathlib import Path
import tempfile
import hashlib
import time
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Core dependencies
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
        QWidget, QPushButton, QLineEdit, QTextEdit, QComboBox,
        QLabel, QProgressBar, QFileDialog, QMessageBox, QGroupBox,
        QCheckBox, QSpinBox, QSlider, QTableWidget, QTableWidgetItem,
        QTabWidget, QDoubleSpinBox
    )
    from PySide6.QtCore import Qt, QThread, Signal, QTimer
    from PySide6.QtGui import QFont, QTextCursor, QColor
except ImportError:
    print("Installing PySide6...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PySide6"])
    from PySide6.QtWidgets import *
    from PySide6.QtCore import *
    from PySide6.QtGui import *

# YouTube dependencies
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
except ImportError:
    print("Installing youtube-transcript-api...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "youtube-transcript-api"])
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

# Audio processing dependencies
try:
    import yt_dlp
    import speech_recognition as sr
    from pydub import AudioSegment
    from pydub.utils import make_chunks
    from pydub.silence import detect_nonsilent
except ImportError:
    print("Installing audio processing dependencies...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "yt-dlp", "SpeechRecognition", "pydub"])
    import yt_dlp
    import speech_recognition as sr
    from pydub import AudioSegment
    from pydub.utils import make_chunks
    from pydub.silence import detect_nonsilent

# Additional speech recognition engines
try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("Azure Speech SDK not available (optional)")

try:
    from ibm_watson import SpeechToTextV1
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False
    print("IBM Watson SDK not available (optional)")

try:
    from google.cloud import speech_v1p1beta1 as speech
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    print("Google Cloud Speech not available (optional)")

try:
    import assemblyai as aai
    ASSEMBLYAI_AVAILABLE = True
except ImportError:
    ASSEMBLYAI_AVAILABLE = False
    print("AssemblyAI SDK not available (optional)")

try:
    from deepgram import Deepgram
    DEEPGRAM_AVAILABLE = True
except ImportError:
    DEEPGRAM_AVAILABLE = False
    print("Deepgram SDK not available (optional)")

try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    print("Vosk not available (optional)")

# NLP and text processing
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    import language_tool_python
except ImportError:
    print("Installing NLP dependencies...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "nltk", "textstat", "language-tool-python"])
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    import language_tool_python
    # Download required NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

@dataclass
class WordSegment:
    """Word-level transcript segment with confidence"""
    word: str
    start: float
    end: float
    confidence: float
    speaker: Optional[str] = None
    
@dataclass
class TranscriptSegment:
    """Enhanced transcript segment with metadata"""
    text: str
    start: float
    end: float
    confidence: float
    words: List[WordSegment]
    speaker: Optional[str] = None
    method: str = "unknown"
    language: str = "en"

class MultiMethodTranscriber:
    """Multi-method speech recognition with fallback"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.methods_available = self._check_available_methods()
        
    def _check_available_methods(self) -> Dict[str, bool]:
        """Check which speech recognition methods are available"""
        return {
            'google': True,  # Always available (free tier)
            'google_cloud': GOOGLE_CLOUD_AVAILABLE,
            'azure': AZURE_AVAILABLE,
            'ibm': IBM_AVAILABLE,
            'assemblyai': ASSEMBLYAI_AVAILABLE,
            'deepgram': DEEPGRAM_AVAILABLE,
            'whisper': True,  # Can be installed on demand
            'vosk': VOSK_AVAILABLE,
            'sphinx': True,  # Offline, always available
        }
    
    def transcribe_with_method(self, audio_path: str, method: str, 
                              language: str = "en-US", 
                              api_keys: Dict[str, str] = None) -> List[TranscriptSegment]:
        """Transcribe audio using specified method"""
        
        if method == 'google':
            return self._transcribe_google(audio_path, language)
        elif method == 'google_cloud' and self.methods_available['google_cloud']:
            return self._transcribe_google_cloud(audio_path, language, api_keys)
        elif method == 'azure' and self.methods_available['azure']:
            return self._transcribe_azure(audio_path, language, api_keys)
        elif method == 'ibm' and self.methods_available['ibm']:
            return self._transcribe_ibm(audio_path, language, api_keys)
        elif method == 'assemblyai' and self.methods_available['assemblyai']:
            return self._transcribe_assemblyai(audio_path, language, api_keys)
        elif method == 'deepgram' and self.methods_available['deepgram']:
            return self._transcribe_deepgram(audio_path, language, api_keys)
        elif method == 'whisper':
            return self._transcribe_whisper(audio_path, language)
        elif method == 'vosk' and self.methods_available['vosk']:
            return self._transcribe_vosk(audio_path, language)
        elif method == 'sphinx':
            return self._transcribe_sphinx(audio_path, language)
        else:
            raise ValueError(f"Method {method} not available")
    
    def _transcribe_google(self, audio_path: str, language: str) -> List[TranscriptSegment]:
        """Google Speech Recognition (free tier)"""
        segments = []
        audio = AudioSegment.from_wav(audio_path)
        
        # Smart chunking based on silence detection
        chunks = self._smart_chunk_audio(audio)
        
        for i, (chunk, start_time) in enumerate(chunks):
            chunk_path = f"temp_chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            
            try:
                with sr.AudioFile(chunk_path) as source:
                    audio_data = self.recognizer.record(source)
                    # Get detailed results with confidence
                    result = self.recognizer.recognize_google(
                        audio_data, 
                        language=language,
                        show_all=True
                    )
                    
                    if result and 'alternative' in result:
                        best = result['alternative'][0]
                        text = best.get('transcript', '')
                        confidence = best.get('confidence', 0.5)
                        
                        # Estimate word-level segments
                        words = self._estimate_word_timestamps(
                            text, start_time, len(chunk) / 1000.0, confidence
                        )
                        
                        segment = TranscriptSegment(
                            text=text,
                            start=start_time,
                            end=start_time + len(chunk) / 1000.0,
                            confidence=confidence,
                            words=words,
                            method='google'
                        )
                        segments.append(segment)
                        
            except sr.UnknownValueError:
                # Add empty segment with low confidence
                segment = TranscriptSegment(
                    text="[inaudible]",
                    start=start_time,
                    end=start_time + len(chunk) / 1000.0,
                    confidence=0.0,
                    words=[],
                    method='google'
                )
                segments.append(segment)
            except Exception as e:
                print(f"Error in chunk {i}: {e}")
            finally:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
        
        return segments
    
    def _transcribe_whisper(self, audio_path: str, language: str) -> List[TranscriptSegment]:
        """OpenAI Whisper (local model)"""
        try:
            import whisper
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper"])
            import whisper
        
        segments = []
        model = whisper.load_model("base")
        result = model.transcribe(
            audio_path, 
            language=language[:2],
            word_timestamps=True,
            verbose=False
        )
        
        for seg in result['segments']:
            words = []
            if 'words' in seg:
                for word_data in seg['words']:
                    word = WordSegment(
                        word=word_data['word'].strip(),
                        start=word_data['start'],
                        end=word_data['end'],
                        confidence=word_data.get('probability', 0.9)
                    )
                    words.append(word)
            
            segment = TranscriptSegment(
                text=seg['text'].strip(),
                start=seg['start'],
                end=seg['end'],
                confidence=seg.get('avg_logprob', -0.5) + 1.5,  # Convert to 0-1 range
                words=words,
                method='whisper'
            )
            segments.append(segment)
        
        return segments
    
    def _transcribe_vosk(self, audio_path: str, language: str) -> List[TranscriptSegment]:
        """Vosk offline speech recognition"""
        if not VOSK_AVAILABLE:
            raise ImportError("Vosk not available")
        
        import vosk
        import wave
        
        # Download model if needed
        model_path = self._get_vosk_model(language)
        model = vosk.Model(model_path)
        
        segments = []
        wf = wave.open(audio_path, 'rb')
        rec = vosk.KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)  # Enable word-level timestamps
        
        current_time = 0.0
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if 'text' in result and result['text']:
                    words = []
                    if 'result' in result:
                        for word_data in result['result']:
                            word = WordSegment(
                                word=word_data['word'],
                                start=word_data['start'],
                                end=word_data['end'],
                                confidence=word_data.get('conf', 0.5)
                            )
                            words.append(word)
                    
                    segment = TranscriptSegment(
                        text=result['text'],
                        start=words[0].start if words else current_time,
                        end=words[-1].end if words else current_time + 1,
                        confidence=sum(w.confidence for w in words) / len(words) if words else 0.5,
                        words=words,
                        method='vosk'
                    )
                    segments.append(segment)
                    
                    if words:
                        current_time = words[-1].end
        
        # Final result
        final_result = json.loads(rec.FinalResult())
        if 'text' in final_result and final_result['text']:
            # Process final segment
            pass
        
        wf.close()
        return segments
    
    def _transcribe_sphinx(self, audio_path: str, language: str) -> List[TranscriptSegment]:
        """PocketSphinx offline recognition (fallback)"""
        segments = []
        
        with sr.AudioFile(audio_path) as source:
            audio_data = self.recognizer.record(source)
            
        try:
            text = self.recognizer.recognize_sphinx(audio_data)
            # Sphinx doesn't provide confidence or word timestamps
            # Estimate based on audio length
            audio = AudioSegment.from_wav(audio_path)
            duration = len(audio) / 1000.0
            
            words = self._estimate_word_timestamps(text, 0, duration, 0.3)
            
            segment = TranscriptSegment(
                text=text,
                start=0,
                end=duration,
                confidence=0.3,  # Low confidence for Sphinx
                words=words,
                method='sphinx'
            )
            segments.append(segment)
            
        except sr.UnknownValueError:
            pass
        except Exception as e:
            print(f"Sphinx error: {e}")
        
        return segments
    
    def _smart_chunk_audio(self, audio: AudioSegment, 
                          min_silence_len: int = 500,
                          silence_thresh: int = -40) -> List[Tuple[AudioSegment, float]]:
        """Smart audio chunking based on silence detection"""
        
        # Detect non-silent chunks
        nonsilent_chunks = detect_nonsilent(
            audio, 
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        
        chunks = []
        for start_ms, end_ms in nonsilent_chunks:
            # Add small padding
            start_ms = max(0, start_ms - 100)
            end_ms = min(len(audio), end_ms + 100)
            
            chunk = audio[start_ms:end_ms]
            start_time = start_ms / 1000.0
            
            # If chunk is too long, split it further
            if len(chunk) > 30000:  # 30 seconds
                sub_chunks = make_chunks(chunk, 30000)
                for i, sub_chunk in enumerate(sub_chunks):
                    sub_start = start_time + (i * 30)
                    chunks.append((sub_chunk, sub_start))
            else:
                chunks.append((chunk, start_time))
        
        return chunks
    
    def _estimate_word_timestamps(self, text: str, start: float, 
                                 duration: float, confidence: float) -> List[WordSegment]:
        """Estimate word-level timestamps when not provided"""
        words = text.split()
        if not words:
            return []
        
        # Estimate based on word length
        total_chars = sum(len(w) for w in words)
        time_per_char = duration / total_chars if total_chars > 0 else 0
        
        word_segments = []
        current_time = start
        
        for word in words:
            word_duration = len(word) * time_per_char
            word_segment = WordSegment(
                word=word,
                start=current_time,
                end=current_time + word_duration,
                confidence=confidence
            )
            word_segments.append(word_segment)
            current_time += word_duration + 0.1  # Small gap between words
        
        return word_segments
    
    def _get_vosk_model(self, language: str) -> str:
        """Download and return path to Vosk model"""
        # Implementation would download appropriate model
        # For now, return a placeholder
        return "vosk-model-en-us-0.22"
    
    # ...existing code...
    # Additional methods for Azure, IBM, AssemblyAI, Deepgram, Google Cloud
    # would be implemented similarly with their respective SDKs

class TranscriptAnalyzer:
    """Analyze transcript quality and statistics"""
    
    def __init__(self):
        self.grammar_tool = language_tool_python.LanguageTool('en-US')
    
    def analyze_transcript(self, segments: List[TranscriptSegment]) -> Dict[str, Any]:
        """Comprehensive transcript analysis"""
        
        full_text = " ".join(seg.text for seg in segments)
        all_words = []
        for seg in segments:
            all_words.extend(seg.words)
        
        # Calculate statistics
        stats = {
            'total_duration': max(seg.end for seg in segments) if segments else 0,
            'segment_count': len(segments),
            'word_count': len(all_words),
            'average_confidence': sum(seg.confidence for seg in segments) / len(segments) if segments else 0,
            'min_confidence': min(seg.confidence for seg in segments) if segments else 0,
            'max_confidence': max(seg.confidence for seg in segments) if segments else 0,
            'words_per_minute': (len(all_words) / (stats['total_duration'] / 60)) if stats['total_duration'] > 0 else 0,
            'readability_score': flesch_reading_ease(full_text) if full_text else 0,
            'grade_level': flesch_kincaid_grade(full_text) if full_text else 0,
            'methods_used': list(set(seg.method for seg in segments)),
        }
        
        # Confidence distribution
        confidence_bins = {
            'high': sum(1 for seg in segments if seg.confidence >= 0.8),
            'medium': sum(1 for seg in segments if 0.5 <= seg.confidence < 0.8),
            'low': sum(1 for seg in segments if seg.confidence < 0.5),
        }
        stats['confidence_distribution'] = confidence_bins
        
        # Grammar check
        grammar_errors = self.grammar_tool.check(full_text)
        stats['grammar_errors'] = len(grammar_errors)
        
        # Word frequency
        word_freq = {}
        for word_seg in all_words:
            word = word_seg.word.lower()
            word_freq[word] = word_freq.get(word, 0) + 1
        
        stats['most_common_words'] = sorted(
            word_freq.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return stats

class TranscriptWorker(QThread):
    """Enhanced worker thread for multi-method transcript extraction"""
    progress = Signal(int)
    status = Signal(str)
    result = Signal(dict)
    error = Signal(str)
    partial_result = Signal(dict)  # For streaming results
    
    def __init__(self):
        super().__init__()
        self.url = ""
        self.methods = ["auto"]  # Can be multiple methods
        self.language = "en-US"
        self.use_multi_method = False
        self.confidence_threshold = 0.5
        self.chunk_size = 30
        self.api_keys = {}
        self.transcriber = MultiMethodTranscriber()
        self.analyzer = TranscriptAnalyzer()
        
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL"""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:watch\?v=)([0-9A-Za-z_-]{11})',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def get_youtube_transcript(self, video_id: str) -> Optional[List[Dict]]:
        """Get transcript from YouTube's closed captions"""
        try:
            self.status.emit("Fetching YouTube transcript...")
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to get transcript in preferred language
            try:
                transcript = transcript_list.find_transcript([self.language])
            except:
                # Fall back to any available transcript
                transcript = transcript_list.find_generated_transcript([self.language])
            
            return transcript.fetch()
            
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            self.status.emit(f"No YouTube transcript available: {str(e)}")
            return None
        except Exception as e:
            self.status.emit(f"Error fetching transcript: {str(e)}")
            return None
    
    def download_audio(self, url: str) -> Optional[str]:
        """Download audio from YouTube video"""
        try:
            self.status.emit("Downloading audio from YouTube...")
            
            output_path = Path("temp_audio")
            output_path.mkdir(exist_ok=True)
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(output_path / '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
                'progress_hooks': [self._download_progress_hook],
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                wav_file = filename.rsplit('.', 1)[0] + '.wav'
                
                if os.path.exists(wav_file):
                    return wav_file
                    
            return None
            
        except Exception as e:
            self.error.emit(f"Error downloading audio: {str(e)}")
            return None
    
    def _download_progress_hook(self, d):
        """Progress hook for yt-dlp"""
        if d['status'] == 'downloading':
            if 'total_bytes' in d:
                percent = int(d['downloaded_bytes'] * 100 / d['total_bytes'])
                self.progress.emit(percent // 2)  # First 50% for download
            elif '_percent_str' in d:
                percent = float(d['_percent_str'].replace('%', ''))
                self.progress.emit(int(percent // 2))
    
    def transcribe_audio_google(self, audio_path: str) -> List[Dict]:
        """Transcribe audio using Google Speech Recognition"""
        try:
            self.status.emit("Transcribing audio with Google Speech Recognition...")
            recognizer = sr.Recognizer()
            
            # Load audio file
            audio = AudioSegment.from_wav(audio_path)
            chunks = make_chunks(audio, self.chunk_size * 1000)  # Convert to milliseconds
            
            transcript = []
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks):
                self.status.emit(f"Processing chunk {i+1}/{total_chunks}...")
                self.progress.emit(50 + int((i / total_chunks) * 50))
                
                # Export chunk to temp file
                chunk_path = f"temp_chunk_{i}.wav"
                chunk.export(chunk_path, format="wav")
                
                try:
                    with sr.AudioFile(chunk_path) as source:
                        audio_data = recognizer.record(source)
                        text = recognizer.recognize_google(audio_data, language=self.language)
                        
                        transcript.append({
                            'text': text,
                            'start': i * self.chunk_size,
                            'duration': len(chunk) / 1000.0
                        })
                except sr.UnknownValueError:
                    self.status.emit(f"Could not understand chunk {i+1}")
                except sr.RequestError as e:
                    self.status.emit(f"API error for chunk {i+1}: {e}")
                finally:
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
            
            return transcript
            
        except Exception as e:
            self.error.emit(f"Transcription error: {str(e)}")
            return []
    
    def transcribe_audio_whisper(self, audio_path: str) -> List[Dict]:
        """Transcribe audio using OpenAI Whisper (local)"""
        try:
            import whisper
        except ImportError:
            self.status.emit("Installing Whisper...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper"])
            import whisper
        
        try:
            self.status.emit("Loading Whisper model (this may take a moment)...")
            model = whisper.load_model("base")
            
            self.status.emit("Transcribing with Whisper...")
            result = model.transcribe(audio_path, language=self.language)
            
            # Convert Whisper segments to our format
            transcript = []
            for segment in result['segments']:
                transcript.append({
                    'text': segment['text'].strip(),
                    'start': segment['start'],
                    'duration': segment['end'] - segment['start']
                })
            
            return transcript
            
        except Exception as e:
            self.error.emit(f"Whisper transcription error: {str(e)}")
            return []
    
    def format_transcript(self, transcript: List[Dict], format_type: str = "text") -> str:
        """Format transcript for output"""
        if format_type == "text":
            return "\n".join([item['text'] for item in transcript])
        
        elif format_type == "srt":
            srt_content = []
            for i, item in enumerate(transcript, 1):
                start_time = self._seconds_to_srt_time(item['start'])
                end_time = self._seconds_to_srt_time(item['start'] + item.get('duration', 5))
                srt_content.append(f"{i}")
                srt_content.append(f"{start_time} --> {end_time}")
                srt_content.append(item['text'])
                srt_content.append("")
            return "\n".join(srt_content)
        
        elif format_type == "json":
            return json.dumps(transcript, indent=2, ensure_ascii=False)
        
        elif format_type == "vtt":
            vtt_content = ["WEBVTT", ""]
            for item in transcript:
                start_time = self._seconds_to_vtt_time(item['start'])
                end_time = self._seconds_to_vtt_time(item['start'] + item.get('duration', 5))
                vtt_content.append(f"{start_time} --> {end_time}")
                vtt_content.append(item['text'])
                vtt_content.append("")
            return "\n".join(vtt_content)
        
        return ""
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format"""
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        seconds = td.seconds % 60
        milliseconds = td.microseconds // 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    
    def _seconds_to_vtt_time(self, seconds: float) -> str:
        """Convert seconds to WebVTT timestamp format"""
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        seconds = td.seconds % 60
        milliseconds = td.microseconds // 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    
    def run(self):
        """Enhanced main worker thread execution"""
        try:
            # Extract video ID
            video_id = self.extract_video_id(self.url)
            if not video_id:
                self.error.emit("Invalid YouTube URL")
                return
            
            all_transcripts = []
            
            # Try YouTube transcript first
            if "youtube" in self.methods or "auto" in self.methods:
                youtube_transcript = self.get_youtube_transcript(video_id)
                if youtube_transcript:
                    # Convert to our format
                    segments = self._convert_youtube_transcript(youtube_transcript)
                    all_transcripts.extend(segments)
                    self.status.emit("✓ YouTube transcript extracted")
            
            # Try speech recognition if needed
            if not all_transcripts or self.use_multi_method:
                audio_path = self.download_audio(self.url)
                if audio_path:
                    # Try multiple methods
                    for method in self.methods:
                        if method == "youtube":
                            continue
                        
                        try:
                            self.status.emit(f"Transcribing with {method}...")
                            segments = self.transcriber.transcribe_with_method(
                                audio_path, method, self.language, self.api_keys
                            )
                            
                            if segments:
                                all_transcripts.extend(segments)
                                self.status.emit(f"✓ {method} transcription complete")
                                
                                # Send partial result
                                self.partial_result.emit({
                                    'method': method,
                                    'segments': segments
                                })
                        except Exception as e:
                            self.status.emit(f"✗ {method} failed: {str(e)}")
                    
                    # Clean up
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
            
            if all_transcripts:
                # Merge and analyze transcripts
                merged_transcript = self._merge_transcripts(all_transcripts)
                analysis = self.analyzer.analyze_transcript(merged_transcript)
                
                # Format results
                result_data = {
                    'video_id': video_id,
                    'url': self.url,
                    'segments': merged_transcript,
                    'analysis': analysis,
                    'text': self._format_segments_to_text(merged_transcript),
                    'srt': self._format_segments_to_srt(merged_transcript),
                    'json': json.dumps([asdict(seg) for seg in merged_transcript], indent=2),
                    'vtt': self._format_segments_to_vtt(merged_transcript),
                    'word_level_json': self._format_word_level_json(merged_transcript),
                }
                
                self.progress.emit(100)
                self.result.emit(result_data)
            else:
                self.error.emit("Could not extract transcript with any method")
                
        except Exception as e:
            self.error.emit(f"Unexpected error: {str(e)}")
    
    def _convert_youtube_transcript(self, youtube_transcript: List[Dict]) -> List[TranscriptSegment]:
        """Convert YouTube transcript to our format"""
        segments = []
        for item in youtube_transcript:
            # Estimate words
            words = self.transcriber._estimate_word_timestamps(
                item['text'], 
                item['start'], 
                item.get('duration', 5),
                0.9  # High confidence for YouTube captions
            )
            
            segment = TranscriptSegment(
                text=item['text'],
                start=item['start'],
                end=item['start'] + item.get('duration', 5),
                confidence=0.9,
                words=words,
                method='youtube'
            )
            segments.append(segment)
        
        return segments
    
    def _merge_transcripts(self, all_segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Merge multiple transcript sources intelligently"""
        
        # Group by time ranges
        time_groups = {}
        for seg in all_segments:
            key = (int(seg.start), int(seg.end))
            if key not in time_groups:
                time_groups[key] = []
            time_groups[key].append(seg)
        
        merged = []
        for time_key, segments in sorted(time_groups.items()):
            if len(segments) == 1:
                merged.append(segments[0])
            else:
                # Vote on best transcript based on confidence
                best_segment = max(segments, key=lambda s: s.confidence)
                
                # Combine word-level data if available
                all_words = []
                for seg in segments:
                    if seg.words:
                        all_words.extend(seg.words)
                
                # Deduplicate and sort words
                if all_words:
                    unique_words = {}
                    for word in all_words:
                        key = (word.word.lower(), round(word.start, 1))
                        if key not in unique_words or word.confidence > unique_words[key].confidence:
                            unique_words[key] = word
                    
                    best_segment.words = sorted(unique_words.values(), key=lambda w: w.start)
                
                # Update method to show combination
                methods = list(set(seg.method for seg in segments))
                best_segment.method = "+".join(methods)
                
                merged.append(best_segment)
        
        return merged
    
    def _format_segments_to_text(self, segments: List[TranscriptSegment]) -> str:
        """Format segments as plain text with confidence indicators"""
        lines = []
        for seg in segments:
            confidence_marker = ""
            if seg.confidence < 0.3:
                confidence_marker = " [LOW CONFIDENCE]"
            elif seg.confidence < 0.6:
                confidence_marker = " [MEDIUM CONFIDENCE]"
            
            lines.append(f"{seg.text}{confidence_marker}")
        
        return "\n".join(lines)
    
    def _format_segments_to_srt(self, segments: List[TranscriptSegment]) -> str:
        """Format segments as SRT with word-level precision"""
        srt_content = []
        for i, seg in enumerate(segments, 1):
            start_time = self._seconds_to_srt_time(seg.start)
            end_time = self._seconds_to_srt_time(seg.end)
            
            # Add confidence in brackets if low
            text = seg.text
            if seg.confidence < 0.5:
                text = f"({int(seg.confidence * 100)}%) {text}"
            
            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")
        
        return "\n".join(srt_content)
    
    def _format_segments_to_vtt(self, segments: List[TranscriptSegment]) -> str:
        """Format segments as WebVTT with metadata"""
        vtt_content = ["WEBVTT", "Kind: captions", "Language: " + segments[0].language if segments else "en", ""]
        
        for seg in segments:
            start_time = self._seconds_to_vtt_time(seg.start)
            end_time = self._seconds_to_vtt_time(seg.end)
            
            # Add speaker label if available
            if seg.speaker:
                vtt_content.append(f"<v {seg.speaker}>")
            
            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(seg.text)
            vtt_content.append("")
        
        return "\n".join(vtt_content)
    
    def _format_word_level_json(self, segments: List[TranscriptSegment]) -> str:
        """Format word-level transcript with all metadata"""
        word_data = []
        
        for seg in segments:
            for word in seg.words:
                word_data.append({
                    'word': word.word,
                    'start': word.start,
                    'end': word.end,
                    'confidence': word.confidence,
                    'segment_confidence': seg.confidence,
                    'method': seg.method,
                    'speaker': word.speaker or seg.speaker
                })
        
        return json.dumps(word_data, indent=2)
    
    # ...existing code...

class YouTubeTranscriptExtractor(QMainWindow):
    """Enhanced GUI for multi-method YouTube transcript extraction"""
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.current_transcript = None
        self.api_keys = {}  # Store API keys
        self.init_ui()
        
    def init_ui(self):
        """Initialize enhanced user interface"""
        self.setWindowTitle("YouTube Transcript Extractor Pro - KingAI Suite")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #2d2d2d;
                color: #ffffff;
                font-family: 'Segoe UI', Tahoma, sans-serif;
            }
            QLineEdit, QTextEdit, QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #0e639c;
                border: none;
                padding: 8px 15px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
            QGroupBox {
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0e639c;
                border-radius: 2px;
            }
        """)
        
        # Central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Main extraction tab
        extraction_tab = QWidget()
        self.tab_widget.addTab(extraction_tab, "Extraction")
        self._setup_extraction_tab(extraction_tab)
        
        # Analysis tab
        analysis_tab = QWidget()
        self.tab_widget.addTab(analysis_tab, "Analysis")
        self._setup_analysis_tab(analysis_tab)
        
        # Settings tab
        settings_tab = QWidget()
        self.tab_widget.addTab(settings_tab, "Settings")
        self._setup_settings_tab(settings_tab)
    
    def _setup_extraction_tab(self, parent):
        """Setup main extraction interface"""
        layout = QVBoxLayout(parent)
        
        # URL Input Group
        url_group = QGroupBox("YouTube Video URL")
        url_layout = QHBoxLayout()
        
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://www.youtube.com/watch?v=...")
        url_layout.addWidget(self.url_input)
        
        self.paste_button = QPushButton("📋 Paste")
        self.paste_button.clicked.connect(self.paste_url)
        url_layout.addWidget(self.paste_button)
        
        url_group.setLayout(url_layout)
        layout.addWidget(url_group)
        
        # Enhanced options group
        options_group = QGroupBox("Extraction Options")
        options_layout = QVBoxLayout()
        
        # Method selection (multi-select)
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Methods:"))
        
        self.method_checks = {}
        methods = [
            ("YouTube CC", "youtube"),
            ("Google", "google"),
            ("Whisper", "whisper"),
            ("Vosk (Offline)", "vosk"),
            ("Azure", "azure"),
            ("IBM Watson", "ibm"),
            ("AssemblyAI", "assemblyai"),
            ("Deepgram", "deepgram"),
            ("Sphinx (Fallback)", "sphinx")
        ]
        
        for label, key in methods:
            check = QCheckBox(label)
            check.setChecked(key in ["youtube", "google"])
            self.method_checks[key] = check
            method_layout.addWidget(check)
        
        options_layout.addLayout(method_layout)
        
        # Multi-method options
        multi_layout = QHBoxLayout()
        self.multi_method_check = QCheckBox("Use multiple methods for comparison")
        multi_layout.addWidget(self.multi_method_check)
        
        multi_layout.addWidget(QLabel("Confidence Threshold:"))
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.1)
        self.confidence_spin.setValue(0.5)
        multi_layout.addWidget(self.confidence_spin)
        
        options_layout.addLayout(multi_layout)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.extract_button = QPushButton("🎬 Extract Transcript")
        self.extract_button.clicked.connect(self.extract_transcript)
        control_layout.addWidget(self.extract_button)
        
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_extraction)
        control_layout.addWidget(self.stop_button)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)
        
        # Output Group
        output_group = QGroupBox("Transcript Output")
        output_layout = QVBoxLayout()
        
        # Format selection and export buttons
        format_layout = QHBoxLayout()
        
        format_layout.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["Plain Text", "SRT Subtitles", "WebVTT", "JSON"])
        self.format_combo.currentIndexChanged.connect(self.update_output_format)
        format_layout.addWidget(self.format_combo)
        
        format_layout.addStretch()
        
        self.copy_button = QPushButton("📋 Copy")
        self.copy_button.clicked.connect(self.copy_transcript)
        self.copy_button.setEnabled(False)
        format_layout.addWidget(self.copy_button)
        
        self.save_button = QPushButton("💾 Save As...")
        self.save_button.clicked.connect(self.save_transcript)
        self.save_button.setEnabled(False)
        format_layout.addWidget(self.save_button)
        
        output_layout.addLayout(format_layout)
        
        # Transcript display
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Consolas", 10))
        output_layout.addWidget(self.output_text)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
    def paste_url(self):
        """Paste URL from clipboard"""
        clipboard = QApplication.clipboard()
        self.url_input.setText(clipboard.text())
    
    def extract_transcript(self):
        """Enhanced transcript extraction with multiple methods"""
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Warning", "Please enter a YouTube URL")
            return
        
        # Get selected methods
        selected_methods = [
            key for key, check in self.method_checks.items() 
            if check.isChecked()
        ]
        
        if not selected_methods:
            QMessageBox.warning(self, "Warning", "Please select at least one method")
            return
        
        # Disable controls during extraction
        self.extract_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.copy_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.output_text.clear()
        self.progress_bar.setValue(0)
        
        # Create and configure worker
        self.worker = TranscriptWorker()
        self.worker.url = url
        self.worker.methods = selected_methods
        self.worker.use_multi_method = self.multi_method_check.isChecked()
        self.worker.confidence_threshold = self.confidence_spin.value()
        
        # Set API keys
        for key, input_field in self.api_inputs.items():
            value = input_field.text().strip()
            if value:
                self.worker.api_keys[key] = value
        
        # Connect signals
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.update_status)
        self.worker.result.connect(self.handle_result)
        self.worker.error.connect(self.handle_error)
        self.worker.partial_result.connect(self.handle_partial_result)
        
        # Start extraction
        self.worker.start()
    
    def stop_extraction(self):
        """Stop ongoing extraction"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.update_status("Extraction stopped")
            self.reset_controls()
    
    def update_status(self, message: str):
        """Update status label"""
        self.status_label.setText(message)
    
    def handle_result(self, result: dict):
        """Handle enhanced results with analysis"""
        self.current_transcript = result
        self.update_output_format()
        self.reset_controls()
        self.copy_button.setEnabled(True)
        self.save_button.setEnabled(True)
        
        # Update analysis tab
        self._update_analysis(result['analysis'])
        
        # Show enhanced summary
        analysis = result['analysis']
        QMessageBox.information(
            self,
            "Success",
            f"Transcript extracted successfully!\n\n"
            f"Video ID: {result['video_id']}\n"
            f"Methods Used: {', '.join(analysis['methods_used'])}\n"
            f"Word Count: {analysis['word_count']}\n"
            f"Average Confidence: {analysis['average_confidence']:.1%}\n"
            f"Readability Score: {analysis['readability_score']:.1f}\n"
            f"Grade Level: {analysis['grade_level']:.1f}"
        )
        
        self.update_status("✅ Transcript extracted and analyzed successfully!")
    
    def handle_partial_result(self, result: dict):
        """Handle streaming partial results"""
        method = result['method']
        segments = result['segments']
        
        # Update status with partial results
        avg_confidence = sum(s.confidence for s in segments) / len(segments) if segments else 0
        self.update_status(f"✓ {method}: {len(segments)} segments, {avg_confidence:.1%} confidence")
    
    def handle_error(self, error: str):
        """Handle extraction error"""
        self.reset_controls()
        self.update_status(f"❌ Error: {error}")
        QMessageBox.critical(self, "Error", error)
    
    def reset_controls(self):
        """Reset UI controls to default state"""
        self.extract_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(0)
    
    def update_output_format(self):
        """Update output display based on selected format"""
        if not self.current_transcript:
            return
        
        format_map = {
            0: 'text',
            1: 'srt',
            2: 'vtt',
            3: 'json'
        }
        
        format_type = format_map[self.format_combo.currentIndex()]
        self.output_text.setPlainText(self.current_transcript[format_type])
    
    def copy_transcript(self):
        """Copy transcript to clipboard"""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.output_text.toPlainText())
        self.update_status("📋 Copied to clipboard")
    
    def save_transcript(self):
        """Save transcript to file"""
        if not self.current_transcript:
            return
        
        # Get file extension based on format
        format_extensions = {
            0: ("Text Files (*.txt)", ".txt"),
            1: ("SRT Files (*.srt)", ".srt"),
            2: ("WebVTT Files (*.vtt)", ".vtt"),
            3: ("JSON Files (*.json)", ".json")
        }
        
        filter_str, extension = format_extensions[self.format_combo.currentIndex()]
        
        # Open file dialog
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Transcript",
            f"transcript_{self.current_transcript['video_id']}{extension}",
            filter_str
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.output_text.toPlainText())
                self.update_status(f"✅ Saved to {filename}")
                QMessageBox.information(self, "Success", f"Transcript saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")

    def _setup_analysis_tab(self, parent):
        """Setup transcript analysis interface"""
        layout = QVBoxLayout(parent)
        
        # Statistics display
        stats_group = QGroupBox("Transcript Statistics")
        stats_layout = QVBoxLayout()
        
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        stats_layout.addWidget(self.stats_table)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Confidence visualization
        confidence_group = QGroupBox("Confidence Analysis")
        confidence_layout = QVBoxLayout()
        
        self.confidence_text = QTextEdit()
        self.confidence_text.setReadOnly(True)
        confidence_layout.addWidget(self.confidence_text)
        
        confidence_group.setLayout(confidence_layout)
        layout.addWidget(confidence_group)
    
    def _update_analysis(self, analysis: Dict[str, Any]):
        """Update analysis tab with statistics"""
        # Update statistics table
        self.stats_table.setRowCount(len(analysis))
        row = 0
        
        for key, value in analysis.items():
            if key == 'most_common_words':
                value = ', '.join([f"{word}({count})" for word, count in value[:5]])
            elif key == 'confidence_distribution':
                value = f"H:{value['high']} M:{value['medium']} L:{value['low']}"
            elif isinstance(value, float):
                value = f"{value:.2f}"
            elif isinstance(value, list):
                value = ', '.join(map(str, value))
            
            self.stats_table.setItem(row, 0, QTableWidgetItem(key.replace('_', ' ').title()))
            self.stats_table.setItem(row, 1, QTableWidgetItem(str(value)))
            row += 1
        
        # Update confidence visualization
        if 'confidence_distribution' in analysis:
            dist = analysis['confidence_distribution']
            confidence_text = f"""
Confidence Distribution:
{'=' * 50}
High (≥80%):   {'█' * dist['high']} {dist['high']} segments
Medium (50-79%): {'█' * dist['medium']} {dist['medium']} segments  
Low (<50%):     {'█' * dist['low']} {dist['low']} segments
            """
            self.confidence_text.setPlainText(confidence_text)

def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("YouTube Transcript Extractor Pro")
    app.setOrganizationName("KingAI")
    
    # Set application icon if available
    icon_path = Path(__file__).parent / "icon.png"
    if icon_path.exists():
        from PySide6.QtGui import QIcon
        app.setWindowIcon(QIcon(str(icon_path)))
    
    window = YouTubeTranscriptExtractor()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()