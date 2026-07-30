"""Local media transcription CLI, companion to the YouTube transcript tools.

The YouTube scrapers fetch captions that already exist on YouTube. This script
covers the other case: a local audio or video file (a downloaded briefing, a
recorded call, an MP4 with no captions) that needs a fresh speech-to-text
transcript. It uses ffmpeg to extract audio and faster-whisper to transcribe,
then emits the SAME stable JSON / Markdown / text schema the YouTube CLI emits,
so local transcripts drop straight into the same compare/diff workflow.

Design rules (match the rest of the repo):
- No hard-coded paths. Every input is a CLI argument.
- Same output flags as youtube_transcript_cli: --format, --output-dir, --stdout.
- Deterministic document (no wall-clock fields) so two runs diff cleanly.
- Lazy heavy imports so --help and --version work with nothing installed.

Examples:
    # one file, Markdown next to nothing (write into ./out)
    python local_media_transcribe.py briefing2.mp4 --format markdown --output-dir out

    # several files at once, to compare (one <stem>.json per input)
    python local_media_transcribe.py brief1.mp4 brief2.mp4 odis101.mp4 --output-dir out

    # a whole folder of recordings, GPU, higher-accuracy model
    python local_media_transcribe.py ./recordings --model medium --device cuda --output-dir out

    # quick look on screen
    python local_media_transcribe.py call.m4a --stdout --format text

Dependencies (install once):
    pip install faster-whisper          # transcription engine (CUDA optional)
    ffmpeg on PATH                      # audio extraction (or pass --ffmpeg PATH)
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence, TextIO

__version__ = "1.0.0"
GENERATOR_NAME = "youtube-transcript-extractor local-media"
SCHEMA_VERSION = "1.0"

FORMAT_EXTENSIONS = {"json": ".json", "markdown": ".md", "text": ".txt"}
MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v",
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".oga", ".opus", ".aac", ".wma",
}

# process exit codes, mirroring the sibling CLI's convention
EXIT_OK = 0
EXIT_USAGE = 1
EXIT_DEPENDENCY = 2
EXIT_RUNTIME = 3


# --------------------------------------------------------------------------- #
# input resolution
# --------------------------------------------------------------------------- #
def resolve_inputs(sources: Sequence[str]) -> list[Path]:
    """Expand files, directories and globs into a de-duplicated media list."""
    found: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            return
        if resolved.suffix.lower() not in MEDIA_EXTENSIONS:
            return
        seen.add(resolved)
        found.append(resolved)

    for raw in sources:
        candidate = Path(raw)
        if candidate.is_dir():
            for child in sorted(candidate.rglob("*")):
                add(child)
        elif candidate.is_file():
            add(candidate)
        else:
            # Treat it as a glob. Use the glob module rather than Path().glob,
            # which raises NotImplementedError on an absolute pattern, so a
            # mistyped absolute path returns the documented usage error instead
            # of an unhandled exception.
            for match in sorted(glob.glob(raw, recursive=True)):
                add(Path(match))
    return found


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# audio extraction
# --------------------------------------------------------------------------- #
def extract_audio(media: Path, wav_path: Path, ffmpeg: str) -> None:
    """Extract 16 kHz mono PCM WAV, the format the ASR models expect."""
    command = [
        ffmpeg, "-nostdin", "-y", "-i", str(media),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-3:]
        raise RuntimeError(f"ffmpeg failed for {media.name}: {' | '.join(tail)}")


# --------------------------------------------------------------------------- #
# transcription
# --------------------------------------------------------------------------- #
def _pick_compute_type(device: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "int8_float16" if device == "cuda" else "int8"


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:  # prefer GPU when torch reports CUDA, else CPU
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_model(model_name: str, device: str, compute_type: str) -> Any:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from exc
    try:
        return WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception:
        if device == "cuda":  # graceful CPU fallback (missing cuDNN/cuBLAS etc.)
            return WhisperModel(model_name, device="cpu", compute_type="int8")
        raise


def transcribe_media(
    media: Path,
    *,
    model: Any,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
    beam_size: int,
    vad_filter: bool,
    ffmpeg: str,
    requested_input: str,
) -> dict[str, Any]:
    """Transcribe one media file into the stable transcript document schema."""
    faster_whisper_version = _dependency_version("faster_whisper")
    with tempfile.TemporaryDirectory(prefix="lmt_") as tmp:
        wav_path = Path(tmp) / "audio.wav"
        extract_audio(media, wav_path, ffmpeg)
        segments_iter, info = model.transcribe(
            str(wav_path),
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
        segments = [
            {
                "index": index,
                "text": segment.text.strip(),
                "start": float(segment.start),
                "duration": float(segment.end) - float(segment.start),
            }
            for index, segment in enumerate(segments_iter, start=1)
        ]

    language_code = str(getattr(info, "language", language or "unknown"))
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": {"name": GENERATOR_NAME, "version": __version__},
        "source": {
            "platform": "local_media",
            "media_id": media.stem,
            "file_name": media.name,
            "file_sha256": _sha256(media),
            "requested_input": requested_input,
            "duration_seconds": round(float(getattr(info, "duration", 0.0)), 3),
        },
        "extraction": {
            "library": "faster-whisper",
            "library_version": faster_whisper_version,
            "model": model_name,
            "device": device,
            "compute_type": compute_type,
            "beam_size": beam_size,
            "vad_filter": vad_filter,
        },
        "transcript": {
            "language": language_code,
            "language_code": language_code,
            "language_probability": round(float(getattr(info, "language_probability", 0.0)), 4),
            "is_generated": True,
            "segment_count": len(segments),
            "segments": segments,
        },
    }


def _dependency_version(module_name: str) -> str:
    try:
        from importlib.metadata import version

        return version(module_name.replace("_", "-"))
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# rendering (matches youtube_transcript_cli output shape)
# --------------------------------------------------------------------------- #
def _timestamp(seconds: float) -> str:
    total_milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _markdown_document(document: dict[str, Any], heading_level: int = 1) -> str:
    source = document["source"]
    extraction = document["extraction"]
    transcript = document["transcript"]
    heading = "#" * heading_level
    transcript_heading = "#" * (heading_level + 1)
    lines = [
        f"{heading} Local Media Transcript: {source['media_id']}",
        "",
        f"- File: {source['file_name']}",
        f"- SHA-256: {source['file_sha256']}",
        f"- Duration: {source['duration_seconds']} s",
        f"- Language: {transcript['language']} (p={transcript['language_probability']})",
        f"- Model: {extraction['model']} on {extraction['device']} ({extraction['compute_type']})",
        f"- Segments: {transcript['segment_count']}",
        f"- Extractor: {GENERATOR_NAME} {__version__} via faster-whisper {extraction['library_version']}",
        "",
        f"{transcript_heading} Transcript",
        "",
    ]
    lines.extend(
        f"[{_timestamp(segment['start'])}] {segment['text']}"
        for segment in transcript["segments"]
    )
    return "\n".join(lines).rstrip() + "\n"


def _text_document(document: dict[str, Any]) -> str:
    source = document["source"]
    extraction = document["extraction"]
    transcript = document["transcript"]
    lines = [
        "LOCAL MEDIA TRANSCRIPT",
        f"media_id: {source['media_id']}",
        f"file: {source['file_name']}",
        f"sha256: {source['file_sha256']}",
        f"language: {transcript['language']}",
        f"model: {extraction['model']} ({extraction['device']}/{extraction['compute_type']})",
        f"segments: {transcript['segment_count']}",
        f"extractor: {GENERATOR_NAME} {__version__}",
        "---",
    ]
    lines.extend(
        f"[{_timestamp(segment['start'])}] {segment['text']}"
        for segment in transcript["segments"]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_document(document: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if output_format == "markdown":
        return _markdown_document(document)
    if output_format == "text":
        return _text_document(document)
    raise ValueError(f"unsupported output format: {output_format}")


def render_batch(documents: Sequence[dict[str, Any]], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(list(documents), ensure_ascii=False, indent=2) + "\n"
    if output_format == "markdown":
        return "\n---\n\n".join(_markdown_document(doc, heading_level=2) for doc in documents)
    if output_format == "text":
        return ("\n" + "=" * 8 + "\n").join(_text_document(doc) for doc in documents)
    raise ValueError(f"unsupported output format: {output_format}")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(content)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-media-transcribe",
        description=(
            "Transcribe local audio/video files to the same JSON, Markdown or "
            "text schema as the YouTube transcript tools, for side-by-side use."
        ),
    )
    parser.add_argument("media", nargs="+", metavar="FILE_DIR_OR_GLOB")
    parser.add_argument("--format", choices=tuple(FORMAT_EXTENSIONS), default="json",
                        help="output format (default: json)")
    parser.add_argument("--output-dir", type=Path,
                        help="write one <stem>.<extension> file per input")
    parser.add_argument("--stdout", action="store_true", dest="to_stdout",
                        help="write a single batch document to standard output")
    parser.add_argument("--model", default="small",
                        help="whisper model: tiny|base|small|medium|large-v3 (default: small)")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"),
                        help="compute device (default: auto -> cuda if available)")
    parser.add_argument("--compute-type", default="auto",
                        help="ctranslate2 compute type or 'auto' (default: auto)")
    parser.add_argument("--language", default=None, metavar="CODE",
                        help="force a language code (default: auto-detect)")
    parser.add_argument("--beam-size", type=int, default=5, help="beam size (default: 5)")
    parser.add_argument("--vad-filter", action="store_true",
                        help="apply voice-activity detection to drop silence")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable (default: ffmpeg)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def run_cli(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None,
            stderr: TextIO | None = None) -> int:
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    args = build_parser().parse_args(argv)

    if args.output_dir is None and not args.to_stdout:
        print("error: at least one output destination is required: --output-dir or --stdout", file=stderr)
        return EXIT_USAGE

    if shutil.which(args.ffmpeg) is None and not Path(args.ffmpeg).is_file():
        print(f"error: ffmpeg not found ('{args.ffmpeg}'). Install ffmpeg or pass --ffmpeg PATH.", file=stderr)
        return EXIT_DEPENDENCY

    media_files = resolve_inputs(args.media)
    if not media_files:
        print("error: no supported media files matched the given inputs", file=stderr)
        return EXIT_USAGE

    device = _resolve_device(args.device)
    compute_type = _pick_compute_type(device, args.compute_type)
    try:
        model = load_model(args.model, device, compute_type)
    except ModuleNotFoundError as exc:
        print(f"error: {exc}", file=stderr)
        return EXIT_DEPENDENCY

    documents: list[dict[str, Any]] = []
    exit_code = EXIT_OK
    for media in media_files:
        print(f"transcribing {media.name} ({args.model}/{device}) ...", file=stderr)
        try:
            document = transcribe_media(
                media, model=model, model_name=args.model, device=device,
                compute_type=compute_type, language=args.language,
                beam_size=args.beam_size, vad_filter=args.vad_filter,
                ffmpeg=args.ffmpeg, requested_input=str(media),
            )
        except Exception as exc:  # keep going through the batch
            print(f"  failed: {type(exc).__name__}: {exc}", file=stderr)
            exit_code = EXIT_RUNTIME
            continue
        documents.append(document)
        if args.output_dir is not None:
            out_path = args.output_dir / f"{media.stem}{FORMAT_EXTENSIONS[args.format]}"
            _write_text(out_path, render_document(document, args.format))
            print(f"  wrote {out_path} ({document['transcript']['segment_count']} segments)", file=stderr)

    if args.to_stdout and documents:
        stdout.write(render_batch(documents, args.format))

    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
