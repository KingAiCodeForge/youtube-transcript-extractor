#!/usr/bin/env python3
"""Deterministic, non-interactive YouTube transcript CLI.

This module intentionally imports only the Python standard library at startup.
In particular, it never imports the GUI modules and never installs packages.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TextIO
from urllib.parse import parse_qs, urlparse


__version__ = "1.0.0"
SCHEMA_VERSION = "1.0"
GENERATOR_NAME = "youtube-transcript-extractor"

EXIT_SUCCESS = 0
EXIT_PARTIAL = 1
EXIT_USAGE = 2
EXIT_EXTRACTION = 3
EXIT_DEPENDENCY = 4
EXIT_OUTPUT = 5

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
FORMAT_EXTENSIONS = {"json": ".json", "markdown": ".md", "text": ".txt"}


class DependencyUnavailable(RuntimeError):
    """Raised when the transcript dependency cannot be imported."""


class CliUsageError(ValueError):
    """Raised for command-line syntax errors that should be emitted as JSON."""


class JsonArgumentParser(argparse.ArgumentParser):
    """Convert argparse failures into the CLI's machine-readable error path."""

    def error(self, message: str) -> None:
        raise CliUsageError(message)


def extract_video_id(value: str) -> str:
    """Return an 11-character ID from a YouTube ID or supported URL.

    Supported URL forms include watch, youtu.be, shorts, live, and embed URLs.
    A ``ValueError`` is raised instead of guessing when input is ambiguous.
    """

    candidate = value.strip()
    if VIDEO_ID_RE.fullmatch(candidate):
        return candidate

    parsed = urlparse(candidate)
    if not parsed.scheme and not parsed.netloc:
        parsed = urlparse(f"https://{candidate}")

    hostname = (parsed.hostname or "").lower().rstrip(".")
    path_parts = [part for part in parsed.path.split("/") if part]
    video_id: str | None = None

    if hostname == "youtu.be" or hostname.endswith(".youtu.be"):
        if path_parts:
            video_id = path_parts[0]
    elif (
        hostname == "youtube.com"
        or hostname.endswith(".youtube.com")
        or hostname == "youtube-nocookie.com"
        or hostname.endswith(".youtube-nocookie.com")
    ):
        if parsed.path.rstrip("/") == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        elif len(path_parts) >= 2 and path_parts[0].lower() in {
            "embed",
            "live",
            "shorts",
            "v",
        }:
            video_id = path_parts[1]

    if video_id and VIDEO_ID_RE.fullmatch(video_id):
        return video_id

    raise ValueError(
        "not a supported YouTube URL or 11-character video ID: "
        f"{value!r}"
    )


def _load_api_factory() -> Callable[[], Any]:
    """Load the instance-based API without installation side effects."""

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        raise DependencyUnavailable(
            "youtube-transcript-api is not installed; run "
            "'python -m pip install youtube-transcript-api'"
        ) from exc
    return YouTubeTranscriptApi


def _dependency_version() -> str:
    try:
        return metadata.version("youtube-transcript-api")
    except metadata.PackageNotFoundError:
        return "unknown"


def _generator() -> dict[str, str]:
    return {"name": GENERATOR_NAME, "version": __version__}


def transcript_document(
    *,
    fetched: Any,
    video_id: str,
    requested_input: str,
    requested_languages: Sequence[str],
    preserve_formatting: bool,
    dependency_version: str,
) -> dict[str, Any]:
    """Convert ``FetchedTranscript`` data to the stable output schema."""

    snippets: Iterable[Any] = getattr(fetched, "snippets", fetched)
    segments = [
        {
            "index": index,
            "text": str(snippet.text),
            "start": float(snippet.start),
            "duration": float(snippet.duration),
        }
        for index, snippet in enumerate(snippets, start=1)
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "generator": _generator(),
        "source": {
            "platform": "youtube",
            "video_id": video_id,
            "canonical_url": f"https://www.youtube.com/watch?v={video_id}",
            "requested_input": requested_input,
        },
        "extraction": {
            "library": "youtube-transcript-api",
            "library_version": dependency_version,
            "requested_languages": list(requested_languages),
            "preserve_formatting": preserve_formatting,
        },
        "transcript": {
            "language": str(getattr(fetched, "language", "unknown")),
            "language_code": str(getattr(fetched, "language_code", "unknown")),
            "is_generated": bool(getattr(fetched, "is_generated", False)),
            "segment_count": len(segments),
            "segments": segments,
        },
    }


def _timestamp(seconds: float) -> str:
    total_milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _markdown_document(
    document: dict[str, Any], heading_level: int = 1
) -> str:
    source = document["source"]
    extraction = document["extraction"]
    transcript = document["transcript"]
    heading = "#" * heading_level
    transcript_heading = "#" * (heading_level + 1)
    caption_kind = "auto-generated" if transcript["is_generated"] else "manual"
    languages = ", ".join(extraction["requested_languages"])
    lines = [
        f"{heading} YouTube Transcript: {source['video_id']}",
        "",
        f"- Source: {source['canonical_url']}",
        (
            f"- Language: {transcript['language']} "
            f"({transcript['language_code']})"
        ),
        f"- Caption type: {caption_kind}",
        f"- Requested languages: {languages}",
        f"- Segments: {transcript['segment_count']}",
        f"- Extractor: {GENERATOR_NAME} {__version__}",
        (
            "- Transcript library: youtube-transcript-api "
            f"{extraction['library_version']}"
        ),
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
        "YOUTUBE TRANSCRIPT",
        f"video_id: {source['video_id']}",
        f"source: {source['canonical_url']}",
        f"language: {transcript['language']} ({transcript['language_code']})",
        f"generated: {str(transcript['is_generated']).lower()}",
        f"segments: {transcript['segment_count']}",
        f"extractor: {GENERATOR_NAME} {__version__}",
        (
            "transcript_library: youtube-transcript-api "
            f"{extraction['library_version']}"
        ),
        "---",
    ]
    lines.extend(
        f"[{_timestamp(segment['start'])}] {segment['text']}"
        for segment in transcript["segments"]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_document(document: dict[str, Any], output_format: str) -> str:
    """Render one transcript with stable ordering and no wall-clock fields."""

    if output_format == "json":
        return json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if output_format == "markdown":
        return _markdown_document(document)
    if output_format == "text":
        return _text_document(document)
    raise ValueError(f"unsupported output format: {output_format}")


def render_batch(
    documents: Sequence[dict[str, Any]], output_format: str
) -> str:
    """Render successful batch results as one valid stdout document."""

    if output_format == "json":
        batch = {
            "schema_version": SCHEMA_VERSION,
            "generator": _generator(),
            "video_count": len(documents),
            "videos": list(documents),
        }
        return json.dumps(batch, ensure_ascii=False, indent=2) + "\n"
    if output_format == "markdown":
        body = "\n---\n\n".join(
            _markdown_document(document, heading_level=2).rstrip()
            for document in documents
        )
        return (
            f"# YouTube Transcript Export\n\nVideos: {len(documents)}\n\n"
            f"{body}\n"
        )
    if output_format == "text":
        return ("\n" + "=" * 80 + "\n").join(
            _text_document(document).rstrip() for document in documents
        ) + "\n"
    raise ValueError(f"unsupported output format: {output_format}")


def _write_error(
    stream: TextIO, code: str, message: str, **context: Any
) -> None:
    error = {"code": code, "message": message}
    error.update(
        {key: value for key, value in context.items() if value is not None}
    )
    stream.write(
        json.dumps({"error": error}, ensure_ascii=False, sort_keys=True) + "\n"
    )


def _write_text(path: Path, content: str, *, overwrite: bool = False) -> None:
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8", newline="\n") as output_file:
        output_file.write(content)


def build_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(
        prog="youtube-transcript",
        description=(
            "Fetch one or more YouTube transcripts as deterministic JSON, "
            "Markdown, or plain text."
        ),
    )
    parser.add_argument("sources", nargs="+", metavar="URL_OR_ID")
    parser.add_argument(
        "--format",
        choices=tuple(FORMAT_EXTENSIONS),
        default="json",
        help="output format (default: json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write one <video-id>.<extension> file per video",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing transcript files (default: fail safely)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        dest="to_stdout",
        help="write a single batch document to standard output",
    )
    parser.add_argument(
        "--language",
        action="append",
        dest="languages",
        metavar="CODE",
        help="preferred language code; repeat in priority order (default: en)",
    )
    parser.add_argument(
        "--preserve-formatting",
        action="store_true",
        help="preserve supported HTML caption formatting",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    api_factory: Callable[[], Any] | None = None,
    dependency_version: str | None = None,
) -> int:
    """Run the CLI and return a documented process exit code.

    ``api_factory`` and ``dependency_version`` are injection points for offline
    tests; normal callers should leave them unset.
    """

    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except CliUsageError as exc:
        _write_error(stderr, "invalid_arguments", str(exc))
        return EXIT_USAGE

    if args.output_dir is None and not args.to_stdout:
        _write_error(
            stderr,
            "invalid_arguments",
            "at least one output destination is required: "
            "--output-dir or --stdout",
        )
        return EXIT_USAGE

    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source in args.sources:
        try:
            video_id = extract_video_id(source)
        except ValueError as exc:
            _write_error(stderr, "invalid_input", str(exc), input=source)
            return EXIT_USAGE
        if video_id not in seen:
            normalized.append((source, video_id))
            seen.add(video_id)

    output_dir: Path | None = args.output_dir
    if output_dir is not None:
        output_dir = output_dir.expanduser()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            if not output_dir.is_dir():
                raise NotADirectoryError(str(output_dir))
        except OSError as exc:
            _write_error(
                stderr,
                "output_failed",
                str(exc),
                path=str(output_dir),
                exception_type=type(exc).__name__,
            )
            return EXIT_OUTPUT

    if api_factory is None:
        try:
            api_factory = _load_api_factory()
        except DependencyUnavailable as exc:
            _write_error(stderr, "dependency_missing", str(exc))
            return EXIT_DEPENDENCY
    dependency_version = dependency_version or _dependency_version()

    try:
        api = api_factory()
    except Exception as exc:
        _write_error(
            stderr,
            "api_initialization_failed",
            str(exc),
            exception_type=type(exc).__name__,
        )
        return EXIT_EXTRACTION

    languages = args.languages or ["en"]
    documents: list[dict[str, Any]] = []
    extraction_failures = 0
    for requested_input, video_id in normalized:
        try:
            fetched = api.fetch(
                video_id,
                languages=languages,
                preserve_formatting=args.preserve_formatting,
            )
            documents.append(
                transcript_document(
                    fetched=fetched,
                    video_id=video_id,
                    requested_input=requested_input,
                    requested_languages=languages,
                    preserve_formatting=args.preserve_formatting,
                    dependency_version=dependency_version,
                )
            )
        except Exception as exc:
            extraction_failures += 1
            _write_error(
                stderr,
                "extraction_failed",
                str(exc),
                video_id=video_id,
                input=requested_input,
                exception_type=type(exc).__name__,
            )

    if not documents:
        return EXIT_EXTRACTION

    output_failed = False
    if output_dir is not None:
        extension = FORMAT_EXTENSIONS[args.format]
        for document in documents:
            video_id = document["source"]["video_id"]
            output_path = output_dir / f"{video_id}{extension}"
            try:
                _write_text(
                    output_path,
                    render_document(document, args.format),
                    overwrite=args.overwrite,
                )
            except FileExistsError:
                output_failed = True
                _write_error(
                    stderr,
                    "output_exists",
                    "output exists; use --overwrite to replace it",
                    path=str(output_path),
                    video_id=video_id,
                )
            except OSError as exc:
                output_failed = True
                _write_error(
                    stderr,
                    "output_failed",
                    str(exc),
                    path=str(output_path),
                    video_id=video_id,
                    exception_type=type(exc).__name__,
                )

    if args.to_stdout:
        try:
            stdout.write(render_batch(documents, args.format))
        except OSError as exc:
            output_failed = True
            _write_error(
                stderr,
                "output_failed",
                str(exc),
                path="stdout",
                exception_type=type(exc).__name__,
            )

    if output_failed:
        return EXIT_OUTPUT
    if extraction_failures:
        return EXIT_PARTIAL
    return EXIT_SUCCESS


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
