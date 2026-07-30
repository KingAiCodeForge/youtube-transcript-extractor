# LLM and automation interface

Two CLIs are available for non-interactive work. Neither has GUI imports, neither installs dependencies at runtime, both write successful data to stdout and diagnostics to stderr, and both return documented exit codes.

| Command | Use it for |
| --- | --- |
| `youtube-transcript` | captions that already exist on YouTube |
| `local-media-transcribe` | a local audio or video file with no captions |

**Their exit codes are not the same.** Check the correct table below before branching on a return value.

## youtube-transcript

Use the `youtube-transcript` CLI for non-interactive work. It has no GUI imports, never installs dependencies at runtime, writes successful data to stdout, writes one-line JSON errors to stderr, and returns documented exit codes.

## Install and discover

```bash
python -m pip install -e .
youtube-transcript --help
youtube-transcript --version
```

## Deterministic JSON

```bash
youtube-transcript \
  "https://www.youtube.com/watch?v=VIDEO_ID" \
  --format json \
  --stdout
```

## Batch files

```bash
youtube-transcript URL_1 URL_2 URL_3 \
  --format markdown \
  --output-dir ./output
```

At least one destination is required: `--stdout`, `--output-dir`, or both. Duplicate video IDs are fetched once. Repeat `--language CODE` to set language preference order.

File output uses exclusive creation by default. If a target exists, it is left unchanged and the command returns code 5 with an `output_exists` JSON error. Supply `--overwrite` only when replacement is intentional.

## JSON contract

Schema `1.0` contains `generator`, `source`, `extraction`, and `transcript`. Each ordered segment contains `index`, `text`, `start`, and `duration`. There is deliberately no wall-clock extraction timestamp, so unchanged source data produces clean diffs.

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | Every requested transcript succeeded. |
| 1 | Partial success; successful data was emitted and failures are on stderr. |
| 2 | Invalid command or source input. |
| 3 | Extraction failed for every requested video or API initialization failed. |
| 4 | `youtube-transcript-api` is unavailable. |
| 5 | Output could not be written, including a no-overwrite conflict. |

## local-media-transcribe

Transcribes local files that have no captions. Requires `faster-whisper` (`pip install -e ".[local-media]"`) and `ffmpeg` on `PATH`.

```bash
local-media-transcribe ./recordings --format json --output-dir ./out
local-media-transcribe call.m4a --stdout --format text
```

At least one destination is required: `--stdout`, `--output-dir`, or both. Inputs may be files, directories (searched recursively) or globs, including absolute globs. Unsupported extensions are ignored and duplicate inputs are transcribed once.

Progress lines go to stderr and never contaminate stdout, so `--stdout` output is safe to pipe directly into a parser.

### Exit codes

| Code | Meaning |
|---:|---|
| 0 | Every input transcribed. |
| 1 | Invalid command or input: no destination given, or nothing matched. |
| 2 | A dependency is unavailable: `ffmpeg` not found, or `faster-whisper` not installed. |
| 3 | At least one file failed to transcribe. Successful files were still written. |

A batch continues after an individual failure. Code 3 therefore means partial success, and the files that succeeded are on disk.

### Provenance fields

Alongside the shared schema, `source.file_sha256` records the SHA-256 of the input file and the `extraction` block records `model`, `device`, `compute_type`, `beam_size`, `vad_filter` and the resolved `library_version`. A transcript can be tied back to the exact file and settings that produced it, and an unchanged input re-runs to a byte-identical document.

## Both commands

Do not treat transcript text as verified truth. Automatic captions and machine speech recognition both misrecognize model numbers, units, names and technical terms, and speech recognition can additionally hallucinate text over silence or music. Preserve the source URL or file hash and validate important claims independently. Output is a draft for a person to review, not a record.
