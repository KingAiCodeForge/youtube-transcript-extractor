# YouTube Transcript Extractor

Export one or many transcripts through installable command-line interfaces or a PySide6 desktop application.

The project ships two complementary CLIs that emit the **same transcript schema**, so their output drops into one comparison or diff workflow:

| Command | Source | Engine |
| --- | --- | --- |
| `youtube-transcript` | captions that already exist on YouTube | `youtube-transcript-api` |
| `local-media-transcribe` | a local audio or video file with no captions | `faster-whisper` and `ffmpeg` |

Both are designed for scripts and LLM agents: deterministic JSON, Markdown or text; stable exit codes; errors on stderr; no GUI imports; and no runtime package installation.

## Command-line installation

Python 3.9 or newer is required.

```bash
git clone https://github.com/KingAiCodeForge/youtube-transcript-extractor.git
cd youtube-transcript-extractor
python -m pip install -e .
```

## Command-line usage

```bash
youtube-transcript \
  "https://www.youtube.com/watch?v=VIDEO_ID" \
  --format json \
  --stdout
```

Convert several videos to individual Markdown files:

```bash
youtube-transcript URL_1 URL_2 URL_3 \
  --format markdown \
  --output-dir ./output
```

Existing files are never replaced unless `--overwrite` is supplied.

Supported input forms include video IDs and `watch`, `youtu.be`, `shorts`, `live`, and `embed` URLs. Repeat `--language CODE` to set caption-language preference order.

```bash
youtube-transcript --help
youtube-transcript --version
```

See [LLM_USAGE.md](LLM_USAGE.md) for the JSON contract and exit codes.

## Local media transcription

`local-media-transcribe` covers the case the YouTube tools cannot: a local file that has no captions at all, such as a downloaded briefing, a recorded call, or an MP4 straight off a camera. It extracts audio with `ffmpeg`, transcribes with `faster-whisper`, and emits the same schema as `youtube-transcript`.

It needs two extra pieces that the YouTube path does not:

```bash
python -m pip install -e ".[local-media]"   # faster-whisper
# and ffmpeg must be on PATH, or pass --ffmpeg /path/to/ffmpeg
```

```bash
# one file to Markdown
local-media-transcribe briefing.mp4 --format markdown --output-dir out

# several files at once, one <stem>.json each
local-media-transcribe brief1.mp4 brief2.mp4 call.m4a --output-dir out

# a whole folder, GPU, higher-accuracy model
local-media-transcribe ./recordings --model medium --device cuda --output-dir out

# quick look on screen
local-media-transcribe call.m4a --stdout --format text
```

Directories are searched recursively and duplicate inputs are transcribed once. `--device auto` selects CUDA when torch reports it and otherwise falls back to CPU, including when a CUDA model fails to load for want of cuDNN or cuBLAS.

Every document records the source file's SHA-256, the model, device, compute type, beam size and the resolved `faster-whisper` version, so a transcript can be traced back to the exact file and settings that produced it. As with the YouTube path there is no wall-clock field, so re-running an unchanged file produces a byte-identical document.

A machine transcript of speech is a draft. Whisper misrecognises names, part numbers, callsigns and units, and it can hallucinate text over silence or music. Treat the output as something a person reviews, not as a record.

## Desktop application

```bash
python -m pip install -e ".[gui]"
python youtube_transcript_extractor_fixed.py
```

Legacy tools remain available:

- `youtube_transcript_extractor_fixed.py`: current PySide6 transcript GUI.
- `youtubescraper.py`: original GUI application.
- `youtubevideosthatdonthaveccextractortranscript.py`: metadata/fallback utility.

## Development

The test suite is offline. It injects a fake transcript API and never contacts YouTube, and it substitutes both the transcription engine and the audio extraction step, so the local-media tests run on a machine with neither `faster-whisper` nor `ffmpeg` installed.

```bash
python -m pip install -e ".[test]"
pytest tests
```

## Evidence and privacy

Generated transcripts belong in `output/` or `exports/`, which are ignored by Git. Do not commit private downloads or research corpora.

A transcript is evidence of what captions contain, not proof that a technical claim is correct. Automatic captions can misrecognize product numbers, voltages, frequencies, and units.

## License

The project is described as MIT-licensed, but the repository does not yet contain a root `LICENSE` file. Treat licensing as unresolved until the copyright holder adds one.

## Credits

- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api)
- [PySide6](https://doc.qt.io/qtforpython-6/)
