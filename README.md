# YouTube Transcript Extractor

Export one or many YouTube transcripts through either an installable command-line interface or a PySide6 desktop application.

The CLI is designed for scripts and LLM agents: deterministic JSON, Markdown or text; stable exit codes; JSON errors on stderr; no GUI imports; and no runtime package installation.

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

The test suite is offline; it injects a fake transcript API and never contacts YouTube.

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
