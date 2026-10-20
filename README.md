# YouTube Transcript Extractor

Export one or many transcripts through installable command-line interfaces.
Version 1.1.0rc1 qualifies offline CLI safety and packaging; live YouTube service
availability, ASR accuracy and GPU execution have not been qualified for this release.

The project ships two complementary CLIs with shared transcript segment fields:

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
python -m pip install .
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

Both commands preserve existing output files unless `--overwrite` is supplied.
Local-media batches reject colliding output names and output/input aliases before
loading a model. Give same-stem media separate output directories.

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

# bias technical vocabulary without rewriting the transcript afterwards
local-media-transcribe webinar.mp4 \
  --model medium --device auto --language en --vad-filter \
  --hotwords-file vocab/ecu_tuning_core.txt \
  --format markdown --output-dir out
```

Repeat `--hotword TERM` for ad-hoc vocabulary or use `--hotwords-file` for one
term/phrase per line. Hotwords are passed into faster-whisper's decoder and the
resolved list is stored in the transcript's extraction provenance. They are a
recognition hint, not a factual correction layer; overly broad or incorrect
hotwords can bias ASR output.

Directories are searched recursively and duplicate inputs are transcribed once. `--device auto` asks CTranslate2—the runtime used by `faster-whisper`—whether a CUDA device is available, with a PyTorch probe retained only as a compatibility fallback. If CUDA model loading still fails (for example because required GPU runtime libraries are unavailable), the command retries on CPU.

Every document records the source file's SHA-256, the model, **actual runtime device**, compute type, beam size and the resolved `faster-whisper` version. A CUDA-to-CPU fallback is recorded as CPU. There is no wall-clock field: rendering identical segment data and provenance is byte-stable. ASR output may still vary by model, runtime and hardware; identical input bytes do not guarantee identical recognized text. Local-media JSON batches are arrays; YouTube JSON batches use a `videos` envelope.

A machine transcript of speech is a draft. Whisper misrecognises names, part numbers, callsigns and units, and it can hallucinate text over silence or music. Treat the output as something a person reviews, not as a record.

## Preserved prototypes

The original GUI and multi-engine experiments are preserved unchanged under
[`legacy/source-9307531/`](legacy/source-9307531/), with exact source blobs and
limitations in [`legacy/MANIFEST.md`](legacy/MANIFEST.md). They can install
dependencies at import time and contact external speech services. They are
unsupported, were not executed, and are excluded from wheel/sdist artifacts.
The historical `src` gitlink has no configured submodule URL; it is recorded
without fetching unknown content. See [RELEASE_REVIEW.md](RELEASE_REVIEW.md).

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
