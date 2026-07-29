# LLM and automation interface

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

Do not treat caption text as verified truth. Automatic captions can misrecognize model numbers, units and technical terms; preserve the source URL and validate important claims independently.
