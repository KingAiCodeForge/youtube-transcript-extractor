# 1.1.0rc1 review scope

Original main: `9307531637adf73d965c0f79be994f068321529a`.
Included PR #2 head: `1ff25996796457cb850989ca08f48a13a08d43b5`.
The integration descends from that exact head; no PR content is discarded.
Original remote branches remain. Local original refs and an all-refs bundle
were saved before changes; public preservation branches retain main and PR #2.

## Script and module ledger

| Path | Purpose and interface | Review/change | Qualification |
| --- | --- | --- | --- |
| youtube_transcript_cli.py | captions API; URL/ID inputs; JSON/Markdown/text CLI | inspected full stable module; explicit overwrite now replaces atomically; missing dependency does not create output directories; malformed Unicode is escaped in errors and output encoding failures are controlled | offline API fakes; URL forms, batch/partial failures, missing dependency, output conflicts, formatting, malformed-Unicode failure without creating a file; installed CLI help/version and invalid-input smoke |
| local_media_transcribe.py | media/glob/directory CLI; ffmpeg then faster-whisper | inspected full stable module; PR runtime/provenance and hotwords included; CTranslate2 zero devices authoritative; collision/alias preflight; exclusive outputs; atomic explicit replacement; controlled model/write/stdout failure; beam validation | offline model/audio fakes; provenance, runtime fallback, hotwords, hashes; real ffmpeg extraction of synthetic WAV; installed CLI and missing-file/dependency smoke |
| install.bat | Windows source installer | interpreter-bound pip; checkout-relative path; propagates failure | invoked from an unrelated directory; clean wheel install separately qualified |
| install.sh | POSIX source installer | script-relative path, quoted interpreter/path, fail-fast shell | syntax and invocation tested under Ubuntu WSL; Python 3.12 install/run qualification |
| tests/test_youtube_transcript_cli.py | offline caption regressions | inspected; retained PR mobile URL case | pytest |
| tests/test_local_media_transcribe.py | offline local-media regressions | inspected; retained PR device/hotword cases | pytest |
| tests/test_release_safety.py | output and failure safety regressions | new, user-owned tests | pytest; hardlink alias, collision, default preservation, explicit replacement, Unicode/space paths, controlled failures |
| legacy/source-9307531/youtube_transcript_extractor_fixed.py | historical PySide6 captions GUI | static import/dependency, output and entrypoint inventory; unchanged source-labelled preservation | syntax only; excluded historical prototype, no runtime qualification |
| legacy/source-9307531/youtubescraper.py | historical GUI captions/audio fallback | static import/dependency, engine, output and entrypoint inventory; unchanged preservation | syntax only; excluded historical prototype, no runtime qualification |
| legacy/source-9307531/youtubevideosthatdonthaveccextractortranscript.py | historical GUI multi-engine transcription/analysis | static import/dependency, credential-provider, output and entrypoint inventory; unchanged preservation | syntax only; excluded historical prototype, engines/GUI/error paths unqualified |
| legacy/source-9307531/src | unresolved historical gitlink | exact ref recorded; no configured URL, content not fetched | not reviewed; excluded from artifacts |

The stable modules are qualified for offline interface and safety behavior, not
live YouTube reliability, speech accuracy, CUDA execution or every external
dependency combination. Python 3.9-3.14 remains the CI declaration; only the
Python/platform versions listed in the release receipt were run locally.
No hardware controls, private inputs, paid services or package registries used.

Local results on 2026-10-20: Windows Python 3.13.5: 73 offline tests passed,
critical flake8 checks returned zero; Ubuntu 24.04 WSL Python 3.12.3: 73 offline
tests passed. Both clean environments installed the package, ran help/version
and missing-input/dependency checks from outside the checkout with space/Unicode
fixture paths. Both installers were invoked from unrelated working directories.
Windows additionally ran real ffmpeg on a generated speech-free WAV and verified
16 kHz mono output and an unchanged source SHA-256. Linux ffmpeg was unavailable;
Linux extraction was covered by injected tests only. macOS and other Python
versions have not been run in this local qualification.

## Exposure and artifact review

`.gitignore` retains output, export, research, secret, local-config, virtualenv
and build exclusions. `.git/info/exclude` contains only stock comments; no
repository or global ignore settings were changed. No ignored private source
was imported. New tests, documentation and manifest are explicit owned additions.
Tracked paths were inspected for machine-specific paths, credential literals,
generated transcripts and personal/customer inputs. Historical source is public
source only; generated test WAVs and qualification environments stay outside Git.

Wheel and sdist contain the two stable modules and package metadata; sdist also
contains stable tests, source installers, docs and the opt-in vocabulary example.
`MANIFEST.in` explicitly prunes historical source. A separate GitHub source archive
retains the public preservation manifest and prototypes. Final artifact members
and SHA-256 hashes are recorded in the release receipt.

No root LICENSE exists; historical README's MIT claim remains unresolved. This
release does not invent a copyright grant. It is a prerelease CLI qualification,
not a general stable-readiness assertion for the historical codebase.
