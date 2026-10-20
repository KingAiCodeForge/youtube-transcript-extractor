# Historical source preservation

Original public repository: KingAiCodeForge/youtube-transcript-extractor.
Source commit: `9307531637adf73d965c0f79be994f068321529a` (original main).
These paths were moved, not rewritten or duplicated. Originals also remain in
Git history and the `preserve/main-20261020` branch. None is a supported launcher.

| Original path | Current path under `source-9307531/` | Original Git object |
| --- | --- | --- |
| youtube_transcript_extractor_fixed.py | youtube_transcript_extractor_fixed.py | 92e222e997617918551fee2c0c9e2d0264be1cde |
| youtubescraper.py | youtubescraper.py | 1913cd5f0d8f27a859ac0bf8113da65de48d3871 |
| youtubevideosthatdonthaveccextractortranscript.py | youtubevideosthatdonthaveccextractortranscript.py | c4811a2fbadf09983c364d9a1b79e50b8cf28c9c |
| requirements.txt | requirements.txt | 1a828513bcdf6b87ff634ec42ca6a5152fb11965 |
| screenshot-app-ui.png | screenshot-app-ui.png | 9b8e01d1a55d1946fa8148c0ab4a0a2fc1ae0424 |
| src (gitlink) | src (gitlink) | 1d32f0b6e8002c0dc962f4113e13707ed31243be |

The gitlink has no `.gitmodules` URL. Its target content is unavailable in this
repository, was not reviewed, and was not fetched from any other repository.

Static inspection found import-time pip installation in all three prototypes;
the multi-engine prototype also supports credential-dependent speech services,
automatic downloads and output replacement. They were syntax-compiled only,
never imported or executed. GUI behavior, speech engines, paid providers and
all prototype error paths are unqualified. Historical requirements describe the
old experiments only. Do not run these as release entrypoints.
