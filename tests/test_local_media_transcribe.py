"""Offline tests for the local media transcription CLI.

Nothing here loads faster-whisper, runs ffmpeg, or touches the network. The
transcription engine and the audio extraction step are both substituted, so the
suite exercises input resolution, the document schema, rendering and the CLI
exit contract on a runner with neither dependency installed.
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_media_transcribe as lmt  # noqa: E402


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class _FakeSegment:
    def __init__(self, text, start, end):
        self.text = text
        self.start = start
        self.end = end


class _FakeInfo:
    language = "en"
    language_probability = 0.987654
    duration = 12.5


class _FakeModel:
    """Stands in for faster_whisper.WhisperModel."""

    def __init__(self, segments=None):
        self._segments = segments or [
            _FakeSegment("  First line.  ", 0.0, 4.0),
            _FakeSegment("Second line.", 4.0, 8.25),
        ]

    def transcribe(self, wav_path, **kwargs):
        self.last_kwargs = kwargs
        return iter(self._segments), _FakeInfo()


@pytest.fixture
def media_file(tmp_path):
    path = tmp_path / "briefing.mp4"
    path.write_bytes(b"not really video, but hashing does not care")
    return path


@pytest.fixture
def no_audio_extraction(monkeypatch):
    """Replace ffmpeg extraction with a no-op that creates the wav."""
    def _fake_extract(media, wav_path, ffmpeg):
        Path(wav_path).write_bytes(b"RIFF")
    monkeypatch.setattr(lmt, "extract_audio", _fake_extract)


@pytest.fixture
def ffmpeg_on_path(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg")


# --------------------------------------------------------------------------- #
# input resolution
# --------------------------------------------------------------------------- #
def test_resolve_inputs_keeps_only_supported_media(tmp_path):
    (tmp_path / "keep.mp4").write_bytes(b"a")
    (tmp_path / "keep.mp3").write_bytes(b"b")
    (tmp_path / "skip.txt").write_text("not media")
    (tmp_path / "skip.pdf").write_bytes(b"c")

    found = lmt.resolve_inputs([str(tmp_path)])

    assert sorted(p.name for p in found) == ["keep.mp3", "keep.mp4"]


def test_resolve_inputs_deduplicates_repeated_sources(media_file):
    found = lmt.resolve_inputs([str(media_file), str(media_file), str(media_file.parent)])
    assert len(found) == 1


def test_resolve_inputs_recurses_into_subdirectories(tmp_path):
    nested = tmp_path / "day1" / "session2"
    nested.mkdir(parents=True)
    (nested / "deep.wav").write_bytes(b"a")
    assert [p.name for p in lmt.resolve_inputs([str(tmp_path)])] == ["deep.wav"]


def test_resolve_inputs_is_case_insensitive_about_extensions(tmp_path):
    (tmp_path / "SHOUTING.MP4").write_bytes(b"a")
    assert len(lmt.resolve_inputs([str(tmp_path)])) == 1


def test_resolve_inputs_returns_empty_for_no_match(tmp_path):
    assert lmt.resolve_inputs([str(tmp_path / "nothing-here")]) == []


# --------------------------------------------------------------------------- #
# timestamps
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("seconds", "expected"), [
    (0.0, "00:00:00.000"),
    (1.5, "00:00:01.500"),
    (61.25, "00:01:01.250"),
    (3661.007, "01:01:01.007"),
    (-5.0, "00:00:00.000"),
])
def test_timestamp_formatting(seconds, expected):
    assert lmt._timestamp(seconds) == expected


# --------------------------------------------------------------------------- #
# document schema
# --------------------------------------------------------------------------- #
def test_transcribe_media_builds_the_documented_schema(media_file, no_audio_extraction):
    document = lmt.transcribe_media(
        media_file, model=_FakeModel(), model_name="small", device="cpu",
        compute_type="int8", language=None, beam_size=5, vad_filter=False,
        ffmpeg="ffmpeg", requested_input=str(media_file),
    )

    assert document["schema_version"] == lmt.SCHEMA_VERSION
    assert document["source"]["file_name"] == "briefing.mp4"
    assert document["source"]["file_sha256"] == lmt._sha256(media_file)
    assert document["extraction"]["model"] == "small"
    assert document["transcript"]["segment_count"] == 2
    assert document["transcript"]["is_generated"] is True

    first, second = document["transcript"]["segments"]
    assert first == {"index": 1, "text": "First line.", "start": 0.0, "duration": 4.0}
    assert second["index"] == 2
    assert second["duration"] == pytest.approx(4.25)


def test_document_carries_no_wall_clock_field(media_file, no_audio_extraction):
    """Two runs must diff cleanly, so nothing may record the current time."""
    document = lmt.transcribe_media(
        media_file, model=_FakeModel(), model_name="small", device="cpu",
        compute_type="int8", language=None, beam_size=5, vad_filter=False,
        ffmpeg="ffmpeg", requested_input=str(media_file),
    )
    flattened = json.dumps(document).lower()
    for forbidden in ("extracted_at", "generated_at", "timestamp", "created_at"):
        assert forbidden not in flattened


def test_repeated_runs_are_byte_identical(media_file, no_audio_extraction):
    kwargs = dict(
        model_name="small", device="cpu", compute_type="int8", language=None,
        beam_size=5, vad_filter=False, ffmpeg="ffmpeg", requested_input=str(media_file),
    )
    first = lmt.transcribe_media(media_file, model=_FakeModel(), **kwargs)
    second = lmt.transcribe_media(media_file, model=_FakeModel(), **kwargs)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_transcribe_passes_language_and_beam_through(media_file, no_audio_extraction):
    model = _FakeModel()
    lmt.transcribe_media(
        media_file, model=model, model_name="small", device="cpu",
        compute_type="int8", language="en", beam_size=3, vad_filter=True,
        ffmpeg="ffmpeg", requested_input=str(media_file),
    )
    assert model.last_kwargs["language"] == "en"
    assert model.last_kwargs["beam_size"] == 3
    assert model.last_kwargs["vad_filter"] is True


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
@pytest.fixture
def document(media_file, no_audio_extraction):
    return lmt.transcribe_media(
        media_file, model=_FakeModel(), model_name="small", device="cpu",
        compute_type="int8", language=None, beam_size=5, vad_filter=False,
        ffmpeg="ffmpeg", requested_input=str(media_file),
    )


def test_render_json_round_trips(document):
    assert json.loads(lmt.render_document(document, "json")) == document


def test_render_markdown_includes_hash_and_timestamps(document):
    rendered = lmt.render_document(document, "markdown")
    assert rendered.startswith("# Local Media Transcript: briefing")
    assert document["source"]["file_sha256"] in rendered
    assert "[00:00:00.000] First line." in rendered
    assert rendered.endswith("\n")


def test_render_text_includes_the_provenance_header(document):
    rendered = lmt.render_document(document, "text")
    assert "LOCAL MEDIA TRANSCRIPT" in rendered
    assert f"sha256: {document['source']['file_sha256']}" in rendered
    assert "[00:00:04.000] Second line." in rendered


def test_render_rejects_an_unknown_format(document):
    with pytest.raises(ValueError, match="unsupported output format"):
        lmt.render_document(document, "yaml")


def test_render_batch_markdown_demotes_headings(document):
    rendered = lmt.render_batch([document, document], "markdown")
    assert "## Local Media Transcript: briefing" in rendered
    assert "\n---\n" in rendered


def test_render_batch_json_is_a_list(document):
    assert len(json.loads(lmt.render_batch([document, document], "json"))) == 2


# --------------------------------------------------------------------------- #
# CLI exit contract
# --------------------------------------------------------------------------- #
def test_cli_requires_an_output_destination(capsys):
    assert lmt.run_cli(["anything.mp4"]) == lmt.EXIT_USAGE
    assert "output destination" in capsys.readouterr().err


def test_cli_reports_missing_ffmpeg_as_a_dependency_failure(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    code = lmt.run_cli(["anything.mp4", "--stdout", "--ffmpeg", "no-such-ffmpeg-binary"])
    assert code == lmt.EXIT_DEPENDENCY
    assert "ffmpeg not found" in capsys.readouterr().err


def test_cli_reports_no_matching_media_as_usage(tmp_path, capsys, ffmpeg_on_path):
    code = lmt.run_cli([str(tmp_path / "absent.mp4"), "--stdout"])
    assert code == lmt.EXIT_USAGE
    assert "no supported media files" in capsys.readouterr().err


def test_cli_reports_a_missing_engine_as_a_dependency_failure(
    media_file, capsys, ffmpeg_on_path, monkeypatch
):
    def _no_engine(*args, **kwargs):
        raise ModuleNotFoundError("faster-whisper is not installed.")
    monkeypatch.setattr(lmt, "load_model", _no_engine)

    code = lmt.run_cli([str(media_file), "--stdout"])
    assert code == lmt.EXIT_DEPENDENCY
    assert "faster-whisper is not installed" in capsys.readouterr().err


def test_cli_writes_one_file_per_input(tmp_path, media_file, ffmpeg_on_path,
                                       no_audio_extraction, monkeypatch):
    monkeypatch.setattr(lmt, "load_model", lambda *a, **k: _FakeModel())
    out_dir = tmp_path / "out"

    code = lmt.run_cli([str(media_file), "--output-dir", str(out_dir), "--format", "markdown"])

    assert code == lmt.EXIT_OK
    written = out_dir / "briefing.md"
    assert written.is_file()
    assert "First line." in written.read_text(encoding="utf-8")


def test_cli_keeps_going_after_one_file_fails(tmp_path, ffmpeg_on_path, monkeypatch, capsys):
    good = tmp_path / "good.mp4"
    bad = tmp_path / "bad.mp4"
    good.write_bytes(b"a")
    bad.write_bytes(b"b")

    def _fail_on_bad(media, wav_path, ffmpeg):
        if Path(media).name == "bad.mp4":
            raise RuntimeError("ffmpeg failed for bad.mp4")
        Path(wav_path).write_bytes(b"RIFF")

    monkeypatch.setattr(lmt, "extract_audio", _fail_on_bad)
    monkeypatch.setattr(lmt, "load_model", lambda *a, **k: _FakeModel())

    code = lmt.run_cli([str(tmp_path), "--output-dir", str(tmp_path / "out")])

    assert code == lmt.EXIT_RUNTIME
    assert (tmp_path / "out" / "good.json").is_file()
    assert "ffmpeg failed for bad.mp4" in capsys.readouterr().err


def test_cli_version_flag_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        lmt.run_cli(["--version"])
    assert excinfo.value.code == 0
    assert lmt.__version__ in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# device selection
# --------------------------------------------------------------------------- #
def test_explicit_device_is_never_overridden():
    assert lmt._resolve_device("cpu") == "cpu"
    assert lmt._resolve_device("cuda") == "cuda"


def test_device_falls_back_to_cpu_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert lmt._resolve_device("auto") == "cpu"


def test_compute_type_defaults_per_device():
    assert lmt._pick_compute_type("cuda", "auto") == "int8_float16"
    assert lmt._pick_compute_type("cpu", "auto") == "int8"
    assert lmt._pick_compute_type("cpu", "float32") == "float32"
