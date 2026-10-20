"""Safety boundaries tested without network, ASR models or GPU access."""
from io import StringIO
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import local_media_transcribe as lmt
import youtube_transcript_cli as yt


def test_local_outputs_reject_collision_before_model_loading(tmp_path, monkeypatch):
    first = tmp_path / "one" / "same.mp4"
    second = tmp_path / "two" / "same.wav"
    for media in (first, second):
        media.parent.mkdir()
        media.write_bytes(b"original")
    monkeypatch.setattr(lmt.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(lmt, "load_model", lambda *a: pytest.fail("loaded a model"))
    out = tmp_path / "output"
    assert lmt.run_cli([str(first), str(second), "--output-dir", str(out)]) == lmt.EXIT_USAGE
    assert not out.exists()
    assert first.read_bytes() == second.read_bytes() == b"original"


def test_local_existing_output_fails_without_mutation(tmp_path):
    media = tmp_path / "video.mp4"
    media.write_bytes(b"source")
    output = tmp_path / "video.json"
    output.write_bytes(b"previous transcript")
    with pytest.raises(ValueError, match="output exists"):
        lmt.plan_outputs([media], tmp_path, "json")
    assert output.read_bytes() == b"previous transcript"


def test_local_output_hardlink_to_source_is_rejected_even_with_overwrite(tmp_path):
    media = tmp_path / "video.mp4"
    media.write_bytes(b"source")
    output = tmp_path / "video.json"
    os.link(media, output)
    with pytest.raises(ValueError, match="aliases an input"):
        lmt.plan_outputs([media], tmp_path, "json", overwrite=True)
    assert media.read_bytes() == b"source"


def test_local_model_failure_is_controlled(tmp_path, monkeypatch):
    media = tmp_path / "video.mp4"
    media.write_bytes(b"source")
    monkeypatch.setattr(lmt.shutil, "which", lambda _: "ffmpeg")

    def fail(*a):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(lmt, "load_model", fail)
    errors = StringIO()
    assert lmt.run_cli([str(media), "--stdout"], stderr=errors) == lmt.EXIT_RUNTIME
    assert "model loading failed" in errors.getvalue()


def test_ctranslate2_without_cuda_is_authoritative(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 0))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    assert lmt._resolve_device("auto") == "cpu"


@pytest.mark.parametrize("module", [lmt, yt])
def test_output_writer_exclusive_and_explicit_replacement(module, tmp_path):
    output = tmp_path / "spaced ünicode.json"
    output.write_text("previous", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module._write_text(output, "new")
    assert output.read_text(encoding="utf-8") == "previous"
    module._write_text(output, "new", overwrite=True)
    assert output.read_text(encoding="utf-8") == "new"
    assert not list(tmp_path.glob("*.tmp"))


def test_youtube_missing_dependency_does_not_create_output_directory(tmp_path, monkeypatch):
    def unavailable():
        raise yt.DependencyUnavailable("test")

    monkeypatch.setattr(yt, "_load_api_factory", unavailable)
    output = tmp_path / "new output ü"
    assert yt.run_cli(["3OhT_nAgJ2I", "--output-dir", str(output)], stderr=StringIO()) == yt.EXIT_DEPENDENCY
    assert not output.exists()


def test_local_invalid_beam_size_does_not_touch_output(tmp_path):
    output = tmp_path / "new"
    assert lmt.run_cli(["missing.mp4", "--output-dir", str(output), "--beam-size", "0"]) == lmt.EXIT_USAGE
    assert not output.exists()


def test_local_output_failure_is_controlled(tmp_path, monkeypatch):
    media = tmp_path / "vid.mp4"
    media.write_bytes(b"source")
    monkeypatch.setattr(lmt.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(lmt, "load_model", lambda *a: object())
    monkeypatch.setattr(lmt, "transcribe_media", lambda *a, **k: {"transcript": {"segment_count": 0}})
    monkeypatch.setattr(lmt, "render_document", lambda *a: "transcript")

    def fail(*a, **k):
        raise PermissionError("blocked")

    monkeypatch.setattr(lmt, "_write_text", fail)
    errors = StringIO()
    assert lmt.run_cli([str(media), "--output-dir", str(tmp_path / "out")], stderr=errors) == lmt.EXIT_RUNTIME
    assert "output failed" in errors.getvalue()
    assert media.read_bytes() == b"source"


@pytest.mark.parametrize("module", [lmt, yt])
def test_explicit_replacement_does_not_truncate_an_aliased_original(module, tmp_path):
    original = tmp_path / "original archive.txt"
    original.write_bytes(b"original content")
    output = tmp_path / "transcript.txt"
    os.link(original, output)
    module._write_text(output, "replacement", overwrite=True)
    assert original.read_bytes() == b"original content"
    assert output.read_text(encoding="utf-8") == "replacement"


def test_local_stdout_failure_is_controlled(tmp_path, monkeypatch):
    media = tmp_path / "vid.mp4"
    media.write_bytes(b"source")
    monkeypatch.setattr(lmt.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(lmt, "load_model", lambda *a: object())
    monkeypatch.setattr(lmt, "transcribe_media", lambda *a, **k: {})
    monkeypatch.setattr(lmt, "render_batch", lambda *a: "transcript")

    class BrokenOutput(StringIO):
        def write(self, text):
            raise BrokenPipeError("pipe closed")

    errors = StringIO()
    assert lmt.run_cli([str(media), "--stdout"], stdout=BrokenOutput(), stderr=errors) == lmt.EXIT_RUNTIME
    assert "stdout failed" in errors.getvalue()
