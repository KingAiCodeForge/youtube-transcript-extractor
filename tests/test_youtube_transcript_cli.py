from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace

import pytest

import youtube_transcript_cli as cli


VIDEO_A = "3OhT_nAgJ2I"
VIDEO_B = "7hesG08NcPk"


class FakeApi:
    def __init__(self, failing_ids=()):
        self.failing_ids = set(failing_ids)
        self.calls = []

    def fetch(self, video_id, *, languages, preserve_formatting):
        self.calls.append((video_id, languages, preserve_formatting))
        if video_id in self.failing_ids:
            raise RuntimeError("captions unavailable")
        return SimpleNamespace(
            language="English",
            language_code="en",
            is_generated=True,
            snippets=[
                SimpleNamespace(text="First line", start=0.0, duration=1.25),
                SimpleNamespace(text="Second line", start=61.5, duration=2.0),
            ],
        )


@pytest.mark.parametrize(
    "value",
    [
        VIDEO_A,
        f"https://www.youtube.com/watch?v={VIDEO_A}",
        f"https://youtu.be/{VIDEO_A}",
        f"https://www.youtube.com/shorts/{VIDEO_A}",
        f"https://www.youtube.com/live/{VIDEO_A}?feature=share",
        f"https://www.youtube-nocookie.com/embed/{VIDEO_A}",
    ],
)
def test_extract_video_id_supported_forms(value):
    assert cli.extract_video_id(value) == VIDEO_A


def test_extract_video_id_rejects_ambiguous_input():
    with pytest.raises(ValueError, match="not a supported YouTube URL"):
        cli.extract_video_id("not a video id")


def test_json_stdout_is_stable_and_machine_readable():
    stdout = StringIO()
    stderr = StringIO()
    fake_api = FakeApi()
    result = cli.run_cli(
        [VIDEO_A, "--stdout", "--format", "json"],
        stdout=stdout,
        stderr=stderr,
        api_factory=lambda: fake_api,
        dependency_version="1.2.3-test",
    )

    assert result == cli.EXIT_SUCCESS
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["schema_version"] == cli.SCHEMA_VERSION
    assert payload["video_count"] == 1
    video = payload["videos"][0]
    assert video["source"]["video_id"] == VIDEO_A
    assert video["transcript"]["segment_count"] == 2
    assert video["transcript"]["segments"][1]["start"] == 61.5
    assert "extracted_at" not in stdout.getvalue()
    assert fake_api.calls == [(VIDEO_A, ["en"], False)]


def test_markdown_output_uses_timestamped_segments(tmp_path):
    result = cli.run_cli(
        [VIDEO_A, "--output-dir", str(tmp_path), "--format", "markdown"],
        stdout=StringIO(),
        stderr=StringIO(),
        api_factory=FakeApi,
        dependency_version="test",
    )

    assert result == cli.EXIT_SUCCESS
    output = (tmp_path / f"{VIDEO_A}.md").read_text(encoding="utf-8")
    assert "[00:00:00.000] First line" in output
    assert "[00:01:01.500] Second line" in output


def test_existing_output_is_not_overwritten_by_default(tmp_path):
    output_path = tmp_path / f"{VIDEO_A}.md"
    output_path.write_text("sentinel\n", encoding="utf-8")
    stderr = StringIO()

    result = cli.run_cli(
        [VIDEO_A, "--output-dir", str(tmp_path), "--format", "markdown"],
        stdout=StringIO(),
        stderr=stderr,
        api_factory=FakeApi,
        dependency_version="test",
    )

    assert result == cli.EXIT_OUTPUT
    assert output_path.read_text(encoding="utf-8") == "sentinel\n"
    error = json.loads(stderr.getvalue())["error"]
    assert error["code"] == "output_exists"


def test_overwrite_replaces_existing_output_when_explicit(tmp_path):
    output_path = tmp_path / f"{VIDEO_A}.md"
    output_path.write_text("sentinel\n", encoding="utf-8")

    result = cli.run_cli(
        [
            VIDEO_A,
            "--output-dir",
            str(tmp_path),
            "--format",
            "markdown",
            "--overwrite",
        ],
        stdout=StringIO(),
        stderr=StringIO(),
        api_factory=FakeApi,
        dependency_version="test",
    )

    assert result == cli.EXIT_SUCCESS
    assert "First line" in output_path.read_text(encoding="utf-8")


def test_parser_errors_are_json():
    stderr = StringIO()
    result = cli.run_cli(
        [VIDEO_A, "--format", "unsupported", "--stdout"],
        stdout=StringIO(),
        stderr=stderr,
        api_factory=FakeApi,
    )

    assert result == cli.EXIT_USAGE
    error = json.loads(stderr.getvalue())["error"]
    assert error["code"] == "invalid_arguments"


def test_partial_batch_returns_document_and_json_error():
    stdout = StringIO()
    stderr = StringIO()
    fake_api = FakeApi(failing_ids={VIDEO_B})
    result = cli.run_cli(
        [VIDEO_A, VIDEO_B, "--stdout"],
        stdout=stdout,
        stderr=stderr,
        api_factory=lambda: fake_api,
        dependency_version="test",
    )

    assert result == cli.EXIT_PARTIAL
    assert json.loads(stdout.getvalue())["video_count"] == 1
    error = json.loads(stderr.getvalue())["error"]
    assert error["code"] == "extraction_failed"
    assert error["video_id"] == VIDEO_B


def test_invalid_input_returns_usage_error_as_json():
    stderr = StringIO()
    result = cli.run_cli(
        ["not a video id", "--stdout"],
        stdout=StringIO(),
        stderr=stderr,
        api_factory=FakeApi,
    )

    assert result == cli.EXIT_USAGE
    assert json.loads(stderr.getvalue())["error"]["code"] == "invalid_input"


def test_duplicate_video_ids_are_fetched_once():
    fake_api = FakeApi()
    result = cli.run_cli(
        [VIDEO_A, f"https://youtu.be/{VIDEO_A}", "--stdout"],
        stdout=StringIO(),
        stderr=StringIO(),
        api_factory=lambda: fake_api,
        dependency_version="test",
    )

    assert result == cli.EXIT_SUCCESS
    assert len(fake_api.calls) == 1


def test_output_destination_is_required():
    stderr = StringIO()

    result = cli.run_cli(
        [VIDEO_A],
        stdout=StringIO(),
        stderr=stderr,
        api_factory=FakeApi,
    )

    assert result == cli.EXIT_USAGE
    error = json.loads(stderr.getvalue())["error"]
    assert error["code"] == "invalid_arguments"
    assert "output destination" in error["message"]


def test_missing_dependency_has_stable_error(monkeypatch):
    def unavailable():
        raise cli.DependencyUnavailable("dependency intentionally unavailable")

    monkeypatch.setattr(cli, "_load_api_factory", unavailable)
    stderr = StringIO()

    result = cli.run_cli(
        [VIDEO_A, "--stdout"],
        stdout=StringIO(),
        stderr=stderr,
    )

    assert result == cli.EXIT_DEPENDENCY
    error = json.loads(stderr.getvalue())["error"]
    assert error["code"] == "dependency_missing"
    assert "intentionally unavailable" in error["message"]


def test_total_extraction_failure_returns_extraction_code():
    stderr = StringIO()
    fake_api = FakeApi(failing_ids={VIDEO_A})

    result = cli.run_cli(
        [VIDEO_A, "--stdout"],
        stdout=StringIO(),
        stderr=stderr,
        api_factory=lambda: fake_api,
        dependency_version="test",
    )

    assert result == cli.EXIT_EXTRACTION
    error = json.loads(stderr.getvalue())["error"]
    assert error["code"] == "extraction_failed"
    assert error["video_id"] == VIDEO_A


def test_output_directory_must_be_a_directory(tmp_path):
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("sentinel\n", encoding="utf-8")
    stderr = StringIO()

    result = cli.run_cli(
        [VIDEO_A, "--output-dir", str(output_file)],
        stdout=StringIO(),
        stderr=stderr,
        api_factory=FakeApi,
    )

    assert result == cli.EXIT_OUTPUT
    error = json.loads(stderr.getvalue())["error"]
    assert error["code"] == "output_failed"
    assert error["path"] == str(output_file)
