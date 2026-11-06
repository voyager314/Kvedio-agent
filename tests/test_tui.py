from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kvedio_agent.interface.commands import command_label, command_suggestions, parse_command
from kvedio_agent.interface.tui import (
    WorkbenchApp,
    event_lines,
    evidence_lines,
    file_path_suggestions,
    job_label,
    parse_file_references,
    referenced_text,
    usage_lines,
)
from kvedio_agent.pipeline.analyze import AnalysisResult
from kvedio_agent.platform.storage import Store


def test_workbench_formatters_render_durable_state(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    source = tmp_path / "video.mp4"
    source.write_bytes(b"source")
    source_id = store.register_source(source, 6, source.stat().st_mtime_ns, "digest")
    job_id = store.create_job(source_id, "objective", {})
    store.set_job_status(job_id, "running", "transcribe")
    store.add_event(job_id, "stage_started", "Transcription started")
    store.add_evidence(job_id, "transcript", 1000, 2000, {}, "Launch approved")

    job = store.job_history()[0]
    assert job_id[:8] in job_label(job)
    assert "stage_started" in event_lines(store.job_events(job_id))
    assert "Launch approved" in evidence_lines(store.job_evidence(job_id))
    assert "Estimated only" in usage_lines(store.usage_summary(job_id))


def test_workbench_app_composes_without_network_calls(tmp_path: Path) -> None:
    app = WorkbenchApp(Store(tmp_path / "workspace"))
    assert app.selected_job_id is None
    assert app.CSS


def test_at_references_resolve_multiple_paths_with_spaces(tmp_path: Path) -> None:
    first = tmp_path / "meeting video.mp4"
    second = tmp_path / "notes file.txt"
    first.write_bytes(b"video")
    second.write_text("agenda", encoding="utf-8")

    paths, objective = parse_file_references(
        f'Analyze @"{first}" with context @"{second}"'
    )

    assert paths == [first.resolve(), second.resolve()]
    assert objective == "Analyze with context"


def test_at_reference_reports_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_file_references(f'Analyze @"{tmp_path / "missing file.txt"}"')


def test_at_reference_rejects_unclosed_quote_and_directory(tmp_path: Path) -> None:
    directory = tmp_path / "folder"
    directory.mkdir()
    with pytest.raises(ValueError, match="Close the quote"):
        parse_file_references('Analyze @"unfinished path')
    with pytest.raises(ValueError, match="existing file"):
        parse_file_references(f"Analyze @{directory}")


def test_command_parser_resolves_alias_and_arguments() -> None:
    assert parse_command("/quit") == ("quit", "")
    assert parse_command("/analyze  find the decision") == ("analyze", "find the decision")
    assert parse_command("find the decision") is None
    assert [item.name for item in command_suggestions("/mo")] == ["model"]
    assert "/quit" in command_label(command_suggestions("/qu")[0])


def test_file_suggestions_and_text_context_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    video = tmp_path / "meeting.mp4"
    notes = tmp_path / "meeting notes.txt"
    video.write_bytes(b"video")
    notes.write_text("agenda", encoding="utf-8")

    suggestions = file_path_suggestions('@"meeting n')
    assert suggestions == [("meeting notes.txt", str(notes))]
    assert referenced_text([video, notes], video).endswith("--- " + str(notes) + " ---\nagenda")


def test_binary_reference_fails_without_returning_file_contents(tmp_path: Path) -> None:
    binary = tmp_path / "document.bin"
    binary.write_bytes(b"\xff\x00secret")
    with pytest.raises(ValueError, match="UTF-8") as raised:
        referenced_text([binary])
    assert "secret" not in str(raised.value)


def test_workbench_uses_one_prompt_and_command_suggestions(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = WorkbenchApp(Store(tmp_path / "workspace"))
        async with app.run_test() as pilot:
            assert len(app.query("Input")) == 1
            await pilot.press("/")
            assert app.query_one("#prompt").value == "/"
            assert app.query_one("#suggestions").has_class("visible")
            await pilot.press("escape")
            assert not app.query_one("#suggestions").has_class("visible")
            app.query_one("#prompt").value = "/help"
            await pilot.pause()
            await pilot.press("enter")
            assert "/model" in str(app.query_one("#events").render())

    asyncio.run(exercise())


def test_natural_prompt_routes_to_follow_up_for_selected_job(tmp_path: Path) -> None:
    async def exercise() -> None:
        store = Store(tmp_path / "workspace")
        source = tmp_path / "video.mp4"
        source.write_bytes(b"video")
        source_id = store.register_source(source, 5, source.stat().st_mtime_ns, "digest")
        job_id = store.create_job(source_id, "objective", {})
        questions: list[str] = []

        def follow_up(selected: str, question: str) -> AnalysisResult:
            assert selected == job_id
            questions.append(question)
            return AnalysisResult(
                objective=question,
                plan=[],
                claims=[],
                uncertainties=[],
                quality_assessment="No unsupported claims.",
            )

        app = WorkbenchApp(store, follow_up=follow_up)
        app.selected_job_id = job_id
        async with app.run_test() as pilot:
            await pilot.press(*"What happened next?")
            await pilot.press("enter")
            await pilot.pause(0.1)
        assert questions == ["What happened next?"]

    asyncio.run(exercise())
