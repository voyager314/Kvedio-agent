from __future__ import annotations

from pathlib import Path

from kvedio_agent.interface.tui import WorkbenchApp, event_lines, evidence_lines, job_label
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


def test_workbench_app_composes_without_network_calls(tmp_path: Path) -> None:
    app = WorkbenchApp(Store(tmp_path / "workspace"))
    assert app.selected_job_id is None
    assert app.CSS
