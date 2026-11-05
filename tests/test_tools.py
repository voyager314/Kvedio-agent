from __future__ import annotations

from pathlib import Path

import pytest

from kvedio_agent.pipeline.tools import JobToolRegistry, ToolCallError
from kvedio_agent.platform.storage import Store


def _job(store: Store, tmp_path: Path) -> str:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    source_id = store.register_source(source, 6, source.stat().st_mtime_ns, "digest")
    return store.create_job(source_id, "objective", {})


def test_job_tools_return_bounded_timestamped_evidence(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    job_id = _job(store, tmp_path)
    evidence_id = store.add_evidence(
        job_id, "transcript", 1000, 2000, {}, "launch approved"
    )
    registry = JobToolRegistry(store, job_id)
    result = registry.call("search_evidence", {"query": "launch"})
    assert result["evidence"][0]["evidence_id"] == evidence_id
    filtered = registry.call(
        "search_evidence", {"query": "launch", "modality": "transcript"}
    )
    assert filtered["evidence"][0]["evidence_id"] == evidence_id
    range_result = registry.call(
        "get_transcript", {"start_ms": 1500, "end_ms": 1600}
    )
    assert range_result["evidence"][0]["start_ms"] == 1000


def test_job_tools_reject_unknown_and_invalid_calls(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    registry = JobToolRegistry(store, _job(store, tmp_path))
    with pytest.raises(ToolCallError):
        registry.call("run_shell", {})
    with pytest.raises(ToolCallError):
        registry.call("get_transcript", {"start_ms": 2, "end_ms": 1})
    with pytest.raises(ToolCallError):
        registry.call("get_transcript", {"start_ms": 0, "end_ms": 1, "secret": "x"})


def test_tool_schemas_forbid_extra_arguments(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    registry = JobToolRegistry(store, _job(store, tmp_path))
    assert all(
        schema["function"]["parameters"]["additionalProperties"] is False
        for schema in registry.schemas()
    )
