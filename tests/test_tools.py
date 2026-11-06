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


def test_clip_tool_is_offered_only_with_an_observer(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    job_id = _job(store, tmp_path)
    names = {schema["function"]["name"] for schema in JobToolRegistry(store, job_id).schemas()}
    assert "inspect_clip" not in names
    observed = JobToolRegistry(store, job_id, clip_observer=lambda start, end: [])
    assert "inspect_clip" in {schema["function"]["name"] for schema in observed.schemas()}


def test_clip_tool_observes_each_range_once_and_narrows_wide_requests(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "workspace")
    job_id = _job(store, tmp_path)
    calls: list[tuple[int, int]] = []

    def observer(start_ms: int, end_ms: int) -> list[dict[str, object]]:
        calls.append((start_ms, end_ms))
        evidence_id = store.add_evidence(
            job_id, "visual", start_ms, end_ms, {}, f"watched {start_ms}..{end_ms} ms"
        )
        return [{"evidence_id": evidence_id, "start_ms": start_ms, "end_ms": end_ms,
                 "modality": "visual", "content": "action"}]

    registry = JobToolRegistry(store, job_id, clip_observer=observer, max_clip_ms=1_000)
    first = registry.call("inspect_clip", {"start_ms": 0, "end_ms": 5_000})
    assert calls == [(0, 1_000)]
    assert first["end_ms"] == 1_000
    assert first["evidence"][0]["evidence_id"]
    assert len(first["evidence"]) == 1
    repeated = registry.call("inspect_clip", {"start_ms": 0, "end_ms": 5_000})
    assert repeated == first
    assert calls == [(0, 1_000)]
    with pytest.raises(ToolCallError):
        registry.call("inspect_clip", {"start_ms": 900, "end_ms": 100})


def test_agent_can_read_and_write_utf8_files_with_spaces(tmp_path: Path) -> None:
    registry = JobToolRegistry(Store(tmp_path / "workspace"), "job")
    target = tmp_path / "folder with spaces" / "notes.txt"

    written = registry.call(
        "write_file", {"path": str(target), "content": "会议结论"}
    )
    read = registry.call("read_file", {"path": str(target)})

    assert written["path"] == str(target.resolve())
    assert read["content"] == "会议结论"


def test_file_tools_reject_oversized_content_and_extra_arguments(tmp_path: Path) -> None:
    registry = JobToolRegistry(Store(tmp_path / "workspace"), "job")
    with pytest.raises(ToolCallError):
        registry.call("write_file", {"path": str(tmp_path / "x"), "content": "a" * 65537})
    with pytest.raises(ToolCallError):
        registry.call("read_file", {"path": str(tmp_path / "x"), "anywhere": True})
