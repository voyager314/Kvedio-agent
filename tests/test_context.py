from __future__ import annotations

from pathlib import Path

from kvedio_agent.context import ContextBudget, EvidenceIndex, Scratchpad
from kvedio_agent.storage import Store


def _job(store: Store, tmp_path: Path) -> str:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    source_id = store.register_source(source, 6, source.stat().st_mtime_ns, "digest")
    return store.create_job(source_id, "explain the launch", {})


def test_scratchpad_roundtrips_as_compact_checkpoint(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    job_id = _job(store, tmp_path)
    pad = Scratchpad(objective="explain the launch", current_stage="timeline")
    pad.completed_ranges.append((0, 60_000))
    pad.transcript_facts.append({"text": "Launch approved", "evidence_id": "e1"})
    pad.visual_facts.append({"text": "Green status slide", "evidence_id": "e2"})
    pad.cross_modal_facts.append({"text": "Speech matches slide", "evidence_ids": ["e1", "e2"]})
    pad.save(store, job_id)
    restored = Scratchpad.from_dict(store.checkpoint(job_id) or {})
    assert restored.completed_ranges == [(0, 60_000)]
    assert restored.cross_modal_facts[0]["evidence_ids"] == ["e1", "e2"]


def test_evidence_retrieval_respects_time_and_size_budgets(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    job_id = _job(store, tmp_path)
    first = store.add_evidence(job_id, "transcript", 0, 1000, {}, "launch date announced")
    store.add_evidence(job_id, "visual", 900, 1500, {}, "launch date shown on slide")
    store.add_evidence(job_id, "transcript", 3000, 4000, {}, "unrelated closing remarks")
    index = EvidenceIndex(store, ContextBudget(max_items=1, max_bytes=30))
    result = index.retrieve(job_id, "launch date")
    assert len(result) == 1
    assert result[0]["id"] in {first}
    assert len(result[0]["content"].encode("utf-8")) <= 30
    assert [item["id"] for item in index.retrieve(job_id, "", (800, 1100))] == [first]
