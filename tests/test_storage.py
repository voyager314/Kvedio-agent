from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kvedio_agent.storage import Store


def _source(store: Store, tmp_path: Path) -> str:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    return store.register_source(source, source.stat().st_size, source.stat().st_mtime_ns, "digest")


def test_store_migrates_and_restores_job_conversation_and_checkpoint(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    source_id = _source(store, tmp_path)
    job_id = store.create_job(source_id, "find the key moment", {"retention_days": 30})
    stage_id = store.upsert_stage(job_id, "probe", "completed", progress=1)
    segment_id = store.add_segment(job_id, "audio", 0, 120_000)
    evidence_id = store.add_evidence(
        job_id, "transcript", 1_000, 2_000, {"producer": "fake-asr"}, "hello"
    )
    conversation_id = store.create_conversation(job_id, "First analysis")
    store.add_message(conversation_id, "user", "What happened?", [evidence_id])
    store.add_message(conversation_id, "assistant", "It happened here.", [evidence_id])
    store.save_checkpoint(job_id, {"current_stage": "probe", "evidence_refs": [evidence_id]})
    store.add_event(job_id, "stage_completed", "probe complete", {"stage": stage_id})

    reopened = Store(tmp_path / "workspace")
    assert reopened.recoverable_jobs()[0]["id"] == job_id
    messages = reopened.conversation_messages(conversation_id)
    assert [row["role"] for row in messages] == ["user", "assistant"]
    assert reopened.evidence_for_range(job_id, 1_500, 1_700)[0]["id"] == evidence_id
    assert segment_id


def test_artifacts_expire_without_deleting_source(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    source_id = store.register_source(
        source_path, source_path.stat().st_size, source_path.stat().st_mtime_ns, "digest"
    )
    job_id = store.create_job(source_id, "objective", {})
    artifact_path = store.root / "artifacts" / "chunk.json"
    artifact_path.parent.mkdir()
    artifact_path.write_text("generated", encoding="utf-8")
    store.add_artifact(job_id, "transcript", artifact_path, "artifact-digest", "fake", 30)

    with sqlite3.connect(store.database) as connection:
        connection.execute(
            "UPDATE artifacts SET expires_at=?",
            ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(),),
        )
    assert store.cleanup_expired(30) == 1
    assert not artifact_path.exists()
    assert source_path.exists()


def test_storage_rejects_secrets_and_external_artifacts(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    source_id = _source(store, tmp_path)
    job_id = store.create_job(source_id, "objective", {})
    with pytest.raises(ValueError, match="secret"):
        store.create_job(source_id, "objective", {"api_key": "secret"})
    external = tmp_path / "outside.json"
    with pytest.raises(ValueError, match="workspace"):
        store.add_artifact(job_id, "data", external, "digest", "fake", 30)
