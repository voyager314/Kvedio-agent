from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kvedio_agent.platform.ingestion import LocalChunkIngestor
from kvedio_agent.platform.storage import Store


def test_chunk_ingestion_resumes_and_verifies_final_digest(tmp_path: Path) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(bytes(range(32)))
    store = Store(tmp_path / "workspace")
    ingestor = LocalChunkIngestor(store, chunk_size=8)

    partial = ingestor.ingest(source, max_chunks=2)
    assert not partial.completed
    assert partial.acknowledged_chunks == 2
    resumed = ingestor.ingest(source, partial.upload_id)
    assert resumed.completed
    assert resumed.target_path is not None
    assert resumed.target_path.read_bytes() == source.read_bytes()
    assert hashlib.sha256(resumed.target_path.read_bytes()).hexdigest() == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert store.upload(partial.upload_id)["status"] == "completed"  # type: ignore[index]


def test_chunk_ingestion_rejects_source_mismatch_on_resume(tmp_path: Path) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"first content")
    store = Store(tmp_path / "workspace")
    upload_id = LocalChunkIngestor(store, 4).ingest(source, max_chunks=1).upload_id
    source.write_bytes(b"changed content")
    with pytest.raises(ValueError, match="do not match"):
        LocalChunkIngestor(store, 4).ingest(source, upload_id)
