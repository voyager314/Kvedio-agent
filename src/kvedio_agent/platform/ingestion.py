"""Resumable, digest-verified chunk ingestion into the local workspace."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path

from kvedio_agent.platform.storage import Store


@dataclass(frozen=True)
class UploadProgress:
    """Durable state returned after processing one local transfer."""

    upload_id: str  # Stable transfer identifier used to resume the same source.
    acknowledged_chunks: int  # Number of verified chunks persisted by the receiver.
    total_chunks: int  # Number of chunks implied by source size and chunk size.
    completed: bool  # True after whole-file digest verification and atomic finalization.
    target_path: Path | None  # Workspace copy when complete; None while receiving.


class LocalChunkIngestor:
    """Copy local video files through durable chunks without modifying their source."""

    def __init__(self, store: Store, chunk_size: int = 8 * 1024 * 1024) -> None:
        if chunk_size < 1:
            raise ValueError("chunk size must be positive")
        self.store = store
        self.chunk_size = chunk_size

    def ingest(
        self, source: Path, upload_id: str | None = None,
        max_chunks: int | None = None,
    ) -> UploadProgress:
        """Persist missing chunks and finalize only when source identity matches the transfer."""

        path = source.expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix.casefold() != ".mp4":
            raise ValueError("upload source must be an MP4 file")
        size_bytes = path.stat().st_size
        if size_bytes < 1:
            raise ValueError("upload source is empty")
        source_digest = _digest(path)
        selected_id = upload_id or str(uuid.uuid4())
        state = self.store.upload(selected_id)
        if state is None:
            self.store.create_upload(
                selected_id, path, size_bytes, source_digest, self.chunk_size
            )
            state = self.store.upload(selected_id)
        if state is None:
            raise RuntimeError("upload record was not created")
        if (
            state["source_path"] != str(path)
            or state["size_bytes"] != size_bytes
            or state["digest"] != source_digest
            or state["chunk_size"] != self.chunk_size
        ):
            raise ValueError("upload source or chunk settings do not match the saved transfer")
        total_chunks = (size_bytes + self.chunk_size - 1) // self.chunk_size
        target_dir = self.store.root / "uploads"
        target_dir.mkdir(parents=True, exist_ok=True)
        partial_path = target_dir / f"{selected_id}.part"
        final_path = (
            Path(state["target_path"])
            if state["target_path"]
            else target_dir / f"{selected_id}.mp4"
        )
        if state["status"] == "completed":
            return UploadProgress(selected_id, total_chunks, total_chunks, True, final_path)

        acknowledged = {
            row["chunk_index"]: row
            for row in state["chunks"]
        }
        processed = 0
        with path.open("rb") as source_file, partial_path.open("a+b") as target_file:
            for chunk_index in range(total_chunks):
                offset = chunk_index * self.chunk_size
                source_file.seek(offset)
                data = source_file.read(min(self.chunk_size, size_bytes - offset))
                chunk_digest = hashlib.sha256(data).hexdigest()
                saved = acknowledged.get(chunk_index)
                target_file.seek(offset)
                existing = target_file.read(len(data)) if saved else b""
                if (
                    saved
                    and saved["size_bytes"] == len(data)
                    and saved["digest"] == chunk_digest
                    and hashlib.sha256(existing).hexdigest() == chunk_digest
                ):
                    continue
                if max_chunks is not None and processed >= max_chunks:
                    break
                target_file.seek(offset)
                target_file.write(data)
                target_file.flush()
                self.store.acknowledge_upload_chunk(
                    selected_id, chunk_index, len(data), chunk_digest
                )
                processed += 1

        latest = self.store.upload(selected_id) or state
        acknowledged_indexes = {row["chunk_index"] for row in latest["chunks"]}
        complete = len(acknowledged_indexes) == total_chunks
        if complete:
            if _digest(partial_path) != source_digest:
                raise ValueError("assembled upload digest does not match its source")
            partial_path.replace(final_path)
            self.store.finish_upload(selected_id, final_path)
            return UploadProgress(selected_id, total_chunks, total_chunks, True, final_path)
        return UploadProgress(
            selected_id, len(acknowledged_indexes), total_chunks, False, None
        )


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as content:
        while block := content.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
