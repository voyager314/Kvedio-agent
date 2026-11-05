"""SQLite persistence for local video jobs, evidence, and conversations.

The store owns durable metadata and generated workspace files. Source media paths
are references only; cleanup never removes a source file.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from kvedio_agent.settings import config_dir

_SCHEMA_VERSION = 1
_SECRET_FIELD = re.compile(r"(^|_)(api_key|token|password|secret|authorization)(_|$)", re.I)


def _timestamp(value: datetime | None = None) -> str:
    return (value or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="microseconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _assert_secret_free(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if _SECRET_FIELD.search(str(key)):
                raise ValueError("secret-bearing fields cannot be persisted")
            _assert_secret_free(child)
    elif isinstance(value, list):
        for child in value:
            _assert_secret_free(child)


class Store:
    """Persist local analysis state and generated artifacts in one workspace."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or config_dir() / "workspace").expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "kvedio.sqlite3"
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _migrate(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_version "
                "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            row = connection.execute(
                "SELECT MAX(version) AS version FROM schema_version"
            ).fetchone()
            version = row["version"] or 0
            if version > _SCHEMA_VERSION:
                raise RuntimeError("database schema is newer than this application")
            if version == 0:
                connection.executescript(
                    """
                    CREATE TABLE video_sources (
                        id TEXT PRIMARY KEY,
                        path TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
                        modified_ns INTEGER NOT NULL,
                        digest TEXT NOT NULL,
                        probe_json TEXT,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE jobs (
                        id TEXT PRIMARY KEY,
                        source_id TEXT NOT NULL REFERENCES video_sources(id) ON DELETE CASCADE,
                        objective TEXT NOT NULL,
                        status TEXT NOT NULL,
                        current_stage TEXT,
                        config_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX jobs_updated_at ON jobs(updated_at);
                    CREATE TABLE stages (
                        id TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        attempt INTEGER NOT NULL DEFAULT 0 CHECK (attempt >= 0),
                        progress REAL NOT NULL DEFAULT 0 CHECK (progress >= 0 AND progress <= 1),
                        error_category TEXT,
                        started_at TEXT,
                        finished_at TEXT,
                        UNIQUE(job_id, name)
                    );
                    CREATE TABLE segments (
                        id TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        kind TEXT NOT NULL,
                        start_ms INTEGER NOT NULL CHECK (start_ms >= 0),
                        end_ms INTEGER NOT NULL CHECK (end_ms > start_ms),
                        status TEXT NOT NULL,
                        artifact_id TEXT,
                        UNIQUE(job_id, kind, start_ms, end_ms)
                    );
                    CREATE INDEX segments_range ON segments(job_id, start_ms, end_ms);
                    CREATE TABLE artifacts (
                        id TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        segment_id TEXT REFERENCES segments(id) ON DELETE SET NULL,
                        kind TEXT NOT NULL,
                        path TEXT NOT NULL,
                        digest TEXT NOT NULL,
                        producer TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL
                    );
                    CREATE INDEX artifacts_expiry ON artifacts(expires_at);
                    CREATE TABLE evidence (
                        id TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        modality TEXT NOT NULL,
                        start_ms INTEGER NOT NULL CHECK (start_ms >= 0),
                        end_ms INTEGER NOT NULL CHECK (end_ms >= start_ms),
                        content TEXT,
                        artifact_id TEXT REFERENCES artifacts(id) ON DELETE SET NULL,
                        confidence REAL CHECK (
                            confidence IS NULL OR (confidence >= 0 AND confidence <= 1)
                        ),
                        provenance_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX evidence_range ON evidence(job_id, start_ms, end_ms);
                    CREATE TABLE checkpoints (
                        job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
                        scratchpad_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE conversations (
                        id TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        title TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX conversations_updated_at ON conversations(updated_at);
                    CREATE TABLE messages (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL
                            REFERENCES conversations(id) ON DELETE CASCADE,
                        sequence INTEGER NOT NULL CHECK (sequence >= 0),
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                        content TEXT NOT NULL,
                        evidence_ids_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(conversation_id, sequence)
                    );
                    CREATE TABLE job_events (
                        id TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        stage_id TEXT REFERENCES stages(id) ON DELETE SET NULL,
                        kind TEXT NOT NULL,
                        message TEXT NOT NULL,
                        data_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX job_events_order ON job_events(job_id, created_at);
                    INSERT INTO schema_version(version, applied_at)
                    VALUES (1, CURRENT_TIMESTAMP);
                    """
                )

    def register_source(
        self,
        path: Path,
        size_bytes: int,
        modified_ns: int,
        digest: str,
        probe: dict[str, Any] | None = None,
    ) -> str:
        """Register a source reference and its observed fingerprint."""

        source_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO video_sources VALUES (?, ?, ?, ?, ?, ?, ?)",
                (source_id, str(path.resolve()), size_bytes, modified_ns, digest,
                 _json(probe) if probe is not None else None, _timestamp()),
            )
        return source_id

    def find_source(self, path: Path, digest: str) -> str | None:
        """Return a matching source registration without accepting stale content."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM video_sources WHERE path=? AND digest=? "
                "ORDER BY created_at DESC LIMIT 1",
                (str(path.resolve()), digest),
            ).fetchone()
        return str(row["id"]) if row else None

    def create_job(
        self, source_id: str, objective: str, config_snapshot: dict[str, Any]
    ) -> str:
        """Create a resumable job with a non-secret configuration snapshot."""

        _assert_secret_free(config_snapshot)
        job_id = str(uuid.uuid4())
        now = _timestamp()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, 'queued', NULL, ?, ?, ?)",
                (job_id, source_id, objective, _json(config_snapshot), now, now),
            )
        return job_id

    def find_job(self, source_id: str, objective: str) -> str | None:
        """Find the latest reusable job with the same source and objective."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM jobs WHERE source_id=? AND objective=? "
                "ORDER BY created_at DESC LIMIT 1",
                (source_id, objective),
            ).fetchone()
        return str(row["id"]) if row else None

    def set_job_status(self, job_id: str, status: str, current_stage: str | None = None) -> None:
        """Update durable job state and its last visible stage."""

        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET status=?, current_stage=?, updated_at=? WHERE id=?",
                (status, current_stage, _timestamp(), job_id),
            )

    def set_segment_status(
        self, segment_id: str, status: str, artifact_id: str | None = None
    ) -> None:
        """Record a segment checkpoint after its generated output is durable."""

        with self._connect() as connection:
            connection.execute(
                "UPDATE segments SET status=?, artifact_id=? WHERE id=?",
                (status, artifact_id, segment_id),
            )

    def upsert_stage(self, job_id: str, name: str, status: str, attempt: int = 0,
                     progress: float = 0, error_category: str | None = None) -> str:
        """Persist stage progress without storing raw exception details."""

        stage_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO stages(id, job_id, name, status, attempt, progress, error_category)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(job_id, name) DO UPDATE SET status=excluded.status,
                   attempt=excluded.attempt, progress=excluded.progress,
                   error_category=excluded.error_category""",
                (stage_id, job_id, name, status, attempt, progress, error_category),
            )
            row = connection.execute(
                "SELECT id FROM stages WHERE job_id=? AND name=?", (job_id, name)
            ).fetchone()
            return str(row["id"])

    def add_segment(self, job_id: str, kind: str, start_ms: int, end_ms: int) -> str:
        """Create or return the durable ID for a time-bounded work unit."""

        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO segments VALUES (?, ?, ?, ?, ?, 'pending', NULL)",
                (str(uuid.uuid4()), job_id, kind, start_ms, end_ms),
            )
            row = connection.execute(
                "SELECT id FROM segments WHERE job_id=? AND kind=? AND start_ms=? AND end_ms=?",
                (job_id, kind, start_ms, end_ms),
            ).fetchone()
            return str(row["id"])

    def add_artifact(
        self, job_id: str, kind: str, path: Path, digest: str, producer: str,
        retention_days: int, segment_id: str | None = None,
    ) -> str:
        """Register a generated artifact and its expiry inside the workspace."""

        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("artifact path must be inside the application workspace")
        created = datetime.now(UTC)
        with self._connect() as connection:
            existing = connection.execute(
                """SELECT id FROM artifacts WHERE job_id=? AND segment_id IS ?
                   AND kind=? AND digest=? ORDER BY created_at DESC LIMIT 1""",
                (job_id, segment_id, kind, digest),
            ).fetchone()
            if existing:
                return str(existing["id"])
            artifact_id = str(uuid.uuid4())
            connection.execute(
                "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (artifact_id, job_id, segment_id, kind, str(resolved), digest, producer,
                 _timestamp(created), _timestamp(created + timedelta(days=retention_days))),
            )
        return artifact_id

    def add_evidence(
        self, job_id: str, modality: str, start_ms: int, end_ms: int,
        provenance: dict[str, Any], content: str | None = None,
        artifact_id: str | None = None, confidence: float | None = None,
    ) -> str:
        """Persist timestamped multimodal evidence and its provenance."""

        _assert_secret_free(provenance)
        evidence_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (evidence_id, job_id, modality, start_ms, end_ms, content, artifact_id,
                 confidence, _json(provenance), _timestamp()),
            )
        return evidence_id

    def save_checkpoint(self, job_id: str, scratchpad: dict[str, Any]) -> None:
        """Atomically replace the compact, secret-free job checkpoint."""

        _assert_secret_free(scratchpad)
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO checkpoints VALUES (?, ?, ?)
                   ON CONFLICT(job_id) DO UPDATE SET scratchpad_json=excluded.scratchpad_json,
                   updated_at=excluded.updated_at""",
                (job_id, _json(scratchpad), _timestamp()),
            )

    def create_conversation(self, job_id: str, title: str) -> str:
        """Create a separate follow-up session for an analysis job."""

        conversation_id = str(uuid.uuid4())
        now = _timestamp()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, ?, ?)",
                (conversation_id, job_id, title, now, now),
            )
        return conversation_id

    def add_message(
        self, conversation_id: str, role: str, content: str, evidence_ids: list[str] | None = None
    ) -> str:
        """Append a message at the next durable sequence number."""

        if role not in {"user", "assistant", "system"}:
            raise ValueError("unsupported message role")
        message_id = str(uuid.uuid4())
        now = _timestamp()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), -1) + 1 AS next "
                "FROM messages WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()
            connection.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
                (message_id, conversation_id, row["next"], role, content,
                 _json(evidence_ids or []), now),
            )
            connection.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id)
            )
        return message_id

    def add_event(
        self, job_id: str, kind: str, message: str, data: dict[str, Any] | None = None,
        stage_id: str | None = None,
    ) -> str:
        """Append a user-visible job event after checking it contains no secrets."""

        payload = data or {}
        _assert_secret_free(payload)
        event_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO job_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                (event_id, job_id, stage_id, kind, message, _json(payload), _timestamp()),
            )
        return event_id

    def recoverable_jobs(self) -> list[dict[str, Any]]:
        """Return jobs whose work can continue after process restart."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE status IN ('queued', 'running', 'interrupted') "
                "ORDER BY created_at"
            ).fetchall()
        return [dict(row) for row in rows]

    def job_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent jobs with source paths for the terminal workbench."""

        with self._connect() as connection:
            rows = connection.execute(
                """SELECT jobs.*, video_sources.path AS source_path FROM jobs
                   JOIN video_sources ON video_sources.id=jobs.source_id
                   ORDER BY jobs.updated_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def conversation_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        """Restore ordered messages for a conversation after restart."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM messages WHERE conversation_id=? ORDER BY sequence",
                (conversation_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def evidence_for_range(self, job_id: str, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
        """Return evidence overlapping a requested source-relative time range."""

        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM evidence WHERE job_id=? AND start_ms < ? AND end_ms > ?
                   ORDER BY start_ms, end_ms""",
                (job_id, end_ms, start_ms),
            ).fetchall()
        return [dict(row) for row in rows]

    def cleanup_expired(self, retention_days: int, now: datetime | None = None) -> int:
        """Remove expired generated artifacts and old jobs, never source videos."""

        current = now or datetime.now(UTC)
        cutoff = _timestamp(current - timedelta(days=retention_days))
        expiry = _timestamp(current)
        deleted_files = 0
        with self._connect() as connection:
            artifacts = connection.execute(
                "SELECT id, path FROM artifacts WHERE expires_at <= ?", (expiry,)
            ).fetchall()
            for artifact in artifacts:
                path = Path(artifact["path"]).resolve()
                if path.is_relative_to(self.root) and path.is_file():
                    path.unlink()
                    deleted_files += 1
                connection.execute("DELETE FROM artifacts WHERE id=?", (artifact["id"],))
            connection.execute("DELETE FROM jobs WHERE updated_at <= ?", (cutoff,))
        return deleted_files
