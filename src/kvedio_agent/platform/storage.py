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

from kvedio_agent.platform.settings import config_dir

_SCHEMA_VERSION = 4
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

    def recover_interrupted_work(self) -> int:
        """Return interrupted jobs and their in-flight stages and segments to retryable state."""

        now = _timestamp()
        with self._connect() as connection:
            jobs = connection.execute(
                "UPDATE jobs SET status='interrupted', updated_at=? WHERE status='running'",
                (now,),
            ).rowcount
            connection.execute(
                "UPDATE stages SET status='interrupted' WHERE status='running'"
            )
            connection.execute(
                "UPDATE segments SET status='pending' WHERE status='running'"
            )
            connection.execute(
                "UPDATE usage_records SET status='uncertain', finished_at=? "
                "WHERE status='reserved'",
                (now,),
            )
        return jobs

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
                    CREATE TABLE usage_records (
                        id TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        stage TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        call_key TEXT NOT NULL,
                        attempt INTEGER NOT NULL DEFAULT 1 CHECK (attempt > 0),
                        status TEXT NOT NULL,
                        estimated_cost_usd REAL CHECK (
                            estimated_cost_usd IS NULL OR estimated_cost_usd >= 0
                        ),
                        actual_cost_usd REAL CHECK (
                            actual_cost_usd IS NULL OR actual_cost_usd >= 0
                        ),
                        input_tokens INTEGER,
                        output_tokens INTEGER,
                        audio_seconds REAL,
                        created_at TEXT NOT NULL,
                        finished_at TEXT,
                        UNIQUE(job_id, provider, call_key)
                    );
                    CREATE INDEX usage_job_cost ON usage_records(job_id, status);
                    INSERT INTO schema_version(version, applied_at)
                    VALUES (3, CURRENT_TIMESTAMP);
                    """
                )
                version = 3
            if version == 2:
                columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(usage_records)").fetchall()
                }
                if "attempt" not in columns:
                    connection.execute(
                        "ALTER TABLE usage_records ADD COLUMN attempt INTEGER NOT NULL DEFAULT 1"
                    )
                connection.execute(
                    "INSERT INTO schema_version(version, applied_at) VALUES (3, CURRENT_TIMESTAMP)"
                )
                version = 3
            if version == 1:
                connection.executescript(
                    """
                    CREATE TABLE usage_records (
                        id TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        stage TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        call_key TEXT NOT NULL,
                        status TEXT NOT NULL,
                        estimated_cost_usd REAL,
                        actual_cost_usd REAL,
                        input_tokens INTEGER,
                        output_tokens INTEGER,
                        audio_seconds REAL,
                        created_at TEXT NOT NULL,
                        finished_at TEXT,
                        UNIQUE(job_id, provider, call_key)
                    );
                    CREATE INDEX usage_job_cost ON usage_records(job_id, status);
                    INSERT INTO schema_version(version, applied_at)
                    VALUES (2, CURRENT_TIMESTAMP);
                    """
                )
                version = 2
            if version == 2:
                connection.executescript(
                    """
                    ALTER TABLE usage_records ADD COLUMN attempt INTEGER NOT NULL DEFAULT 1;
                    INSERT INTO schema_version(version, applied_at)
                    VALUES (3, CURRENT_TIMESTAMP);
                    """
                )
                version = 3
            if version == 3:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS uploads (
                        id TEXT PRIMARY KEY,
                        source_path TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
                        digest TEXT NOT NULL,
                        chunk_size INTEGER NOT NULL CHECK (chunk_size > 0),
                        status TEXT NOT NULL CHECK (status IN ('receiving', 'completed')),
                        target_path TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS upload_chunks (
                        upload_id TEXT NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
                        size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
                        digest TEXT NOT NULL,
                        acknowledged_at TEXT NOT NULL,
                        PRIMARY KEY(upload_id, chunk_index)
                    );
                    INSERT INTO schema_version(version, applied_at)
                    VALUES (4, CURRENT_TIMESTAMP);
                    """
                )
            self._ensure_evidence_search(connection)

    def create_upload(
        self, upload_id: str, source_path: Path, size_bytes: int,
        digest: str, chunk_size: int,
    ) -> None:
        """Create a durable local transfer record before acknowledging any chunk."""

        now = _timestamp()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO uploads VALUES (?, ?, ?, ?, ?, 'receiving', NULL, ?, ?)",
                (upload_id, str(source_path.resolve()), size_bytes, digest, chunk_size, now, now),
            )

    def upload(self, upload_id: str) -> dict[str, Any] | None:
        """Return transfer metadata and acknowledged chunk indexes."""

        with self._connect() as connection:
            row = connection.execute("SELECT * FROM uploads WHERE id=?", (upload_id,)).fetchone()
            if row is None:
                return None
            chunks = connection.execute(
                "SELECT chunk_index, size_bytes, digest FROM upload_chunks "
                "WHERE upload_id=? ORDER BY chunk_index", (upload_id,),
            ).fetchall()
        return {**dict(row), "chunks": [dict(item) for item in chunks]}

    def upload_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent upload status without exposing source contents."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, source_path, size_bytes, status, target_path, updated_at "
                "FROM uploads ORDER BY updated_at DESC LIMIT ?", (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def acknowledge_upload_chunk(
        self, upload_id: str, chunk_index: int, size_bytes: int, digest: str
    ) -> None:
        """Record a chunk only after its bytes are durably written and verified."""

        with self._connect() as connection:
            connection.execute(
                "INSERT INTO upload_chunks VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(upload_id, chunk_index) DO UPDATE SET "
                "size_bytes=excluded.size_bytes, digest=excluded.digest, "
                "acknowledged_at=excluded.acknowledged_at",
                (upload_id, chunk_index, size_bytes, digest, _timestamp()),
            )
            connection.execute(
                "UPDATE uploads SET updated_at=? WHERE id=?", (_timestamp(), upload_id)
            )

    def finish_upload(self, upload_id: str, target_path: Path) -> None:
        """Mark a fully verified transfer complete with its immutable workspace path."""

        with self._connect() as connection:
            connection.execute(
                "UPDATE uploads SET status='completed', target_path=?, updated_at=? WHERE id=?",
                (str(target_path.resolve()), _timestamp(), upload_id),
            )

    @staticmethod
    def _ensure_evidence_search(connection: sqlite3.Connection) -> None:
        """Create and backfill the FTS index used for bounded evidence retrieval."""

        connection.execute(
            """CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(
                evidence_id UNINDEXED, job_id UNINDEXED, modality UNINDEXED,
                start_ms UNINDEXED, end_ms UNINDEXED, content,
                tokenize='unicode61')"""
        )
        indexed = connection.execute("SELECT COUNT(*) AS count FROM evidence_fts").fetchone()
        has_evidence = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='evidence'"
        ).fetchone()
        if not has_evidence:
            return
        evidence = connection.execute(
            "SELECT id, job_id, modality, start_ms, end_ms, content FROM evidence "
            "WHERE content IS NOT NULL"
        ).fetchall()
        if indexed["count"] != len(evidence):
            connection.execute("DELETE FROM evidence_fts")
            connection.executemany(
                "INSERT INTO evidence_fts VALUES (?, ?, ?, ?, ?, ?)",
                [tuple(row) for row in evidence],
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

    def find_job(
        self, source_id: str, objective: str, config_snapshot: dict[str, Any]
    ) -> str | None:
        """Find the latest job with matching source, objective, and execution settings."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM jobs WHERE source_id=? AND objective=? AND config_json=? "
                "ORDER BY created_at DESC LIMIT 1",
                (source_id, objective, _json(config_snapshot)),
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

    def pending_segments(self, job_id: str, kind: str) -> list[dict[str, Any]]:
        """Return work units not yet completed for one processing kind."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM segments WHERE job_id=? AND kind=? AND status!='completed' "
                "ORDER BY start_ms, end_ms",
                (job_id, kind),
            ).fetchall()
        return [dict(row) for row in rows]

    def evidence_exists(self, job_id: str, modality: str, start_ms: int, end_ms: int) -> bool:
        """Check whether a normalized result for the exact work range is stored."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM evidence WHERE job_id=? AND modality=? "
                "AND start_ms=? AND end_ms=? LIMIT 1",
                (job_id, modality, start_ms, end_ms),
            ).fetchone()
        return row is not None

    def evidence_for_modality(self, job_id: str, modality: str) -> list[dict[str, Any]]:
        """Return one evidence modality ordered by source time."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM evidence WHERE job_id=? AND modality=? ORDER BY start_ms, end_ms",
                (job_id, modality),
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_evidence(self, job_id: str, limit: int) -> list[dict[str, Any]]:
        """Return a bounded set of the most recent source-time evidence."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM evidence WHERE job_id=? ORDER BY start_ms DESC, end_ms DESC LIMIT ?",
                (job_id, limit),
            ).fetchall()
        return sorted((dict(row) for row in rows), key=lambda item: item["start_ms"])

    def upsert_stage(self, job_id: str, name: str, status: str, attempt: int = 0,
                     progress: float = 0, error_category: str | None = None) -> str:
        """Persist stage progress without storing raw exception details."""

        stage_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO stages(id, job_id, name, status, attempt, progress, error_category)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(job_id, name) DO UPDATE SET status=excluded.status,
                   attempt=MAX(stages.attempt, excluded.attempt), progress=excluded.progress,
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
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """SELECT id FROM evidence WHERE job_id=? AND modality=? AND start_ms=?
                   AND end_ms=? AND content IS ? AND artifact_id IS ? LIMIT 1""",
                (job_id, modality, start_ms, end_ms, content, artifact_id),
            ).fetchone()
            if existing:
                return str(existing["id"])
            connection.execute(
                "INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (evidence_id, job_id, modality, start_ms, end_ms, content, artifact_id,
                 confidence, _json(provenance), _timestamp()),
            )
            if content is not None:
                connection.execute(
                    "INSERT INTO evidence_fts VALUES (?, ?, ?, ?, ?, ?)",
                    (evidence_id, job_id, modality, start_ms, end_ms, content),
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

    def reserve_usage(
        self,
        job_id: str,
        stage: str,
        provider: str,
        call_key: str,
        estimated_cost_usd: float | None,
        max_job_cost_usd: float | None,
    ) -> tuple[str, str | None, bool]:
        """Reserve estimated call cost before dispatch, rejecting unknown cost under a ceiling."""

        usage_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT id, status, attempt FROM usage_records WHERE job_id=? AND provider=? "
                "AND (call_key=? OR call_key LIKE ?) ORDER BY attempt DESC LIMIT 1",
                (job_id, provider, call_key, f"{call_key}:retry:%"),
            ).fetchone()
            if existing:
                if existing["status"] == "completed":
                    raise ValueError(
                        "provider call is already complete; expected cached output evidence"
                    )
                if existing["status"] == "uncertain":
                    raise ValueError(
                        "provider call outcome is uncertain; refusing duplicate dispatch"
                    )
                if existing["status"] != "retryable_failed":
                    raise ValueError(
                        "provider call already has a durable outcome; refusing duplicate dispatch"
                    )
                attempt = int(existing["attempt"]) + 1
                call_key = f"{call_key}:retry:{attempt}"
            else:
                attempt = 1
            if max_job_cost_usd is not None:
                if estimated_cost_usd is None:
                    raise ValueError(
                        "call cost is unknown; configure pricing before enabling a budget"
                    )
                row = connection.execute(
                    """SELECT COALESCE(
                           SUM(COALESCE(actual_cost_usd, estimated_cost_usd)), 0
                       ) AS cost FROM usage_records WHERE job_id=?""",
                    (job_id,),
                ).fetchone()
                if row["cost"] + estimated_cost_usd > max_job_cost_usd:
                    raise ValueError("job spend ceiling would be exceeded")
            connection.execute(
                """INSERT INTO usage_records
                   (id, job_id, stage, provider, call_key, attempt, status,
                    estimated_cost_usd, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'reserved', ?, ?)""",
                (usage_id, job_id, stage, provider, call_key, attempt,
                 estimated_cost_usd, _timestamp()),
            )
        return usage_id, None, False

    def finish_usage(
        self,
        usage_id: str,
        actual_cost_usd: float | None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        audio_seconds: float | None = None,
        failed: bool = False,
        retryable: bool = False,
    ) -> None:
        """Settle usage or retain its reservation as uncertain after a failed call."""

        status = (
            "retryable_failed" if failed and retryable
            else "uncertain" if failed
            else "completed"
        )
        with self._connect() as connection:
            connection.execute(
                """UPDATE usage_records SET status=?, actual_cost_usd=?, input_tokens=?,
                   output_tokens=?, audio_seconds=?, finished_at=? WHERE id=?""",
                (status, actual_cost_usd, input_tokens, output_tokens, audio_seconds,
                 _timestamp(), usage_id),
            )

    def usage_summary(self, job_id: str) -> dict[str, Any]:
        """Separate actual, estimated, reserved, failed, and unknown provider exposure."""

        with self._connect() as connection:
            row = connection.execute(
                """SELECT COUNT(*) AS calls,
                   COALESCE(SUM(actual_cost_usd), 0) AS actual_cost_usd,
                   COALESCE(SUM(CASE WHEN actual_cost_usd IS NULL AND status='completed'
                       THEN estimated_cost_usd ELSE 0 END), 0) AS estimated_only_cost_usd,
                   COALESCE(SUM(CASE WHEN status='reserved' THEN estimated_cost_usd ELSE 0 END), 0)
                       AS reserved_cost_usd,
                   SUM(CASE WHEN actual_cost_usd IS NULL AND estimated_cost_usd IS NOT NULL
                       AND status='completed' THEN 1 ELSE 0 END) AS estimated_only_calls,
                   SUM(CASE WHEN status='uncertain' OR
                       (actual_cost_usd IS NULL AND estimated_cost_usd IS NULL)
                       THEN 1 ELSE 0 END) AS unknown_calls,
                   COALESCE(SUM(COALESCE(actual_cost_usd, estimated_cost_usd)), 0) AS known_cost_usd
                   FROM usage_records WHERE job_id=?""",
                (job_id,),
            ).fetchone()
        return dict(row)

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

    def latest_conversation(self, job_id: str) -> str | None:
        """Return the most recently updated conversation for a job."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM conversations WHERE job_id=? ORDER BY updated_at DESC LIMIT 1",
                (job_id,),
            ).fetchone()
        return str(row["id"]) if row else None

    def evidence_by_id(self, evidence_id: str) -> dict[str, Any] | None:
        """Return one evidence row for citation rendering."""

        with self._connect() as connection:
            row = connection.execute("SELECT * FROM evidence WHERE id=?", (evidence_id,)).fetchone()
        return dict(row) if row else None

    def job_evidence(self, job_id: str) -> list[dict[str, Any]]:
        """Return all evidence for a job in source-time order."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM evidence WHERE job_id=? ORDER BY start_ms, end_ms", (job_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def job_events(self, job_id: str) -> list[dict[str, Any]]:
        """Return durable job events in insertion order."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM job_events WHERE job_id=? ORDER BY rowid", (job_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def job_stages(self, job_id: str) -> list[dict[str, Any]]:
        """Return durable stage status, attempt, and progress for one job."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT name, status, attempt, progress, error_category FROM stages "
                "WHERE job_id=? ORDER BY rowid", (job_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_job(self, job_id: str) -> None:
        """Delete generated job metadata and workspace files without touching sources."""

        with self._connect() as connection:
            artifact_paths = connection.execute(
                "SELECT path FROM artifacts WHERE job_id=?", (job_id,)
            ).fetchall()
            source = connection.execute(
                "SELECT source_id FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            connection.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            remaining = (
                connection.execute(
                    "SELECT 1 FROM jobs WHERE source_id=? LIMIT 1", (source["source_id"],)
                ).fetchone()
                if source
                else None
            )
            if source and not remaining:
                connection.execute("DELETE FROM video_sources WHERE id=?", (source["source_id"],))
        for row in artifact_paths:
            path = Path(row["path"]).resolve()
            if path.is_relative_to(self.root) and path.is_file():
                path.unlink()

    def evidence_for_range(self, job_id: str, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
        """Return evidence overlapping a requested source-relative time range."""

        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM evidence WHERE job_id=? AND start_ms < ? AND end_ms > ?
                   ORDER BY start_ms, end_ms""",
                (job_id, end_ms, start_ms),
            ).fetchall()
        return [dict(row) for row in rows]

    def search_evidence(
        self, job_id: str, query_terms: list[str], limit: int,
        modality: str | None = None, start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return ranked FTS evidence with optional modality and time filters."""

        terms = [term.strip() for term in query_terms if term.strip()]
        if not terms or limit <= 0:
            return []
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        filters = ["e.job_id=?"]
        parameters: list[Any] = [job_id]
        if modality is not None:
            filters.append("e.modality=?")
            parameters.append(modality)
        if start_ms is not None:
            filters.append("e.start_ms < ?")
            parameters.append(start_ms)
        if end_ms is not None:
            filters.append("e.end_ms > ?")
            parameters.append(end_ms)
        parameters.append(match)
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT e.* FROM evidence e JOIN evidence_fts f ON f.evidence_id=e.id "
                f"WHERE {' AND '.join(filters)} AND evidence_fts MATCH ? "
                "ORDER BY bm25(evidence_fts), e.start_ms, e.end_ms LIMIT ?",
                parameters,
            ).fetchall()
            if not rows:
                like_filters = ["job_id=?", "content IS NOT NULL"]
                like_values: list[Any] = [job_id]
                like_filters.extend("content LIKE ?" for _ in terms)
                like_values.extend(f"%{term}%" for term in terms)
                if modality is not None:
                    like_filters.append("modality=?")
                    like_values.append(modality)
                if start_ms is not None:
                    like_filters.append("start_ms < ?")
                    like_values.append(start_ms)
                if end_ms is not None:
                    like_filters.append("end_ms > ?")
                    like_values.append(end_ms)
                like_values.append(limit)
                rows = connection.execute(
                    f"SELECT * FROM evidence WHERE {' AND '.join(like_filters)} "
                    "ORDER BY start_ms, end_ms LIMIT ?", like_values,
                ).fetchall()
        return [dict(row) for row in rows]

    def checkpoint(self, job_id: str) -> dict[str, Any] | None:
        """Restore the persisted structured scratchpad for a job."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT scratchpad_json FROM checkpoints WHERE job_id=?", (job_id,)
            ).fetchone()
        return json.loads(row["scratchpad_json"]) if row else None

    def add_artifact_summary(
        self, job_id: str, kind: str, start_ms: int, end_ms: int, summary: str,
        producer: str, retention_days: int, evidence_ids: list[str] | None = None,
    ) -> str:
        """Store a generated text summary as an artifact and searchable evidence."""

        artifact_dir = self.root / "artifacts" / job_id / "summaries"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        digest = __import__("hashlib").sha256(summary.encode("utf-8")).hexdigest()
        path = artifact_dir / f"{kind}-{start_ms}-{end_ms}-{digest[:12]}.txt"
        if not path.exists():
            path.write_text(summary, encoding="utf-8")
        artifact_id = self.add_artifact(
            job_id, kind, path, digest, producer, retention_days
        )
        evidence_id = self.add_evidence(
            job_id, kind, start_ms, end_ms,
            {"producer": producer, "evidence_ids": evidence_ids or []}, summary, artifact_id
        )
        return evidence_id

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
