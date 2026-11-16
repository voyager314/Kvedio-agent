"""The SQLite schema and the migration that reaches it.

The schema version lives in SQLite's own ``user_version`` header, so opening a
database written by a newer build is detectable before any row is touched.
"""

from __future__ import annotations

import sqlite3

from .errors import StorageError

SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE video (
    id            INTEGER PRIMARY KEY,
    path          TEXT    NOT NULL,
    size_bytes    INTEGER NOT NULL,
    mtime_ns      INTEGER NOT NULL,
    fingerprint   TEXT    NOT NULL UNIQUE,
    duration_ms   INTEGER,
    container     TEXT,
    video_codec   TEXT,
    width         INTEGER,
    height        INTEGER,
    fps           REAL,
    audio_codec   TEXT,
    sample_rate   INTEGER,
    channels      INTEGER,
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL
);

CREATE TABLE job (
    id          TEXT    PRIMARY KEY,
    video_id    INTEGER NOT NULL REFERENCES video(id),
    question    TEXT    NOT NULL,
    status      TEXT    NOT NULL,
    created_at  TEXT    NOT NULL,
    started_at  TEXT,
    finished_at TEXT,
    error       TEXT
);

CREATE TABLE task (
    id              TEXT    PRIMARY KEY,
    job_id          TEXT    NOT NULL REFERENCES job(id),
    agent           TEXT    NOT NULL,
    action          TEXT    NOT NULL,
    status          TEXT    NOT NULL,
    depends_on      TEXT    NOT NULL,
    dependency_mode TEXT    NOT NULL,
    attempt         INTEGER NOT NULL DEFAULT 0,
    max_attempts    INTEGER NOT NULL DEFAULT 1,
    progress        REAL    NOT NULL DEFAULT 0,
    started_at      TEXT,
    finished_at     TEXT,
    error           TEXT
);

CREATE INDEX idx_task_job ON task(job_id);
"""


def migrate(connection: sqlite3.Connection) -> None:
    """Bring the file to ``SCHEMA_VERSION``.

    Refuses a file written by a newer build: the columns it added have no
    migration to run backwards, and continuing would drop them on the next write.
    """
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        raise StorageError(
            f"数据库版本 {version} 高于本程序支持的 {SCHEMA_VERSION}，"
            "请升级程序或改用其他数据库文件"
        )
    if version == SCHEMA_VERSION:
        return
    if version < 1:
        connection.executescript(_DDL)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    connection.commit()
