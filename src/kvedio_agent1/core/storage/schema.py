"""The SQLite schema and the migrations that reach it.

The schema version lives in SQLite's own ``user_version`` header, so opening a
database written by a newer build is detectable before any row is touched. Each
migration is applied in order, which lets an older file catch up without a
migration framework.
"""

from __future__ import annotations

import sqlite3

from .errors import StorageError

SCHEMA_VERSION = 4

_INITIAL = """
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
    job_id          TEXT    NOT NULL REFERENCES job(id),
    id              TEXT    NOT NULL,
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
    error           TEXT,
    PRIMARY KEY (job_id, id)
);
"""

_TRANSCRIPT = """
CREATE TABLE transcript (
    id           INTEGER PRIMARY KEY,
    video_id     INTEGER NOT NULL UNIQUE REFERENCES video(id),
    cache_key    TEXT    NOT NULL,
    model        TEXT    NOT NULL,
    created_at   TEXT    NOT NULL,
    completed_at TEXT
);

CREATE TABLE transcript_segment (
    id            INTEGER PRIMARY KEY,
    transcript_id INTEGER NOT NULL REFERENCES transcript(id),
    start_ms      INTEGER NOT NULL,
    end_ms        INTEGER NOT NULL,
    text          TEXT    NOT NULL
);

CREATE INDEX idx_segment_transcript ON transcript_segment(transcript_id, start_ms);

CREATE TABLE checkpoint (
    video_id    INTEGER NOT NULL REFERENCES video(id),
    stage       TEXT    NOT NULL,
    cache_key   TEXT    NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_ms    INTEGER NOT NULL,
    end_ms      INTEGER NOT NULL,
    created_at  TEXT    NOT NULL,
    PRIMARY KEY (video_id, stage, cache_key, chunk_index)
);
"""

_ANSWERS = """
CREATE TABLE answer (
    id         INTEGER PRIMARY KEY,
    job_id     TEXT    NOT NULL UNIQUE REFERENCES job(id),
    video_id   INTEGER NOT NULL REFERENCES video(id),
    question   TEXT    NOT NULL,
    text       TEXT    NOT NULL,
    model      TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);

CREATE TABLE answer_citation (
    id        INTEGER PRIMARY KEY,
    answer_id INTEGER NOT NULL REFERENCES answer(id),
    start_ms  INTEGER NOT NULL,
    end_ms    INTEGER NOT NULL,
    quote     TEXT    NOT NULL
);

CREATE INDEX idx_citation_answer ON answer_citation(answer_id, start_ms);
"""

_MEMORY = """
CREATE TABLE coverage_block (
    id         INTEGER PRIMARY KEY,
    video_id   INTEGER NOT NULL REFERENCES video(id),
    cache_key  TEXT    NOT NULL,
    position   INTEGER NOT NULL,          -- 该块在录像中的次序
    start_ms   INTEGER NOT NULL,
    end_ms     INTEGER NOT NULL,
    gist       TEXT    NOT NULL,
    created_at TEXT    NOT NULL,
    UNIQUE (video_id, cache_key, position)
);

CREATE TABLE segment_vector (
    video_id   INTEGER NOT NULL REFERENCES video(id),
    cache_key  TEXT    NOT NULL,
    segment_id INTEGER NOT NULL REFERENCES transcript_segment(id),
    start_ms   INTEGER NOT NULL,          -- 冗余一份，检索时不必回头 join 排序
    dims       INTEGER NOT NULL,
    vector     BLOB    NOT NULL,          -- float32 小端，紧密排列
    PRIMARY KEY (video_id, cache_key, segment_id)
);

CREATE INDEX idx_vector_key ON segment_vector(video_id, cache_key, start_ms);
"""

_MIGRATIONS: dict[int, str] = {1: _INITIAL, 2: _TRANSCRIPT, 3: _ANSWERS, 4: _MEMORY}


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
    for target in range(version + 1, SCHEMA_VERSION + 1):
        connection.executescript(_MIGRATIONS[target])
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    connection.commit()
