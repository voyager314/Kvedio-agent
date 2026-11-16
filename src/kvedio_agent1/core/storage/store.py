"""Persistence for videos, jobs and tasks.

Owns the connection, the write lock and the SQL. Every write is committed before
it returns; every stored instant is UTC. A read never creates a row.

Does not decide what a status means, does not schedule, and does not retry: the
state machine and the scheduler own those rules.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ..domain import (
    Chunk,
    DependencyMode,
    JobRecord,
    JobStatus,
    MediaInfo,
    NewTask,
    Segment,
    TaskRecord,
    TaskStatus,
    Transcript,
    VideoRecord,
)
from .errors import StorageError
from .schema import migrate

INTERRUPTED = "进程退出时作业未完成"


class Storage:
    """One SQLite file, shared by the interface thread and the worker threads.

    Calls are serialized on an internal lock, so a handler may write its result
    while the interface reads the job list.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(path, check_same_thread=False)
        except (OSError, sqlite3.Error) as exc:
            raise StorageError(f"无法打开数据库：{path}（{exc}）") from exc
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            migrate(connection)
        except Exception:
            connection.close()
            raise
        self._connection = connection

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def register_video(
        self, path: Path, *, size_bytes: int, mtime_ns: int, fingerprint: str
    ) -> VideoRecord:
        """Return the record for this fingerprint, inserting a bare one when unseen.

        The inserted record carries identity only; ``media`` stays ``None`` until
        probing records it. A repeated fingerprint does not rewrite the existing
        row, so the first reference keeps owning it.
        """
        moment = _instant(_now())
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO video"
                " (path, size_bytes, mtime_ns, fingerprint, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (str(path), size_bytes, mtime_ns, fingerprint, moment, moment),
            )
            row = self._connection.execute(
                "SELECT * FROM video WHERE fingerprint = ?", (fingerprint,)
            ).fetchone()
        if row is None:
            raise StorageError(f"视频登记失败：{path}")
        return _video(row)

    def record_media(self, video_id: int, media: MediaInfo) -> VideoRecord:
        """Replace the stored media block with what probing reported."""
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "UPDATE video SET duration_ms = ?, container = ?, video_codec = ?, width = ?,"
                " height = ?, fps = ?, audio_codec = ?, sample_rate = ?, channels = ?,"
                " updated_at = ? WHERE id = ?",
                (
                    media.duration_ms,
                    media.container,
                    media.video_codec,
                    media.width,
                    media.height,
                    media.fps,
                    media.audio_codec,
                    media.sample_rate,
                    media.channels,
                    _instant(_now()),
                    video_id,
                ),
            )
            if cursor.rowcount == 0:
                raise StorageError(f"视频不存在：{video_id}")
            row = self._connection.execute(
                "SELECT * FROM video WHERE id = ?", (video_id,)
            ).fetchone()
        return _video(row)

    def video(self, video_id: int) -> VideoRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM video WHERE id = ?", (video_id,)
            ).fetchone()
        return _video(row) if row is not None else None

    def videos(self, limit: int = 50) -> list[VideoRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM video ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_video(row) for row in rows]

    def create_job(self, video_id: int, question: str, tasks: Sequence[NewTask]) -> JobRecord:
        """Persist a job and its pending nodes as one transaction.

        Raises when a node id repeats or a dependency names a node outside the
        graph: either would leave the job waiting forever.
        """
        identifiers = [task.id for task in tasks]
        repeated = sorted({name for name in identifiers if identifiers.count(name) > 1})
        if repeated:
            raise StorageError(f"任务 id 重复：{', '.join(repeated)}")
        declared = set(identifiers)
        unknown = sorted(
            {name for task in tasks for name in task.depends_on if name not in declared}
        )
        if unknown:
            raise StorageError(f"依赖指向不存在的任务：{', '.join(unknown)}")
        _reject_cycle([(task.id, task.depends_on) for task in tasks])

        job_id = f"job_{uuid4().hex[:12]}"
        moment = _instant(_now())
        try:
            with self._lock, self._connection:
                self._connection.execute(
                    "INSERT INTO job (id, video_id, question, status, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (job_id, video_id, question, JobStatus.QUEUED.value, moment),
                )
                self._connection.executemany(
                    "INSERT INTO task (job_id, id, agent, action, status, depends_on,"
                    " dependency_mode, attempt, max_attempts, progress)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, 0)",
                    [
                        (
                            job_id,
                            task.id,
                            task.agent,
                            task.action,
                            TaskStatus.PENDING.value,
                            json.dumps(list(task.depends_on)),
                            task.dependency_mode.value,
                            task.max_attempts,
                        )
                        for task in tasks
                    ],
                )
                row = self._connection.execute(
                    "SELECT * FROM job WHERE id = ?", (job_id,)
                ).fetchone()
        except sqlite3.IntegrityError as exc:
            raise StorageError(f"作业无法落库：视频 {video_id} 不存在") from exc
        return _job(row)

    def job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM job WHERE id = ?", (job_id,)).fetchone()
        return _job(row) if row is not None else None

    def jobs(self, limit: int = 20) -> list[JobRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM job ORDER BY rowid DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_job(row) for row in rows]

    def tasks_for_job(self, job_id: str) -> list[TaskRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM task WHERE job_id = ? ORDER BY rowid", (job_id,)
            ).fetchall()
        return [_task(row) for row in rows]

    def save_job(self, job: JobRecord) -> None:
        """Write the fields a job changes after creation; identity and question are fixed."""
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "UPDATE job SET status = ?, started_at = ?, finished_at = ?, error = ?"
                " WHERE id = ?",
                (
                    job.status.value,
                    _optional_instant(job.started_at),
                    _optional_instant(job.finished_at),
                    job.error,
                    job.id,
                ),
            )
            if cursor.rowcount == 0:
                raise StorageError(f"作业不存在：{job.id}")

    def save_task(self, task: TaskRecord) -> None:
        """Write the fields a node changes while it runs.

        Identity, dependency edges and the attempt ceiling are fixed at creation,
        so a handler cannot rewrite the graph it is part of. A node id repeats
        across jobs, so the write is scoped by job as well.
        """
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "UPDATE task SET status = ?, attempt = ?, progress = ?, started_at = ?,"
                " finished_at = ?, error = ? WHERE job_id = ? AND id = ?",
                (
                    task.status.value,
                    task.attempt,
                    task.progress,
                    _optional_instant(task.started_at),
                    _optional_instant(task.finished_at),
                    task.error,
                    task.job_id,
                    task.id,
                ),
            )
            if cursor.rowcount == 0:
                raise StorageError(f"任务不存在：{task.job_id}/{task.id}")

    def transcript(self, video_id: int) -> Transcript | None:
        """The transcript of one video with its aggregates, or ``None`` when there is none."""
        with self._lock:
            row = self._connection.execute(
                "SELECT t.id, t.video_id, t.cache_key, t.model, t.created_at, t.completed_at,"
                " COUNT(s.id) AS segment_count, MIN(s.start_ms) AS first_ms,"
                " MAX(s.end_ms) AS last_ms"
                " FROM transcript t LEFT JOIN transcript_segment s ON s.transcript_id = t.id"
                " WHERE t.video_id = ? GROUP BY t.id",
                (video_id,),
            ).fetchone()
        return _transcript(row) if row is not None else None

    def open_transcript(
        self, video_id: int, cache_key: str, model: str, stage: str
    ) -> Transcript:
        """Return the transcript being written for this key, discarding one built under another.

        A video has one transcript. A different key means the earlier result came
        from inputs that no longer apply — another model, another chunk length —
        so its segments and this stage's checkpoints are dropped rather than mixed
        into the new run. The same key resumes, keeping whatever was written.
        """
        moment = _instant(_now())
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT id, cache_key FROM transcript WHERE video_id = ?", (video_id,)
            ).fetchone()
            if row is not None and row["cache_key"] != cache_key:
                self._connection.execute(
                    "DELETE FROM transcript_segment WHERE transcript_id = ?", (row["id"],)
                )
                self._connection.execute("DELETE FROM transcript WHERE id = ?", (row["id"],))
                self._connection.execute(
                    "DELETE FROM checkpoint WHERE video_id = ? AND stage = ?", (video_id, stage)
                )
                row = None
            if row is None:
                self._connection.execute(
                    "INSERT INTO transcript (video_id, cache_key, model, created_at)"
                    " VALUES (?, ?, ?, ?)",
                    (video_id, cache_key, model, moment),
                )
        found = self.transcript(video_id)
        if found is None:
            raise StorageError(f"转录无法建立：视频 {video_id}")
        return found

    def save_chunk(
        self,
        transcript_id: int,
        video_id: int,
        stage: str,
        cache_key: str,
        chunk: Chunk,
        segments: Sequence[Segment],
    ) -> None:
        """Write one chunk's segments and its completion marker as one transaction.

        Segments overlapping the chunk's interval are removed first, so a chunk
        transcribed twice leaves one copy rather than two, and a crash between the
        model call and the marker replays cleanly.
        """
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM transcript_segment WHERE transcript_id = ?"
                " AND start_ms < ? AND end_ms > ?",
                (transcript_id, chunk.end_ms, chunk.start_ms),
            )
            self._connection.executemany(
                "INSERT INTO transcript_segment (transcript_id, start_ms, end_ms, text)"
                " VALUES (?, ?, ?, ?)",
                [
                    (transcript_id, segment.start_ms, segment.end_ms, segment.text)
                    for segment in segments
                ],
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO checkpoint"
                " (video_id, stage, cache_key, chunk_index, start_ms, end_ms, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    video_id,
                    stage,
                    cache_key,
                    chunk.index,
                    chunk.start_ms,
                    chunk.end_ms,
                    _instant(_now()),
                ),
            )

    def completed_chunks(self, video_id: int, stage: str, cache_key: str) -> set[int]:
        """Indices of the chunks already written under this key."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT chunk_index FROM checkpoint"
                " WHERE video_id = ? AND stage = ? AND cache_key = ?",
                (video_id, stage, cache_key),
            ).fetchall()
        return {row["chunk_index"] for row in rows}

    def finish_transcript(self, transcript_id: int) -> None:
        """Mark the transcript complete; until then it is not cited anywhere."""
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE transcript SET completed_at = ? WHERE id = ?",
                (_instant(_now()), transcript_id),
            )

    def abandon_interrupted(self) -> list[str]:
        """Fail every job an earlier process left running, together with its unfinished nodes.

        Nothing re-dispatches that work, so reporting it as running would be a
        claim the next reader has to discover is false.
        """
        moment = _instant(_now())
        with self._lock, self._connection:
            identifiers = [
                row["id"]
                for row in self._connection.execute(
                    "SELECT id FROM job WHERE status = ?", (JobStatus.RUNNING.value,)
                )
            ]
            for job_id in identifiers:
                self._connection.execute(
                    "UPDATE task SET status = ?, error = ?, finished_at = ?"
                    " WHERE job_id = ? AND status IN (?, ?, ?)",
                    (
                        TaskStatus.FAILED.value,
                        INTERRUPTED,
                        moment,
                        job_id,
                        TaskStatus.PENDING.value,
                        TaskStatus.READY.value,
                        TaskStatus.RUNNING.value,
                    ),
                )
                self._connection.execute(
                    "UPDATE job SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                    (JobStatus.FAILED.value, INTERRUPTED, moment, job_id),
                )
        return identifiers


def _reject_cycle(edges: Sequence[tuple[str, tuple[str, ...]]]) -> None:
    """Refuse a graph whose nodes wait on each other: no scheduler can start it."""
    waiting = {name: set(dependencies) for name, dependencies in edges}
    resolved: set[str] = set()
    while waiting:
        free = [name for name, dependencies in waiting.items() if not dependencies - resolved]
        if not free:
            raise StorageError(f"任务依赖成环：{', '.join(sorted(waiting))}")
        for name in free:
            del waiting[name]
        resolved.update(free)


def _now() -> datetime:
    return datetime.now(UTC)


def _instant(moment: datetime) -> str:
    return moment.isoformat()


def _optional_instant(moment: datetime | None) -> str | None:
    return moment.isoformat() if moment is not None else None


def _moment(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _video(row: sqlite3.Row) -> VideoRecord:
    media = None
    if row["duration_ms"] is not None:
        media = MediaInfo(
            duration_ms=row["duration_ms"],
            container=row["container"],
            video_codec=row["video_codec"],
            width=row["width"],
            height=row["height"],
            fps=row["fps"],
            audio_codec=row["audio_codec"],
            sample_rate=row["sample_rate"],
            channels=row["channels"],
        )
    return VideoRecord(
        id=row["id"],
        path=Path(row["path"]),
        size_bytes=row["size_bytes"],
        mtime_ns=row["mtime_ns"],
        fingerprint=row["fingerprint"],
        media=media,
        created_at=_moment(row["created_at"]),
        updated_at=_moment(row["updated_at"]),
    )


def _job(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        id=row["id"],
        video_id=row["video_id"],
        question=row["question"],
        status=JobStatus(row["status"]),
        created_at=_moment(row["created_at"]),
        started_at=_moment(row["started_at"]) if row["started_at"] else None,
        finished_at=_moment(row["finished_at"]) if row["finished_at"] else None,
        error=row["error"],
    )


def _transcript(row: sqlite3.Row) -> Transcript:
    return Transcript(
        id=row["id"],
        video_id=row["video_id"],
        cache_key=row["cache_key"],
        model=row["model"],
        segment_count=row["segment_count"],
        first_ms=row["first_ms"],
        last_ms=row["last_ms"],
        created_at=_moment(row["created_at"]),
        completed_at=_moment(row["completed_at"]) if row["completed_at"] else None,
    )


def _task(row: sqlite3.Row) -> TaskRecord:
    return TaskRecord(
        id=row["id"],
        job_id=row["job_id"],
        agent=row["agent"],
        action=row["action"],
        status=TaskStatus(row["status"]),
        depends_on=tuple(json.loads(row["depends_on"])),
        dependency_mode=DependencyMode(row["dependency_mode"]),
        attempt=row["attempt"],
        max_attempts=row["max_attempts"],
        progress=row["progress"],
        started_at=_moment(row["started_at"]) if row["started_at"] else None,
        finished_at=_moment(row["finished_at"]) if row["finished_at"] else None,
        error=row["error"],
    )
