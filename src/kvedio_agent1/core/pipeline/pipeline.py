"""Turning one request into a job graph, and running that graph.

Owns the wiring between file identity, the store, the media probe and the
scheduler. Both entry points block, so they belong on a worker thread: ``submit``
while hashing the file, ``run`` until every node is terminal.

Does not render, does not parse command syntax, and does not decide how a file
should be analysed: the interface hands over an absolute path and the user's own
wording.
"""

from __future__ import annotations

from pathlib import Path

from ..config.schema import RuntimeLimits
from ..domain import Event, EventKind, JobRecord, NewTask
from ..media import identify, probe
from ..storage import Storage, StorageError
from ..tasks import EventBus, Scheduler, TaskContext

GRAPH: tuple[NewTask, ...] = (NewTask(id="probe", agent="media", action="probe"),)


class Pipeline:
    """One reference, from file on disk to a finished job record."""

    def __init__(self, store: Storage, bus: EventBus, runtime: RuntimeLimits) -> None:
        self._store = store
        self._bus = bus
        self._runtime = runtime
        self._scheduler = Scheduler(store, bus, max_workers=runtime.max_global_workers)

    def submit(self, path: Path, question: str) -> JobRecord:
        """Register the file, persist the job and its graph, and announce it.

        Raises ``OSError`` when the file cannot be read and ``StorageError`` when
        the database refuses the rows; no job exists afterwards in either case.
        """
        identity = identify(path)
        video = self._store.register_video(
            identity.path,
            size_bytes=identity.size_bytes,
            mtime_ns=identity.mtime_ns,
            fingerprint=identity.fingerprint,
        )
        job = self._store.create_job(video.id, question, GRAPH)
        self._bus.publish(
            Event.stamp(
                EventKind.JOB_QUEUED,
                job_id=job.id,
                payload={"question": question, "video_id": video.id},
            )
        )
        return job

    def run(self, job: JobRecord) -> JobRecord:
        """Execute the job's graph on the calling thread and return its final state."""
        return self._scheduler.run(job, {"probe": self._probe})

    def _probe(self, context: TaskContext) -> None:
        """Write the stream layout onto the video, refusing one over the configured length.

        The length ceiling is checked before the record is written: a source the
        product will not analyse should not appear on the timeline as usable.
        """
        video = self._store.video(context.job.video_id)
        if video is None:
            raise StorageError(f"视频不存在：{context.job.video_id}")
        context.report(0.05, "打开容器")
        media = probe(video.path)
        ceiling = self._runtime.max_video_duration_seconds
        if media.duration_ms > ceiling * 1000:
            raise ValueError(
                f"时长 {media.duration_ms // 1000} 秒超过上限 {ceiling} 秒"
            )
        context.report(0.7, "写入流信息")
        self._store.record_media(video.id, media)
        self._bus.publish(
            Event.stamp(
                EventKind.VIDEO_REGISTERED,
                job_id=context.job.id,
                task_id=context.task.id,
                payload={"video_id": video.id, "duration_ms": media.duration_ms},
            )
        )
        context.report(1.0)
