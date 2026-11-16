"""Turning one request into a job graph, and running that graph.

Owns the wiring between file identity, the store, the media stages and the
scheduler. Both entry points block, so they belong on a worker thread: ``submit``
while hashing the file, ``run`` until every node is terminal.

Does not render, does not parse command syntax, and does not decide how a file
should be analysed: the interface hands over an absolute path and the user's own
wording.
"""

from __future__ import annotations

from pathlib import Path

from ..asr import AsrStage
from ..config.schema import RuntimeLimits
from ..domain import Event, EventKind, JobRecord, NewTask, VideoRecord
from ..media import extract_audio, identify, probe
from ..storage import Storage, StorageError
from ..tasks import EventBus, Scheduler, TaskContext
from ..workspace import Workspace

GRAPH: tuple[NewTask, ...] = (
    NewTask(id="probe", agent="media", action="probe"),
    NewTask(id="audio", agent="media", action="extract_audio", depends_on=("probe",)),
    NewTask(id="transcribe", agent="asr", action="transcribe", depends_on=("audio",)),
)


class Pipeline:
    """One reference, from file on disk to a finished job record."""

    def __init__(
        self,
        store: Storage,
        bus: EventBus,
        runtime: RuntimeLimits,
        workspace: Workspace,
        asr: AsrStage | None = None,
    ) -> None:
        self._store = store
        self._bus = bus
        self._runtime = runtime
        self._workspace = workspace
        self._asr = asr
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
        handlers = {
            "probe": self._probe,
            "audio": self._audio,
            "transcribe": self._transcribe,
        }
        return self._scheduler.run(job, handlers)

    def _probe(self, context: TaskContext) -> None:
        """Write the stream layout onto the video, refusing one over the configured length.

        The length ceiling is checked before the record is written: a source the
        product will not analyse should not appear on the timeline as usable.
        """
        video = self._video(context)
        context.report(0.05, "打开容器")
        media = probe(video.path)
        ceiling = self._runtime.max_video_duration_seconds
        if media.duration_ms > ceiling * 1000:
            raise ValueError(f"时长 {media.duration_ms // 1000} 秒超过上限 {ceiling} 秒")
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

    def _audio(self, context: TaskContext) -> None:
        """Prepare the audio file once per video.

        A source without an audio track is a finished stage, not a failure: there
        is nothing to transcribe, and the job's conclusion is text-free for a
        reason the interface can state.
        """
        video = self._video(context)
        if video.media is not None and video.media.audio_codec is None:
            self._log(context, "没有音轨，跳过转录")
            context.report(1.0)
            return
        target = self._workspace.audio_path(video.id)
        if target.exists():
            context.report(1.0, "复用已有音轨")
            return
        context.report(0.0, "提取音轨")
        expected = video.media.duration_ms if video.media is not None else 0
        extract_audio(video.path, target, expected_ms=expected, progress=context.report)
        context.report(1.0)

    def _transcribe(self, context: TaskContext) -> None:
        """Produce the transcript, reusing every chunk a previous run finished."""
        video = self._video(context)
        if video.media is not None and video.media.audio_codec is None:
            context.report(1.0)
            return
        if self._asr is None:
            raise ValueError("未配置语音模型：asr")
        context.report(0.0, "开始转录")
        count = self._asr.run(video, context.report)
        self._log(context, f"转录完成：{count} 段")
        context.report(1.0)

    def _video(self, context: TaskContext) -> VideoRecord:
        video = self._store.video(context.job.video_id)
        if video is None:
            raise StorageError(f"视频不存在：{context.job.video_id}")
        return video

    def _log(self, context: TaskContext, message: str) -> None:
        """Publish one line for the interface; progress alone cannot say why a stage ended."""
        self._bus.publish(
            Event.stamp(
                EventKind.LOG_LINE,
                job_id=context.job.id,
                task_id=context.task.id,
                payload={"message": message},
            )
        )
