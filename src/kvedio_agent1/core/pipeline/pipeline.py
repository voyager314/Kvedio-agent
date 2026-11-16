"""Turning one request into a job graph, and running that graph.

Owns the wiring between file identity, the store, the media stages, the
supervisor and the scheduler. Both entry points block, so they belong on a worker
thread: ``submit`` while hashing the file, ``run`` until every node is terminal.

The graph holds only the nodes whose output is missing. Referencing a file for
the first time runs probe, audio and transcription; asking a second question
about the same file runs the answer alone, because the transcript is already
there.

Does not render, does not parse command syntax, and does not decide how a file
should be analysed: the interface hands over an absolute path and the user's own
wording.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..asr import AsrStage
from ..config.schema import RuntimeLimits
from ..domain import Answer, Event, EventKind, JobRecord, NewTask, VideoRecord
from ..media import extract_audio, identify, probe
from ..storage import Storage, StorageError
from ..supervisor import Answering
from ..tasks import EventBus, Scheduler, TaskContext
from ..workspace import Workspace


class Answerer(Protocol):
    """What the pipeline needs from the supervisor: one stored answer per job."""

    def answer(
        self, job: JobRecord, report: Callable[[float, str | None], None]
    ) -> Answer: ...


@dataclass(frozen=True)
class Stages:
    """What a pipeline needs beyond the store and the bus.

    A single value so adding a stage in a later milestone does not ripple through
    every caller's parameter list.
    """

    runtime: RuntimeLimits
    workspace: Workspace
    asr: AsrStage | None = None
    answering: Answering | Answerer | None = None


class Pipeline:
    """One reference, from file on disk to a finished job record."""

    def __init__(self, store: Storage, bus: EventBus, stages: Stages) -> None:
        self._store = store
        self._bus = bus
        self._stages = stages
        self._scheduler = Scheduler(
            store, bus, max_workers=stages.runtime.max_global_workers
        )

    def submit(self, path: Path, question: str) -> JobRecord:
        """Register the file, then ask the question about it.

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
        return self.ask(video.id, question)

    def ask(self, video_id: int, question: str) -> JobRecord:
        """Persist a question about a video already in the store, with the nodes it needs."""
        video = self._store.video(video_id)
        if video is None:
            raise StorageError(f"视频不存在：{video_id}")
        job = self._store.create_job(video_id, question, self._graph_for(video))
        self._bus.publish(
            Event.stamp(
                EventKind.JOB_QUEUED,
                job_id=job.id,
                payload={"question": question, "video_id": video_id},
            )
        )
        return job

    def run(self, job: JobRecord) -> JobRecord:
        """Execute the job's graph on the calling thread and return its final state."""
        handlers = {
            "probe": self._probe,
            "audio": self._audio,
            "transcribe": self._transcribe,
            "answer": self._answer,
        }
        return self._scheduler.run(job, handlers)

    def _graph_for(self, video: VideoRecord) -> tuple[NewTask, ...]:
        """The nodes whose output is missing, chained in the order they run."""
        tasks: list[NewTask] = []
        if video.media is None:
            tasks.append(NewTask(id="probe", agent="media", action="probe"))
        if self._needs_audio(video):
            tasks.append(
                NewTask(id="audio", agent="media", action="extract_audio", depends_on=_tail(tasks))
            )
        if self._needs_transcript(video):
            tasks.append(
                NewTask(id="transcribe", agent="asr", action="transcribe", depends_on=_tail(tasks))
            )
        tasks.append(
            NewTask(id="answer", agent="supervisor", action="answer", depends_on=_tail(tasks))
        )
        return tuple(tasks)

    def _needs_audio(self, video: VideoRecord) -> bool:
        if video.media is not None and video.media.audio_codec is None:
            return False
        return not self._stages.workspace.audio(video.id).exists()

    def _needs_transcript(self, video: VideoRecord) -> bool:
        if video.media is not None and video.media.audio_codec is None:
            return False
        if self._stages.asr is None:
            return True  # the stage reports the missing model instead of answering from nothing
        return not self._stages.asr.current(video)

    def _probe(self, context: TaskContext) -> None:
        """Write the stream layout onto the video, refusing one over the configured length.

        The length ceiling is checked before the record is written: a source the
        product will not analyse should not appear on the timeline as usable.
        """
        video = self._video(context)
        context.report(0.05, "打开容器")
        media = probe(video.path)
        ceiling = self._stages.runtime.max_video_duration_seconds
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
        is nothing to transcribe, and the answer stage will say so.
        """
        video = self._video(context)
        if video.media is not None and video.media.audio_codec is None:
            self._log(context, "没有音轨，跳过转录")
            context.report(1.0)
            return
        target = self._stages.workspace.audio(video.id)
        if target.exists():
            context.report(1.0, "复用已有音轨")
            return
        context.report(0.0, "提取音轨")
        expected = video.media.duration_ms if video.media is not None else 0
        decoded = extract_audio(video.path, target, expected_ms=expected, progress=context.report)
        if decoded <= 0:
            raise ValueError("音轨没有可用的采样")
        context.report(1.0)

    def _transcribe(self, context: TaskContext) -> None:
        """Produce the transcript, reusing every chunk a previous run finished."""
        video = self._video(context)
        if video.media is not None and video.media.audio_codec is None:
            context.report(1.0)
            return
        asr = self._stages.asr
        if asr is None:
            raise ValueError("未配置语音模型：asr")
        context.report(0.0, "开始转录")
        count = asr.run(video, context.report)
        self._log(context, f"转录完成：{count} 段")
        context.report(1.0)

    def _answer(self, context: TaskContext) -> None:
        """Answer from the transcript, citing only the moments retrieval returned."""
        video = self._video(context)
        if video.media is not None and video.media.audio_codec is None:
            raise ValueError("该视频没有音轨，无法作答")
        answering = self._stages.answering
        if answering is None:
            raise ValueError("未配置对话模型：supervisor")
        context.report(0.0, "开始作答")
        answer = answering.answer(context.job, context.report)
        context.report(1.0, f"结论引用 {len(answer.citations)} 处")

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


def _tail(tasks: list[NewTask]) -> tuple[str, ...]:
    """The node just added, which the next one depends on; empty for the first."""
    return (tasks[-1].id,) if tasks else ()
