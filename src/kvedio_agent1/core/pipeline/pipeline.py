"""Turning one request into a job graph, and running that graph.

Owns the wiring between file identity, the store, the media stages, the
supervisor and the scheduler. Both entry points block, so they belong on a worker
thread: ``submit`` while hashing the file, ``run`` until every node is terminal.

The graph holds only the nodes whose output is missing. Referencing a file for
the first time runs probe, then two branches that read the source independently:
audio, transcription and memory on one side, scene description and events on the
other. Asking a second question about the same file runs the answer alone,
because both branches are already there.

Does not render, does not parse command syntax, and does not decide how a file
should be analysed: the interface hands over an absolute path and the user's own
wording.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..asr import AsrStage, citable_transcript
from ..config.schema import RuntimeLimits
from ..domain import (
    Answer,
    DependencyMode,
    Event,
    EventKind,
    JobRecord,
    NewTask,
    TaskRecord,
    Transcript,
    VideoRecord,
)
from ..media import extract_audio, identify, needs_level_fix, probe
from ..memory import MemoryBuilder
from ..storage import Storage, StorageError
from ..supervisor import Answering, Route
from ..tasks import (
    Cancellation,
    Cancelled,
    EventBus,
    Scheduler,
    TaskContext,
    TaskTimeout,
    stage_budget,
)
from ..vision import EventBuilder, VisionBuilder
from ..workspace import Workspace


class Answerer(Protocol):
    """What the pipeline needs from the supervisor: one stored answer per pass."""

    def answer(
        self,
        job: JobRecord,
        report: Callable[[float, str | None], None],
        *,
        cancelled: Callable[[], bool] | None = None,
        preliminary: bool = False,
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
    memory: MemoryBuilder | None = None
    vision: VisionBuilder | None = None
    events: EventBuilder | None = None
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
        self._cancellations: dict[str, Cancellation] = {}
        self._tokens = threading.Lock()

    def reference(self, path: Path) -> VideoRecord:
        """Register the file the user pointed at, without queueing any work.

        The analysis is a property of the file, not of the wording that
        accompanied it, so it is a step of its own: what the user said is asked
        afterwards, once the products are there to answer from.
        """
        identity = identify(path)
        return self._store.register_video(
            identity.path,
            size_bytes=identity.size_bytes,
            mtime_ns=identity.mtime_ns,
            fingerprint=identity.fingerprint,
        )

    def submit(
        self,
        path: Path,
        question: str,
        route: Route | None = None,
        *,
        analyze_only: bool = False,
    ) -> JobRecord | None:
        """Register the file, then ask the question about it.

        ``route`` decides which branches the job asks for; ``None`` keeps the
        full two-branch analysis. With ``analyze_only`` the graph stops after the
        branches: the question is answered by a later job, from what they stored.
        Raises ``OSError`` when the file cannot be read and ``StorageError`` when
        the database refuses the rows; no job exists afterwards in either case.
        """
        video = self.reference(path)
        return self.ask(video.id, question, route, analyze_only=analyze_only)

    def ask(
        self,
        video_id: int,
        question: str,
        route: Route | None = None,
        *,
        analyze_only: bool = False,
    ) -> JobRecord | None:
        """Persist the work this video still needs, and its question if it has one.

        Returns ``None`` when there is nothing left to run — an analysis-only
        request for a video whose products are all present costs no job, so the
        caller can ask its question straight away.
        """
        video = self._store.video(video_id)
        if video is None:
            raise StorageError(f"视频不存在：{video_id}")
        decided = route or Route("ask_video", ("asr", "video"), "默认分析两路")
        tasks = self._graph_for(video, decided.agents, analyze_only=analyze_only)
        if not tasks:
            return None
        job = self._store.create_job(
            video_id,
            question,
            tasks,
            agents=decided.agents,
        )
        self._bus.publish(
            Event.stamp(
                EventKind.JOB_QUEUED,
                job_id=job.id,
                payload={
                    "question": question,
                    "video_id": video_id,
                    "route": decided.kind,
                    "agents": list(decided.agents),
                },
            )
        )
        return job

    def run(self, job: JobRecord) -> JobRecord:
        """Execute the job's graph on the calling thread and return its final state."""
        signal = Cancellation()
        with self._tokens:
            self._cancellations[job.id] = signal
        handlers = {
            "probe": self._probe,
            "audio": self._audio,
            "transcribe": self._transcribe,
            "memory": self._memory,
            "vision": self._vision,
            "events": self._events,
            "answer_preliminary": self._answer_early,
            "answer": self._answer,
        }

        def budget(task: TaskRecord) -> float:
            video = self._store.video(job.video_id)
            duration = (
                video.media.duration_ms if video is not None and video.media is not None else 0
            )
            return stage_budget(task.action, duration, self._stages.runtime)

        try:
            return self._scheduler.run(job, handlers, signal, budget)
        finally:
            with self._tokens:
                self._cancellations.pop(job.id, None)

    def cancel(self, job_id: str) -> bool:
        """Ask a running job to stop at its next safe point.

        Returns whether the job was running: a finished or unknown job has nothing
        to cancel, and saying so is better than pretending it worked.
        """
        with self._tokens:
            signal = self._cancellations.get(job_id)
        if signal is None:
            return False
        signal.cancel()
        return True

    def transcript(self, video: VideoRecord) -> Transcript | None:
        """The transcript the configured speech inputs would produce, or ``None``.

        A stored transcript built under another model, address, chunk length or
        audio profile is withheld: its words describe a different decode, so an
        answer citing them would point at moments the recording does not carry.
        """
        return citable_transcript(self._store, self._stages.asr, video)

    def _graph_for(
        self, video: VideoRecord, agents: tuple[str, ...], *, analyze_only: bool = False
    ) -> tuple[NewTask, ...]:
        """The nodes the requested branches still need, meeting at the answer.

        Only a branch named in ``agents`` is built. Transcription and picture
        description read the same file and neither consumes the other's output,
        so they hang from the probe as separate branches. The answer joins the
        last node of each branch under ``PARTIAL``: a branch that failed costs its
        evidence, not the conclusion.

        ``analyze_only`` stops the graph after those branches. A reference is
        analysed before anything is asked of it, so it has no answer node and no
        preliminary: both would need a question the user has not asked yet.

        When both branches are requested and can still deliver a citable source,
        a preliminary answer runs on the first one to arrive. The final answer
        waits for the preliminary and both branches: it reads the earlier text as
        a prior, so the two passes cannot race to overwrite each other.
        """
        tasks: list[NewTask] = []
        source: tuple[str, ...] = ()
        if video.media is None:
            tasks.append(NewTask(id="probe", agent="media", action="probe"))
            source = ("probe",)
        spoken = self._spoken_branch(video, source, tasks) if "asr" in agents else ()
        seen = self._visual_branch(video, source, tasks) if "video" in agents else ()
        if analyze_only:
            return tuple(tasks)
        early = (
            self._early_graph(video, spoken, seen, tasks)
            if "asr" in agents and "video" in agents
            else ()
        )
        joined = _join(spoken, seen, early)
        tasks.append(
            NewTask(
                id="answer",
                agent="supervisor",
                action="answer",
                depends_on=joined,
                dependency_mode=DependencyMode.PARTIAL,
            )
        )
        return tuple(tasks)

    def _early_graph(
        self,
        video: VideoRecord,
        spoken: tuple[str, ...],
        seen: tuple[str, ...],
        tasks: list[NewTask],
    ) -> tuple[str, ...]:
        """Add the preliminary answer when the two branches will arrive apart.

        Returns its id, or an empty tuple when one source already covers the video
        or only one branch can ever contribute. A source already in the store lets
        the preliminary run at once; otherwise it waits for either branch to
        succeed.
        """
        spoken_ready = self._spoken_ready(video)
        visual_ready = self._visual_ready(video)
        if spoken_ready and visual_ready:
            return ()
        if not (self._expects_speech(video) and self._expects_visual(video)):
            return ()
        if video.media is None and (spoken_ready or visual_ready):
            return ()
        early: tuple[str, ...] = ()
        if not (spoken_ready or visual_ready):
            early = _join(spoken, seen)
        tasks.append(
            NewTask(
                id="answer_preliminary",
                agent="supervisor",
                action="answer_preliminary",
                depends_on=early,
                dependency_mode=DependencyMode.ANY,
            )
        )
        return ("answer_preliminary",)

    def _spoken_ready(self, video: VideoRecord) -> bool:
        """Whether a finished transcript with words is here under the configured inputs."""
        transcript = self.transcript(video)
        return transcript is not None and transcript.ready and transcript.segment_count > 0

    def _visual_ready(self, video: VideoRecord) -> bool:
        """Whether at least one scene description is already in the store."""
        return bool(self._store.scenes(video.id))

    def _expects_speech(self, video: VideoRecord) -> bool:
        """Whether a citable transcript is here or a transcribe node will run."""
        if self._spoken_ready(video):
            return True
        if self._stages.asr is None:
            return False
        if video.media is not None and video.media.audio_codec is None:
            return False
        return self._needs_transcript(video)

    def _expects_visual(self, video: VideoRecord) -> bool:
        """Whether citable scenes are here or a vision node will run."""
        if self._visual_ready(video):
            return True
        if self._stages.vision is None:
            return False
        if video.media is not None and video.media.video_codec is None:
            return False
        return self._needs_vision(video)

    def _spoken_branch(
        self, video: VideoRecord, source: tuple[str, ...], tasks: list[NewTask]
    ) -> tuple[str, ...]:
        """Append the audio, transcript and memory nodes; return the branch's last node."""
        tail = source
        if self._needs_audio(video):
            tasks.append(
                NewTask(id="audio", agent="media", action="extract_audio", depends_on=tail)
            )
            tail = ("audio",)
        transcribe = self._needs_transcript(video)
        if transcribe:
            tasks.append(
                NewTask(id="transcribe", agent="asr", action="transcribe", depends_on=tail)
            )
            tail = ("transcribe",)
        if self._needs_memory(video, after_transcript=transcribe):
            tasks.append(NewTask(id="memory", agent="memory", action="build", depends_on=tail))
            tail = ("memory",)
        return tail

    def _visual_branch(
        self, video: VideoRecord, source: tuple[str, ...], tasks: list[NewTask]
    ) -> tuple[str, ...]:
        """Append the scene and event nodes; return the branch's last node."""
        tail = source
        vision = self._needs_vision(video)
        if vision:
            tasks.append(NewTask(id="vision", agent="vision", action="describe", depends_on=tail))
            tail = ("vision",)
        if self._needs_events(video, after_vision=vision):
            tasks.append(NewTask(id="events", agent="events", action="extract", depends_on=tail))
            tail = ("events",)
        return tail

    def _needs_audio(self, video: VideoRecord) -> bool:
        if video.media is not None and video.media.audio_codec is None:
            return False
        target = self._stages.workspace.audio(video.id)
        return not target.exists() or needs_level_fix(target)

    def _needs_memory(self, video: VideoRecord, *, after_transcript: bool) -> bool:
        builder = self._stages.memory
        if builder is None:
            return False
        if video.media is not None and video.media.audio_codec is None:
            return False
        if after_transcript:
            return True  # the transcript is about to change, so the memory must follow it
        transcript = self.transcript(video)
        if transcript is None or not transcript.ready:
            return False  # the map would describe words this configuration did not produce
        return not builder.current(video)

    def _needs_vision(self, video: VideoRecord) -> bool:
        builder = self._stages.vision
        if builder is None:
            return False
        return not builder.current(video)

    def _needs_events(self, video: VideoRecord, *, after_vision: bool) -> bool:
        builder = self._stages.events
        if builder is None:
            return False
        if after_vision:
            return True  # the shots are about to change, so the events must follow them
        return not builder.current(video)

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
        context.check()
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

        A file left by an earlier run is reused only when its level is already
        usable: a quiet one was decoded before the level was raised and would cost
        the transcript again. A source without an audio track is a finished stage,
        not a failure: there is nothing to transcribe, and the answer stage will
        say so.
        """
        context.check()
        video = self._video(context)
        if video.media is not None and video.media.audio_codec is None:
            self._log(context, "没有音轨，跳过转录")
            context.report(1.0)
            return
        target = self._stages.workspace.audio(video.id)
        if target.exists() and not needs_level_fix(target):
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
        context.check()
        video = self._video(context)
        if video.media is not None and video.media.audio_codec is None:
            context.report(1.0)
            return
        asr = self._stages.asr
        if asr is None:
            raise ValueError("未配置语音模型：asr")
        context.report(0.0, "开始转录")
        count = asr.run(video, context.report, cancelled=lambda: context.cancelled)
        self._log(context, f"转录完成：{count} 段")
        context.report(1.0)

    def _memory(self, context: TaskContext) -> None:
        """Build the coverage map and the vector index, skipping what is already there."""
        context.check()
        builder = self._stages.memory
        if builder is None:
            context.report(1.0)
            return
        video = self._video(context)
        transcript = self.transcript(video)
        if transcript is None or not transcript.ready:
            self._log(context, "没有可用的转录，跳过记忆")
            context.report(1.0)
            return
        context.report(0.0, "建立记忆")
        stats = builder.build(video, context.report, cancelled=lambda: context.cancelled)
        self._log(context, f"记忆完成：覆盖图 {stats.blocks} 块 · 向量 {stats.vectors} 条")
        context.report(1.0)

    def _vision(self, context: TaskContext) -> None:
        """Cut the recording into scenes and describe each one, skipping described ones."""
        context.check()
        builder = self._stages.vision
        if builder is None:
            context.report(1.0)
            return
        video = self._video(context)
        if video.media is not None and video.media.video_codec is None:
            self._log(context, "没有视频轨，跳过场景")
            context.report(1.0)
            return
        context.report(0.0, "识别场景")
        stats = builder.build(video, context.report, cancelled=lambda: context.cancelled)
        self._log(context, f"场景完成：{stats.described}/{stats.scenes} 个已描述")
        context.report(1.0)

    def _events(self, context: TaskContext) -> None:
        """Name the events the shots belong to, so a reader can find a stretch."""
        context.check()
        builder = self._stages.events
        if builder is None:
            context.report(1.0)
            return
        video = self._video(context)
        context.report(0.0, "抽取事件")
        stats = builder.build(video, context.report, cancelled=lambda: context.cancelled)
        self._log(context, f"事件完成：{stats.events} 个")
        context.report(1.0)

    def _answer_early(self, context: TaskContext) -> None:
        """Publish a conclusion from the first branch that produced usable evidence.

        A preview is an optimization, not a deliverable: a failed call is logged
        and the task still succeeds, so a preview that never reached the user does
        not turn the final job into a partial one. Cancellation is not swallowed.
        """
        try:
            context.check()
        except Cancelled:
            raise
        except TaskTimeout as error:
            self._log(context, f"初步结论超时：{error}")
            return
        answering = self._stages.answering
        if answering is None:
            self._log(context, "未配置对话模型，跳过初步结论")
            return
        try:
            answer = answering.answer(
                context.job,
                context.report,
                cancelled=lambda: context.cancelled,
                preliminary=True,
            )
        except Cancelled:
            raise
        except TaskTimeout as error:
            self._log(context, f"初步结论超时：{error}")
            return
        except Exception as error:  # a preview must not decide the job's fate
            self._log(context, f"初步结论未生成：{error}")
            return
        self._log(context, f"初步结论引用 {len(answer.citations)} 处")

    def _answer(self, context: TaskContext) -> None:
        """Answer from whatever evidence the video produced, naming any gap in the conclusion."""
        context.check()
        answering = self._stages.answering
        if answering is None:
            raise ValueError("未配置对话模型：supervisor")
        context.report(0.0, "开始作答")
        answering.answer(
            context.job,
            context.report,
            cancelled=lambda: context.cancelled,
            preliminary=False,
        )

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


def _join(*tails: tuple[str, ...]) -> tuple[str, ...]:
    """The distinct last nodes of the branches, which the answer waits for."""
    return tuple(dict.fromkeys(name for tail in tails for name in tail))
