"""Answering one question about one video from the evidence it actually has.

The transcript and the scene descriptions are separate products of separate
branches, and either can be missing. This module first takes stock of what the
store holds, offers the planner only the tools whose data exists, and writes a
conclusion from that subset. A branch that failed narrows the answer; it does not
cancel it, and the conclusion names the gap.

A job can be answered twice: once from the branch that arrived first, and again
after the other branch settles. The second pass receives the first conclusion and
the new evidence, so it integrates and corrects rather than starting over. Only
the latest text is stored; each replacement increments its revision.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ...tools import TOOLS, Evidence, Tool, ToolBox
from ..domain import (
    Answer,
    Event,
    EventKind,
    JobRecord,
    TaskRecord,
    TaskStatus,
    VideoRecord,
)
from ..llm import Chat, Embedder
from ..memory import VideoMemory
from ..storage import Storage, StorageError
from ..tasks import EventBus, stop_if
from .planner import Plan, make_plan
from .synthesis import Conclusion, write, write_absence

Report = Callable[[float, str | None], None]


@dataclass(frozen=True)
class Sources:
    """Which evidence one video can offer, and what is absent.

    ``transcript`` and ``scenes`` are the two citable sources; events group the
    scenes and are never cited on their own. ``limits`` names every absent source
    for the prompt. ``pending`` holds the sources still being produced, which a
    preliminary answer discloses without treating them as failures. ``missing``
    holds the sources this job tried to fill and could not, which every conclusion
    has to disclose.
    """

    transcript: bool
    scenes: bool
    events: bool
    limits: tuple[str, ...]
    pending: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def any(self) -> bool:
        """Whether at least one citable source exists."""
        return self.transcript or self.scenes

    @property
    def tools(self) -> tuple[Tool, ...]:
        """The catalogue entries whose backing data exists, in catalogue order."""
        names: set[str] = set()
        if self.transcript:
            names.update(("overview", "search", "read_range"))
        if self.scenes:
            names.add("scenes")
        if self.events:
            names.add("events")
        return tuple(tool for tool in TOOLS if tool.name in names)


class Answering:
    """Runs one question through plan, retrieval and synthesis, then stores the result.

    Holds no prompt of its own: planning and writing live in separate modules so
    each can be tested alone. Every step is announced on the bus, which is how the
    interface shows the trace — a conclusion is easier to trust when the searches
    behind it are visible.

    A video with neither words nor pictures has nothing to answer from and raises.
    Any narrower set of sources answers from what is there. ``preliminary`` marks
    the first pass, written while the other branch is still producing; its stored
    stage is ``partial``.
    """

    def __init__(
        self,
        store: Storage,
        bus: EventBus,
        chat: Chat,
        model: str,
        embedder: Embedder | None = None,
    ) -> None:
        self._store = store
        self._bus = bus
        self._chat = chat
        self._model = model
        self._embedder = embedder

    def answer(
        self,
        job: JobRecord,
        report: Report,
        *,
        cancelled: Callable[[], bool] | None = None,
        preliminary: bool = False,
    ) -> Answer:
        """Plan, retrieve, write, check, store — or raise when no source exists.

        ``preliminary`` writes the first conclusion from the branch that is
        already usable and does not stream it: the final pass owns the live text.
        A final pass reads the stored preliminary and hands it to the writer as a
        prior, so new evidence supplements the earlier answer instead of
        replacing it without explanation.
        """
        stop_if(cancelled)
        video = self._store.video(job.video_id)
        if video is None:
            raise StorageError(f"视频不存在：{job.video_id}")
        sources = self._sources(video, job)
        if not sources.any:
            raise ValueError(_nothing(sources))
        previous = None if preliminary else self._store.answer(job.id)

        toolbox = ToolBox(
            self._store,
            video,
            VideoMemory(self._store, video, self._embedder),
            tools=sources.tools,
        )
        report(0.05, "生成计划")
        plan = make_plan(self._chat, job.question, toolbox.tools)
        self._log(job, plan.note)
        if sources.limits:
            self._log(job, "证据范围：" + "；".join(sources.limits))

        evidence = self._gather(job, toolbox, plan, report, cancelled)
        if not evidence:
            report(0.6, "没有检索到可用片段")
            self._log(job, "没有检索到可用片段")
            if previous is not None:
                self._log(job, "新证据没有增加，沿用上一版结论")
                conclusion = Conclusion(
                    text=_prior(previous) or previous.text,
                    citations=previous.citations,
                )
            else:
                conclusion = Conclusion(
                    text=write_absence(self._chat, question=job.question, limits=sources.limits),
                    citations=(),
                )
        else:
            report(0.6, "撰写结论")
            conclusion = write(
                self._chat,
                question=job.question,
                evidence=evidence,
                duration_ms=video.media.duration_ms if video.media is not None else 0,
                limits=sources.limits,
                previous=_prior(previous),
                on_delta=None if preliminary else self._delta(job),
            )
        report(0.95, "保存结论")
        answer = self._store.record_answer(
            job.id,
            video.id,
            job.question,
            _disclosed(conclusion.text, sources, preliminary=preliminary),
            self._model,
            conclusion.citations,
            stage="partial" if preliminary else "final",
        )
        self._result(job, answer, preliminary=preliminary, pending=sources.pending)
        self._log(job, f"结论引用了 {len(answer.citations)} 处时间点")
        return answer

    def _sources(self, video: VideoRecord, job: JobRecord) -> Sources:
        """Inventory the evidence on disk and why a branch has not delivered.

        A source is absent either because the file has no such track, because the
        branch is still working, or because a node of this job failed. Only the
        last kind enters ``missing``: a running branch is disclosed as pending,
        and a track the file does not have is a limit, not a failure.
        """
        tasks = {task.id: task for task in self._store.tasks_for_job(job.id)}
        transcript = self._store.transcript(video.id)
        scenes = self._store.scenes(video.id)
        events = self._store.events(video.id)
        spoken = transcript is not None and transcript.ready and transcript.segment_count > 0
        limits: list[str] = []
        pending: list[str] = []
        missing: list[str] = []

        if video.media is None:
            reason = _with_cause("媒体信息不可用", _cause_of(tasks, "probe"))
            limits.append(reason)
            if _attempted(tasks, "probe"):
                missing.append(reason)
            return Sources(False, False, False, tuple(limits), (), tuple(missing))

        if not spoken:
            if video.media.audio_codec is None:
                limits.append("该视频没有音轨，没有文字转录")
            elif _producing(tasks, "transcribe"):
                limits.append("文字转录仍在生成")
                pending.append("文字转录")
            elif transcript is not None and transcript.ready:
                limits.append("转录为空，没有可用台词")
            elif transcript is not None and transcript.segment_count > 0:
                reason = _with_cause("文字转录不完整", _cause_of(tasks, "transcribe"))
                limits.append(reason)
                missing.append(reason)
            else:
                reason = _with_cause("文字转录不可用", _cause_of(tasks, "transcribe"))
                limits.append(reason)
                if _attempted(tasks, "transcribe"):
                    missing.append(reason)

        if not scenes:
            if video.media.video_codec is None:
                limits.append("该视频没有画面，没有场景描述")
            elif _producing(tasks, "vision"):
                limits.append("画面描述仍在生成")
                pending.append("画面描述")
            else:
                reason = _with_cause("画面描述不可用", _cause_of(tasks, "vision"))
                limits.append(reason)
                if _attempted(tasks, "vision"):
                    missing.append(reason)
        elif _producing(tasks, "vision"):
            limits.append("画面描述仍在生成，当前只有部分场景")
            pending.append("画面描述")
        elif _attempted(tasks, "vision"):
            reason = _with_cause("画面描述不完整", _cause_of(tasks, "vision"))
            limits.append(reason)
            missing.append(reason)

        return Sources(
            spoken,
            bool(scenes),
            bool(events) and bool(scenes),
            tuple(limits),
            tuple(pending),
            tuple(missing),
        )

    def _gather(
        self,
        job: JobRecord,
        toolbox: ToolBox,
        plan: Plan,
        report: Report,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[Evidence, ...]:
        """Run the plan, keep the evidence in timeline order, and announce each step."""
        collected: dict[tuple[int, int, str], Evidence] = {}
        total = max(1, len(plan.steps))
        for index, step in enumerate(plan.steps, start=1):
            stop_if(cancelled)
            result = toolbox.call(step.tool, step.arguments)
            self._log(job, f"{step.why} · {result.note}" if step.why else result.note)
            for item in result.evidence:
                # The same stretch may arrive as a scene and as speech; both are
                # evidence, and dropping one would drop the only citable copy.
                collected.setdefault((item.start_ms, item.end_ms, item.label), item)
            report(0.1 + 0.45 * index / total, None)
        return tuple(collected[key] for key in sorted(collected))

    def _result(
        self, job: JobRecord, answer: Answer, *, preliminary: bool, pending: Sequence[str]
    ) -> None:
        """Announce the stored text as partial or final, with the gap it still has."""
        kind = EventKind.RESULT_PARTIAL if preliminary else EventKind.RESULT_FINAL
        self._bus.publish(
            Event.stamp(
                kind,
                job_id=job.id,
                payload={
                    "answer_id": answer.id,
                    "stage": answer.stage,
                    "revision": answer.revision,
                    "citations": len(answer.citations),
                    "pending": list(pending),
                },
            )
        )

    def _delta(self, job: JobRecord) -> Callable[[str], None]:
        """Publish each piece of the answer as it is written, so the panel can show it."""

        def publish(piece: str) -> None:
            self._bus.publish(
                Event.stamp(
                    EventKind.ANSWER_DELTA, job_id=job.id, payload={"delta": piece}
                )
            )

        return publish

    def _log(self, job: JobRecord, message: str) -> None:
        """Publish one line of the trace; the interface renders it in the chat panel."""
        self._bus.publish(
            Event.stamp(EventKind.LOG_LINE, job_id=job.id, payload={"message": message})
        )


def _attempted(tasks: Mapping[str, TaskRecord], name: str) -> bool:
    """Whether this job ran the node and it did not succeed."""
    task = tasks.get(name)
    return task is not None and task.status is not TaskStatus.SUCCEEDED


def _producing(tasks: Mapping[str, TaskRecord], name: str) -> bool:
    """Whether the node exists and has not reached a terminal status yet."""
    task = tasks.get(name)
    return task is not None and task.status in (
        TaskStatus.PENDING,
        TaskStatus.READY,
        TaskStatus.RUNNING,
    )


def _cause_of(tasks: Mapping[str, TaskRecord], name: str) -> str | None:
    """The failure behind a node, or ``None`` when the node never ran or succeeded."""
    task = tasks.get(name)
    if task is None or task.status is TaskStatus.SUCCEEDED:
        return None
    return _cause(tasks, task)


def _cause(tasks: Mapping[str, TaskRecord], task: TaskRecord) -> str | None:
    """The first failure a node inherits from its ancestry, or its own error.

    A skipped node records only that a dependency failed; walking back to the
    failing ancestor turns "依赖未成功，已跳过" into the reason the user needs.
    """
    seen: set[str] = set()
    pending = [task]
    while pending:
        current = pending.pop()
        if current.id in seen:
            continue
        seen.add(current.id)
        if current.status in (TaskStatus.FAILED, TaskStatus.CANCELED) and current.error:
            return current.error
        pending.extend(tasks[name] for name in current.depends_on if name in tasks)
    return task.error


def _with_cause(label: str, cause: str | None) -> str:
    """Attach a failure message to the name of the source it cost."""
    return f"{label}（{cause}）" if cause else label


def _prior(previous: Answer | None) -> str | None:
    """The earlier text without the system note appended to it, or ``None``."""
    if previous is None:
        return None
    return previous.text.split("\n\n证据范围：", 1)[0].strip() or None


def _nothing(sources: Sources) -> str:
    """The failure for a video with no citable source, naming each reason."""
    detail = "；".join(sources.limits)
    if not detail:
        return "没有可用的转录或画面描述，无法作答"
    return f"没有可用的转录或画面描述，无法作答：{detail}"


def _disclosed(text: str, sources: Sources, *, preliminary: bool) -> str:
    """Append the evidence gap, so a provisional or partial conclusion says so."""
    notes: list[str] = []
    if preliminary and sources.pending:
        notes.append(f"初步结论，{'；'.join(sources.pending)}仍在生成，完成后会补充")
    elif sources.pending:
        notes.append(f"{'；'.join(sources.pending)}尚未完成")
    if sources.missing:
        notes.append(f"证据缺失——{'；'.join(sources.missing)}")
    if not notes:
        return text
    return f"{text.rstrip()}\n\n证据范围：{'；'.join(notes)}。"
