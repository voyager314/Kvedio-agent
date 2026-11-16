"""Answering one question about one video from its transcript."""

from __future__ import annotations

from collections.abc import Callable

from ...tools import Evidence, ToolBox
from ..domain import Answer, Event, EventKind, JobRecord
from ..llm import Chat, Embedder
from ..memory import VideoMemory
from ..storage import Storage, StorageError
from ..tasks import EventBus, stop_if
from .planner import Plan, make_plan
from .synthesis import Conclusion, write, write_absence

Report = Callable[[float, str | None], None]


class Answering:
    """Runs one question through plan, retrieval and synthesis, then stores the result.

    Holds no prompt of its own: planning and writing live in separate modules so
    each can be tested alone. Every step is announced on the bus, which is how the
    interface shows the trace — a conclusion is easier to trust when the searches
    behind it are visible.
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
        self, job: JobRecord, report: Report, *, cancelled: Callable[[], bool] | None = None
    ) -> Answer:
        """Plan, retrieve, write, check, store — or raise, leaving no answer behind."""
        stop_if(cancelled)
        video = self._store.video(job.video_id)
        if video is None:
            raise StorageError(f"视频不存在：{job.video_id}")
        transcript = self._store.transcript(video.id)
        if transcript is None or not transcript.ready:
            raise ValueError("没有可用的转录，无法作答")

        toolbox = ToolBox(self._store, video, VideoMemory(self._store, video, self._embedder))
        report(0.05, "生成计划")
        plan = make_plan(self._chat, job.question, toolbox.tools)
        self._log(job, plan.note)

        evidence = self._gather(job, toolbox, plan, report, cancelled)
        if not evidence:
            report(0.6, "没有检索到可用片段")
            self._log(job, "没有检索到可用片段")
            conclusion = Conclusion(
                text=write_absence(self._chat, question=job.question), citations=()
            )
        else:
            report(0.6, "撰写结论")
            conclusion = write(
                self._chat,
                question=job.question,
                evidence=evidence,
                duration_ms=video.media.duration_ms if video.media is not None else 0,
                on_delta=self._delta(job),
            )
        report(0.95, "保存结论")
        answer = self._store.record_answer(
            job.id, video.id, job.question, conclusion.text, self._model, conclusion.citations
        )
        self._log(job, f"结论引用了 {len(answer.citations)} 处时间点")
        return answer

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
