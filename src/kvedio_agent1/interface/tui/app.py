"""The workbench: four panels fed by task events, and one input line.

The interface thread is the only writer of the widget tree. Worker threads
publish events; a timer drains them here and renders. No panel reaches into a
worker's memory, which is what keeps a long probe from freezing the screen.

Store reads on this thread are single-row queries: short enough not to stall the
pump, and they keep the panels showing what the store holds instead of a second
copy that can drift from it.
"""

from __future__ import annotations

import queue
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Header, Input, RichLog, Static

from ...core.config import Loaded, Role, is_usable
from ...core.domain import (
    Answer,
    Citation,
    Event,
    EventKind,
    JobRecord,
    JobStatus,
    TaskRecord,
    VideoRecord,
    format_moment,
)
from ...core.pipeline import Pipeline
from ...core.storage import Storage
from ...core.tasks import EventBus
from ..submission import USAGE, Command, Reference, parse_submission
from .format import (
    agent_label,
    format_container,
    format_duration,
    format_progress,
    format_size,
    format_tokens,
    job_status_label,
    model_label,
    role_label,
    task_status_label,
    usage_kind_label,
)

HELP = """\
/help              显示本说明
/model             列出各角色的模型与端点
/cancel            取消正在跑的作业，并清空排队
/usage             列出各模型累计调用的用量
@路径 问题         引用视频并分析，路径含空格加引号
直接输入问题       就当前视频追问，结论带时间戳
/quit              退出"""

_TASK_COLUMNS = ("task", "agent", "action", "status", "progress")
_TASK_HEADERS = ("任务", "角色", "动作", "状态", "进度")

_TASK_EVENTS = frozenset(
    {
        EventKind.TASK_READY,
        EventKind.TASK_STARTED,
        EventKind.TASK_SUCCEEDED,
        EventKind.TASK_FAILED,
        EventKind.TASK_SKIPPED,
        EventKind.TASK_CANCELED,
    }
)

_JOB_END_EVENTS = frozenset(
    {
        EventKind.JOB_SUCCEEDED,
        EventKind.JOB_PARTIAL,
        EventKind.JOB_FAILED,
        EventKind.JOB_CANCELED,
    }
)


class Workbench(App[None]):
    """The terminal workbench: chat, tasks, timeline and evidence over one database."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #body {
        height: 1fr;
    }
    #chat-column {
        width: 3fr;
        padding: 0 1;
    }
    #side-column {
        width: 2fr;
        padding: 0 1;
    }
    #chat {
        height: 1fr;
        border: round $primary;
        padding: 0 1;
    }
    #prompt {
        border: round $accent;
    }
    #answer {
        display: none;
        height: auto;
        max-height: 6;
        border: round $accent;
        padding: 0 1;
    }
    #tasks {
        height: 2fr;
        border: round $primary;
    }
    #timeline {
        height: 6;
        border: round $primary;
        padding: 0 1;
    }
    #evidence {
        height: 1fr;
        border: round $primary;
        padding: 0 1;
    }
    #status {
        height: 1;
        background: $panel;
        color: $text;
        padding: 0 1;
    }
    """

    BINDINGS = [("ctrl+q", "quit", "退出")]

    def __init__(
        self, loaded: Loaded, store: Storage, pipeline: Pipeline, bus: EventBus
    ) -> None:
        super().__init__()
        self._config = loaded.config
        self._config_path = loaded.path
        self._created_config = loaded.created
        self._store = store
        self._pipeline = pipeline
        self._bus = bus
        self._events = bus.subscribe()
        self._watched: str | None = None
        self._task_rows: set[str] = set()
        self._latest: JobRecord | None = None
        self._running_jobs: set[str] = set()
        self._streamed: list[str] = []  # the answer currently being written
        self._queue: list[Reference | str] = []  # requests waiting for the worker
        self._busy = False
        self._shown: dict[str, int] = {}  # job id -> answer revision already in the chat

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="chat-column"):
                yield RichLog(id="chat", markup=False, wrap=True, min_width=0)
                yield Static(id="answer")
                yield Input(id="prompt", placeholder="@路径 问题　或　/help")
            with Vertical(id="side-column"):
                yield DataTable(id="tasks", zebra_stripes=True)
                yield Static(id="timeline")
                yield Static(id="evidence")
        yield Static(id="status")

    def on_mount(self) -> None:
        table = self.query_one("#tasks", DataTable)
        for column, header in zip(_TASK_COLUMNS, _TASK_HEADERS, strict=True):
            table.add_column(header, key=column)
        self._running_jobs = {
            job.id for job in self._store.jobs(limit=20) if job.status is JobStatus.RUNNING
        }
        self._say("Kvedio 工作台", style="bold")
        self._say(f"配置 {self._config_path}")
        if self._created_config:
            self._say("已按模板创建配置：填入密钥或其环境变量引用后即可调用模型。", style="yellow")
        self._say(HELP, style="dim")
        self._report_configuration()
        self._reload()
        self._render_status()
        self.set_interval(0.12, self._pump)
        self.query_one("#prompt", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        if not text.startswith(("@", "/")):
            event.input.value = ""
            self._ask(text)
            return
        submission = parse_submission(text, Path.cwd())
        if submission is None:
            self._say(USAGE, style="yellow")
            return
        event.input.value = ""
        if isinstance(submission, Command):
            self._command(submission)
        else:
            self._enqueue(submission)

    def _enqueue(self, item: Reference | str) -> None:
        """Put one request in line; a second job must not re-run the first job's stages."""
        self._queue.append(item)
        if self._busy:
            self._say("已排队，等当前作业结束。", style="dim")
        self._drain()

    def _drain(self) -> None:
        """Start the next request, if nothing is running."""
        if self._busy or not self._queue:
            return
        item = self._queue.pop(0)
        self._busy = True
        if isinstance(item, Reference):
            self._start_reference(item)
        else:
            self._start_question(item)

    def _release(self) -> None:
        """A job ended, or never started; let the next request in line go."""
        self._busy = False
        self._drain()

    def _start_reference(self, reference: Reference) -> None:
        """Reference and analyse on a worker thread; only events come back from it."""
        self._say(f"引用 {reference.path.name}：{reference.question}")

        def work() -> None:
            started = False
            try:
                job = self._pipeline.submit(reference.path, reference.question)
                started = True
                self._pipeline.run(job)
            except Exception as error:  # a dying worker must speak rather than vanish silently
                self._notify(f"无法处理 {reference.path.name}：{error}")
            finally:
                if not started:
                    self.call_from_thread(self._release)

        self.run_worker(work, thread=True, group="reference")

    def _start_question(self, question: str) -> None:
        """Ask about the video on screen; the answer cites what the store holds."""
        if self._latest is None:
            self._say("还没有引用视频：先用 @路径 问题 引用一个。", style="yellow")
            self._release()
            return
        video_id = self._latest.video_id
        self._say(f"提问：{question}")

        def work() -> None:
            started = False
            try:
                job = self._pipeline.ask(video_id, question)
                started = True
                self._pipeline.run(job)
            except Exception as error:
                self._notify(f"作答中断：{error}")
            finally:
                if not started:
                    self.call_from_thread(self._release)

        self.run_worker(work, thread=True, group="answer")

    def _ask(self, question: str) -> None:
        """Queue a question about the video on screen, if there is one."""
        # A reference may still be starting: the video is known once its job is watched.
        if self._latest is None and not self._busy:
            self._say("还没有引用视频：先用 @路径 问题 引用一个。", style="yellow")
            return
        self._enqueue(question)

    def _notify(self, message: str) -> None:
        """Hand one line from a worker to the interface thread; the only way a worker may speak."""
        self._bus.publish(
            Event.stamp(EventKind.LOG_LINE, payload={"message": message})
        )

    def _pump(self) -> None:
        """Drain published events on the interface thread and render each one."""
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return
            self._apply(event)

    def _apply(self, event: Event) -> None:
        kind = event.kind
        if kind is EventKind.LOG_LINE:
            message = event.payload.get("message")
            if message:
                self._say(str(message))
            return
        if event.job_id is None:
            return

        if kind is EventKind.ANSWER_DELTA:
            delta = event.payload.get("delta")
            if isinstance(delta, str) and delta:
                self._streamed.append(delta)
                self._render_stream()
            return
        if kind is EventKind.JOB_QUEUED:
            self._watch(event.job_id)
        elif kind is EventKind.JOB_STARTED:
            self._close_stream()
            self._watch(event.job_id)
            self._running_jobs.add(event.job_id)
        elif kind is EventKind.TASK_PROGRESS:
            self._show_progress(event)
        elif kind in _TASK_EVENTS:
            self._refresh_tasks(event.job_id)
        elif kind is EventKind.VIDEO_REGISTERED:
            self._render_timeline()
            self._render_evidence()
        elif kind is EventKind.RESULT_PARTIAL:
            self._present_result(
                event.job_id,
                partial=True,
                expected=_revision(event),
                pending=_pending(event),
            )
        elif kind is EventKind.RESULT_FINAL:
            self._present_result(event.job_id, partial=False, expected=_revision(event))
        elif kind in _JOB_END_EVENTS:
            self._running_jobs.discard(event.job_id)
            self._announce(event.job_id)
        self._render_status()

    def _watch(self, job_id: str) -> None:
        """Point the task panel at one job; switching clears rows instead of mixing graphs."""
        if self._watched == job_id:
            return
        self._watched = job_id
        table = self.query_one("#tasks", DataTable)
        table.clear()
        self._task_rows.clear()
        self._latest = self._store.job(job_id)
        for task in self._store.tasks_for_job(job_id):
            self._upsert(task)

    def _refresh_tasks(self, job_id: str) -> None:
        if job_id != self._watched:
            return
        for task in self._store.tasks_for_job(job_id):
            self._upsert(task)

    def _upsert(self, task: TaskRecord) -> None:
        table = self.query_one("#tasks", DataTable)
        values = (
            task.id,
            agent_label(task.agent),
            task.action,
            task_status_label(task.status),
            format_progress(task.progress),
        )
        if task.id in self._task_rows:
            for column, value in zip(_TASK_COLUMNS, values, strict=True):
                table.update_cell(task.id, column, value)
        else:
            table.add_row(*values, key=task.id)
            self._task_rows.add(task.id)

    def _show_progress(self, event: Event) -> None:
        if event.job_id != self._watched or event.task_id not in self._task_rows:
            return
        self.query_one("#tasks", DataTable).update_cell(
            event.task_id, "progress", format_progress(float(event.payload.get("progress", 0.0)))
        )

    def _announce(self, job_id: str) -> None:
        job = self._store.job(job_id)
        if job is None:
            return
        self._latest = job
        detail = job.error
        if detail is None:
            video = self._latest_video()
            if video is not None and video.media is not None:
                detail = format_duration(video.media.duration_ms)
        label = job_status_label(job.status)
        message = f"作业 {label}：{detail}" if detail else f"作业 {label}"
        self._say(message, style="bold red" if job.status is JobStatus.FAILED else "green")
        self._close_stream()
        self._announce_answer(job.id)
        self._render_timeline()
        self._render_evidence()
        self._release()

    def _announce_answer(self, job_id: str) -> None:
        """Print the conclusion on screen when no result event reached the chat first."""
        self._present_result(job_id, partial=None)

    def _present_result(
        self,
        job_id: str,
        *,
        partial: bool | None,
        expected: int | None = None,
        pending: tuple[str, ...] = (),
    ) -> None:
        """Print one stored revision, once, marking a preview or a supplemented text.

        ``expected`` is the revision the event announced: a preview that was
        already replaced by the time the pump ran is skipped rather than shown
        with the final text under a provisional label. ``pending`` names the
        branches still working, which is different from a branch that failed.
        """
        answer = self._store.answer(job_id)
        if answer is None:
            return
        if expected is not None and answer.revision != expected:
            return
        if self._shown.get(job_id) == answer.revision:
            return
        self._shown[job_id] = answer.revision
        self._latest = self._store.job(job_id) or self._latest
        if partial is True:
            detail = "、".join(pending) if pending else "部分证据"
            self._say(f"初步结论（{detail}尚未到位）", style="yellow")
        elif answer.stage == "partial":
            self._say("初步结论（作业未完成）", style="yellow")
        elif answer.revision > 1:
            self._say("结论已补充", style="green")
        self._close_stream()
        self._say(answer.text)
        if not answer.citations:
            self._say("没有可引用的片段。", style="dim")
        else:
            moments = "、".join(_span(citation) for citation in answer.citations)
            self._say(f"证据 {len(answer.citations)} 处：{moments}", style="dim")
        self._render_evidence()

    def _reload(self) -> None:
        """Render what the store already holds, so a restart does not look like an empty install."""
        jobs = self._store.jobs(limit=1)
        if jobs:
            self._watch(jobs[0].id)
        else:
            self._say("尚无视频：用 @路径 问题 引用一个本地文件。", style="dim")
        self._render_timeline()
        self._render_evidence()

    def _render_timeline(self) -> None:
        panel = self.query_one("#timeline", Static)
        video = self._latest_video()
        if video is None:
            panel.update("时间轴：尚未引用视频")
            return
        if video.media is None:
            panel.update(f"时间轴：{video.path.name}（时长未知，等待探测）")
            return
        lines = [
            f"时间轴：00:00:00 ────────────── {format_duration(video.media.duration_ms)}",
            f"{format_size(video.size_bytes)}　·　{format_container(video.media.container)}",
            self._transcript_line(video),
            self._memory_line(video),
        ]
        panel.update("\n".join(lines))

    def _transcript_line(self, video: VideoRecord) -> str:
        """How much of the recording has words in the store, or why it has none."""
        if video.media is not None and video.media.audio_codec is None:
            return "转录：无音轨"
        transcript = self._store.transcript(video.id)
        if transcript is None:
            return "转录：尚无"
        if not transcript.ready:
            return "转录：进行中"
        if transcript.segment_count == 0:
            return f"转录：{transcript.model}　无语音"
        first = format_duration(transcript.first_ms or 0)
        last = format_duration(transcript.last_ms or 0)
        return f"转录：{transcript.segment_count} 段 · {first}–{last} · {transcript.model}"

    def _render_stream(self) -> None:
        """Show the tail of the answer as it is written; the log gets the final text."""
        panel = self.query_one("#answer", Static)
        panel.styles.display = "block"
        panel.update("".join(self._streamed)[-600:])

    def _close_stream(self) -> None:
        self._streamed.clear()
        panel = self.query_one("#answer", Static)
        panel.update("")
        panel.styles.display = "none"

    def _memory_line(self, video: VideoRecord) -> str:
        """Whether the map and the vector index exist, and how large they are."""
        blocks, vectors = self._store.memory_summary(video.id)
        scenes = len(self._store.scenes(video.id))
        events = len(self._store.events(video.id))
        parts = []
        if blocks:
            parts.append(f"覆盖图 {blocks} 块")
        if vectors:
            parts.append(f"向量 {vectors} 条")
        if scenes:
            parts.append(f"场景 {scenes} 个")
        if events:
            parts.append(f"事件 {events} 个")
        return "记忆：" + (" · ".join(parts) if parts else "未建立")

    def _render_evidence(self) -> None:
        panel = self.query_one("#evidence", Static)
        video = self._latest_video()
        if video is None:
            panel.update("证据：暂无")
            return
        if self._latest is not None:
            answer = self._store.answer(self._latest.id)
            if answer is not None and answer.citations:
                panel.update(self._citation_text(answer))
                return
        lines = [f"路径　{video.path}", f"指纹　{video.fingerprint[:16]}…"]
        media = video.media
        if media is None:
            lines.append("流信息　等待探测")
        else:
            lines.extend(
                [
                    f"时长　{format_duration(media.duration_ms)}",
                    f"容器　{format_container(media.container)}",
                    f"画面　{media.video_codec or '无'}　{media.width}×{media.height}"
                    f"　{media.fps or 0:.2f} fps",
                    f"音轨　{media.audio_codec or '无'}　{media.sample_rate or 0} Hz"
                    f"　{media.channels or 0} ch",
                ]
            )
        if self._latest is not None and self._latest.error:
            lines.append(f"错误　{self._latest.error}")
        panel.update("\n".join(lines))

    def _citation_text(self, answer: Answer) -> str:
        """The moments behind the conclusion on screen, with its revision."""
        stage = "初步" if answer.stage == "partial" else "最终"
        lines = [f"引用 {len(answer.citations)} 处 · 第 {answer.revision} 版（{stage}）"]
        for citation in answer.citations[:3]:
            lines.append(f"{_span(citation)}　{citation.quote[:34]}")
        if len(answer.citations) > 3:
            lines.append(f"…另有 {len(answer.citations) - 3} 处")
        return "\n".join(lines)

    def _render_status(self) -> None:
        parts = [f"库 {self._store.path.name}"]
        for role in Role:
            binding = self._config.for_role(role)
            parts.append(f"{role_label(role)} {model_label(binding, is_usable(binding))}")
        parts.append(f"运行中 {len(self._running_jobs)}")
        self.query_one("#status", Static).update(" │ ".join(parts))

    def _latest_video(self) -> VideoRecord | None:
        if self._latest is None:
            return None
        return self._store.video(self._latest.video_id)

    def _report_configuration(self) -> None:
        """Say once which roles cannot call out, rather than leaving a silent status bar."""
        missing = [role_label(role) for role in Role if self._config.for_role(role) is None]
        if missing:
            self._say(
                f"配置缺少角色模型：{'、'.join(missing)}；编辑 {self._config_path} 补齐",
                style="yellow",
            )
        for role in Role:
            binding = self._config.for_role(role)
            if binding is None or is_usable(binding):
                continue
            reason = f"环境变量 {binding.env_var} 未设置" if binding.env_var else "未填写密钥"
            self._say(f"{role_label(role)}未配置密钥：{reason}", style="yellow")

    def _command(self, command: Command) -> None:
        match command.name:
            case "quit" | "q":
                self.exit()
            case "help":
                self._say(HELP)
            case "model":
                self._list_models(command.arguments)
            case "cancel":
                self._cancel()
            case "usage":
                self._usage_report()
            case _:
                self._say(f"未知命令 /{command.name}；/help 查看全部", style="yellow")

    def _cancel(self) -> None:
        """Stop the running job and drop what is queued; finished work stays."""
        dropped = len(self._queue)
        self._queue.clear()
        running = self._watched is not None and self._pipeline.cancel(self._watched)
        if not running and not dropped:
            self._say("没有正在运行的作业。", style="dim")
            return
        self._close_stream()
        parts = []
        if running:
            parts.append("已请求取消当前作业")
        if dropped:
            parts.append(f"清空排队 {dropped} 条")
        self._say("；".join(parts) + "。", style="yellow")

    def _usage_report(self) -> None:
        """Print what the model calls have cost so far, by kind and model."""
        totals = self._store.usage_totals()
        if not totals:
            self._say("还没有模型调用。", style="dim")
            return
        lines = []
        for item in totals:
            spent = []
            if item.prompt_tokens or item.completion_tokens:
                spent.append(
                    f"{format_tokens(item.prompt_tokens)} 输入 / "
                    f"{format_tokens(item.completion_tokens)} 输出"
                )
            if item.seconds:
                spent.append(f"{item.seconds / 60:.1f} 分钟音频")
            if not spent:
                spent.append("用量未上报")
            lines.append(
                f"{usage_kind_label(item.kind)} {item.model}：{item.calls} 次 · "
                + " · ".join(spent)
            )
        self._say("\n".join(lines))

    def _list_models(self, arguments: tuple[str, ...]) -> None:
        """Show what each role calls; changing a model is an edit to the file, not a command."""
        if arguments:
            self._say(f"更换模型：编辑 {self._config_path} 后重启工作台", style="yellow")
            return
        self._say("\n".join(self._model_lines()))

    def _model_lines(self) -> list[str]:
        lines: list[str] = []
        for role in Role:
            binding = self._config.for_role(role)
            if binding is None:
                lines.append(f"{role_label(role)}（{role.value}）未配置")
                continue
            usable = "已配置" if is_usable(binding) else "未配置"
            lines.append(f"{role_label(role)}（{role.value}）{usable}")
            lines.append(f"  模型　{binding.model}")
            lines.append(f"  地址　{binding.url}")
            if binding.env_var is not None:
                lines.append(f"  密钥　环境变量 {binding.env_var}")
        return lines

    def _say(self, message: str, *, style: str = "") -> None:
        """Write one line to the chat panel; the text is never parsed as markup.

        Long unbroken runs — a path, a URL — fold rather than crop: a truncated
        endpoint reads as a different endpoint.
        """
        self.query_one("#chat", RichLog).write(
            Text(message, style=style or None, overflow="fold")
        )


def _revision(event: Event) -> int | None:
    """The answer revision a result event announced, when it carries one."""
    value = event.payload.get("revision")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _pending(event: Event) -> tuple[str, ...]:
    """The branches a partial result still expects, as the event named them."""
    value = event.payload.get("pending")
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


def _span(citation: Citation) -> str:
    """``mm:ss`` for a single moment, ``mm:ss–mm:ss`` for a stretch."""
    if citation.end_ms - citation.start_ms <= 1_000:
        moment = format_moment(citation.start_ms)
    else:
        moment = f"{format_moment(citation.start_ms)}–{format_moment(citation.end_ms)}"
    return f"画面 {moment}" if citation.kind == "scene" else moment
