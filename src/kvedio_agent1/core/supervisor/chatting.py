"""Answering a request that belongs to the supervisor rather than to an agent.

A text-only question, a referenced text file, or a question about material that
is already analysed does not need a new video pass. This module reads the text
file, asks the planner which lookups to run over the video on screen, and has the
conversation model answer from the evidence they return. It never decodes media
and never starts a stage: delegation is the router's decision, not this module's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...tools import TOOLS, Evidence, Tool, ToolBox
from .. import prompts
from ..domain import VideoRecord
from ..llm import Chat, Embedder, Message
from ..memory import VideoMemory
from ..storage import Storage
from .planner import make_plan

MAX_TEXT_CHARS = 40_000


@dataclass(frozen=True)
class ChatReply:
    """One supervisor answer, with the trace lines that produced it."""

    text: str
    notes: tuple[str, ...] = ()


class Chatting:
    """Answers a text request from a text file and the analysed video on screen."""

    def __init__(
        self,
        store: Storage,
        chat: Chat,
        embedder: Embedder | None = None,
    ) -> None:
        self._store = store
        self._chat = chat
        self._embedder = embedder

    def answer(
        self,
        question: str,
        *,
        video: VideoRecord | None = None,
        text_path: Path | None = None,
    ) -> ChatReply:
        """Read the material, ask for an answer, and return the trace behind it."""
        text_file = text_path.name if text_path is not None else None
        text = _read_text(text_path) if text_path is not None else ""
        evidence, notes = self._video_evidence(video, question)
        system = prompts.render(
            "chat",
            question=question,
            text_file=text_file,
            text=text,
            video_name=video.path.name if video is not None else None,
            evidence=evidence,
            duration_ms=video.media.duration_ms if video is not None and video.media else 0,
        )
        reply = self._chat.complete([Message(role="system", content=system)])
        return ChatReply(text=reply.text.strip(), notes=notes)

    def _video_evidence(
        self, video: VideoRecord | None, question: str
    ) -> tuple[tuple[Evidence, ...], tuple[str, ...]]:
        """Run the planned lookups and return their evidence, with a line per call.

        The plan is the model's, not a fixed sweep: which tool to call, in what
        order, and what to search for are judgements about the question, and a
        keyword heuristic gets them wrong on paraphrases and on anything not
        typed in the two languages it knows.
        """
        if video is None:
            return (), ("没有在屏的视频。",)
        tools = _available(video, self._store)
        if not tools:
            return (), ("该视频还没有可用的转录或场景描述。",)
        toolbox = ToolBox(
            self._store,
            video,
            VideoMemory(self._store, video, self._embedder),
            tools=tools,
        )
        plan = make_plan(self._chat, question, tools)
        collected: dict[tuple[int, int, str], Evidence] = {}
        notes: list[str] = [plan.note]
        for step in plan.steps:
            result = toolbox.call(step.tool, step.arguments)
            notes.append(f"{step.why} · {result.note}" if step.why else result.note)
            for item in result.evidence:
                collected.setdefault((item.start_ms, item.end_ms, item.label), item)
        return tuple(collected[key] for key in sorted(collected)), tuple(notes)


def _available(video: VideoRecord, store: Storage) -> tuple[Tool, ...]:
    """The tools whose backing data exists for this video, in catalogue order."""
    names: set[str] = set()
    transcript = store.transcript(video.id)
    if transcript is not None and transcript.ready and transcript.segment_count > 0:
        names.update(("overview", "search", "read_range"))
    scenes = store.scenes(video.id)
    if scenes:
        names.add("scenes")
        if store.events(video.id):
            names.add("events")
    return tuple(tool for tool in TOOLS if tool.name in names)


def _read_text(path: Path) -> str:
    """Read a referenced text file, truncating one too large to put in a prompt."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"无法读取文本文件：{path.name}") from error
    text = raw.decode("utf-8", errors="replace")
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return text[:MAX_TEXT_CHARS] + "\n\n[文本过长，已截断]"
