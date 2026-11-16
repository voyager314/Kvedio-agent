"""Deciding who does the work: a video agent, or the supervisor itself.

The rules cover the requests whose intent is visible in the wording or the file
extension. Everything else is put to the supervisor model, which sees the same
request plus what the video already contains. A reply that cannot be read, or
that asks for work the request cannot support, falls back to the rules rather
than sending the job down a path that cannot finish.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import prompts
from ..domain import VideoRecord
from ..llm import Chat, Message

TEXT_SUFFIXES = frozenset(
    {".txt", ".md", ".markdown", ".json", ".csv", ".tsv", ".log", ".yaml", ".yml"}
)

_ROUTES = frozenset({"analyze_video", "ask_video", "chat", "text_file"})
_AGENTS = frozenset({"asr", "video"})

_VISUAL_CUES = (
    "画面",
    "镜头",
    "屏幕",
    "演示",
    "动作",
    "表情",
    "图表",
    "幻灯片",
    "白板",
    "出现",
    "几个人",
    "看向",
    "拿起",
    "picture",
    "screen",
    "scene",
    "visible",
    "frame",
    "demo",
)
_SPEECH_CUES = (
    "说",
    "讲",
    "提到",
    "台词",
    "谁",
    "对话",
    "讨论",
    "录音",
    "语音",
    "念",
    "回答",
    "speak",
    "said",
    "mention",
    "talk",
    "discuss",
    "transcript",
)
_SUMMARY_CUES = (
    "总结",
    "概括",
    "讲了什么",
    "主要内容",
    "概述",
    "复盘",
    "梳理",
    "summary",
    "summarize",
    "overview",
)
_GENERAL_CUES = (
    "写",
    "翻译",
    "润色",
    "邮件",
    "代码",
    "生成",
    "整理成",
    "笔记",
    "表格",
    "解释一下",
    "帮我",
    "write",
    "translate",
    "draft",
    "email",
    "code",
    "note",
    "table",
)
_ARTIFACT_CUES = (
    "整理成",
    "笔记",
    "表格",
    "邮件",
    "翻译",
    "润色",
    "写一封",
    "生成一份",
    "markdown",
    "note",
    "table",
    "email",
    "translate",
    "draft",
)


@dataclass(frozen=True)
class Request:
    """One submitted line, with what the store already knows about the video."""

    question: str
    path: Path | None = None  # a newly referenced file, before registration
    video: VideoRecord | None = None  # the video already on screen
    transcript_ready: bool = False
    scenes: int = 0
    events: int = 0
    text_path: Path | None = None  # a referenced text file, already identified as text

    @property
    def has_evidence(self) -> bool:
        """Whether the video on screen already has a citable product."""
        return self.transcript_ready or self.scenes > 0

    @property
    def referenced_text(self) -> Path | None:
        """The text file this request references, whichever field carries it."""
        if self.text_path is not None:
            return self.text_path
        if self.path is not None and self.path.suffix.lower() in TEXT_SUFFIXES:
            return self.path
        return None


@dataclass(frozen=True)
class Route:
    """Who runs the request, and which branches it asks for.

    ``kind`` is ``analyze_video`` (a new video needs analysis), ``ask_video`` (a
    question about the video on screen), ``chat`` (the supervisor answers) or
    ``text_file`` (the supervisor reads a referenced file). ``agents`` is the
    subset of ``asr`` and ``video`` the orchestrator should run.
    """

    kind: str
    agents: tuple[str, ...] = ()
    reason: str = ""


class Router:
    """Picks a route with rules first and the supervisor model only when ambiguous."""

    def __init__(self, chat: Chat | None = None) -> None:
        self._chat = chat

    def route(self, request: Request) -> Route:
        """Return a usable route for one request, never raising for a model failure."""
        decided = _rules(request)
        if decided is not None:
            return _checked(decided, request)
        if self._chat is not None:
            try:
                return _checked(_ask(self._chat, request), request)
            except Exception:  # a bad route reply must not lose the request
                pass
        return _checked(_fallback(request), request)


def _ask(chat: Chat, request: Request) -> Route:
    """Put the ambiguous request to the supervisor model as a routing question."""
    text_file = request.referenced_text
    message = prompts.render(
        "route",
        question=request.question,
        video_name=request.video.path.name if request.video is not None else None,
        transcript_state="ready" if request.transcript_ready else "absent",
        scenes=request.scenes,
        events=request.events,
        text_file=text_file.name if text_file is not None else None,
        no_video=request.video is None and request.path is None,
    )
    reply = chat.complete([Message(role="system", content=message)], json_mode=True)
    return _parse(reply.text)


def _parse(text: str) -> Route:
    payload = _json_object(text)
    kind = payload.get("route")
    agents = payload.get("agents", [])
    reason = payload.get("reason", "")
    if kind not in _ROUTES or not isinstance(agents, list):
        raise ValueError("路由回复不可用")
    names = tuple(str(name) for name in agents if str(name) in _AGENTS)
    return Route(str(kind), names, str(reason)[:120].strip())


def _json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _rules(request: Request) -> Route | None:
    """The routes whose intent is visible without a model call."""
    if request.referenced_text is not None:
        return Route("text_file", (), "引用了文本文件")
    if request.path is not None:
        return Route("analyze_video", ("asr", "video"), "新引用的视频，建立完整索引")
    if request.video is None:
        return Route("chat", (), "没有在屏的视频")
    if request.has_evidence and _has(request.question, _ARTIFACT_CUES):
        return Route("chat", (), "文本产出，由 supervisor 使用已有证据")
    if _has(request.question, _GENERAL_CUES) and not _has(
        request.question, _VISUAL_CUES + _SPEECH_CUES + _SUMMARY_CUES
    ):
        return Route("chat", (), "文本类任务，由 supervisor 直接处理")
    visual = _has(request.question, _VISUAL_CUES)
    speech = _has(request.question, _SPEECH_CUES)
    summary = _has(request.question, _SUMMARY_CUES)
    if visual or speech or summary:
        agents = _missing_agents(request, visual=visual, speech=speech, summary=summary)
        return Route("ask_video", agents, "基于已分析视频作答")
    return None


def _missing_agents(
    request: Request, *, visual: bool, speech: bool, summary: bool
) -> tuple[str, ...]:
    """Only the branches whose product is absent; empty means answer from the store."""
    wanted: list[str] = []
    if (summary or speech) and not request.transcript_ready:
        wanted.append("asr")
    if (summary or visual) and request.scenes <= 0:
        wanted.append("video")
    return tuple(dict.fromkeys(wanted))


def _checked(route: Route, request: Request) -> Route:
    """Normalise a route against what the request can actually support."""
    if route.kind not in _ROUTES:
        return _checked(_fallback(request), request)
    if request.referenced_text is not None and route.kind in ("analyze_video", "ask_video"):
        return Route("text_file", (), route.reason)
    if route.kind in ("chat", "text_file"):
        if route.kind == "text_file" and request.referenced_text is None:
            return _checked(Route("chat", (), route.reason), request)
        return Route(route.kind, (), route.reason)
    if request.video is None and request.path is None:
        return _checked(Route("chat", (), route.reason), request)
    agents = tuple(dict.fromkeys(name for name in route.agents if name in _AGENTS))
    if route.kind == "analyze_video" and not agents:
        agents = ("asr", "video")
    if route.kind == "ask_video" and not agents and not request.has_evidence:
        agents = ("asr", "video")
    return Route(route.kind, agents, route.reason)


def _fallback(request: Request) -> Route:
    """The route used when neither the rules nor the model produced one."""
    if request.referenced_text is not None:
        return Route("text_file", (), "按文本文件处理")
    if request.path is not None:
        return Route("analyze_video", ("asr", "video"), "按视频分析处理")
    if request.video is None:
        return Route("chat", (), "按普通任务处理")
    agents = _missing_agents(request, visual=True, speech=True, summary=True)
    return Route("ask_video", agents, "按已有视频作答")


def _has(text: str, words: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(word.lower() in lowered for word in words)
