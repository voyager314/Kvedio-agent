"""Deciding who does the work: a video agent, or the supervisor itself.

Two kinds of decision meet here, and only one of them is a model's to make.

Facts are settled by the request and the store: a referenced text file, a video
referenced for the first time, no video on screen at all. Asking a model whether
``notes.md`` is a video would be theatre, so these are read off the request and
never put to it.

Intent is the other kind: is this message a question about the recording, or a
task for the supervisor that merely happens to arrive while a video is on screen?
That is a semantic judgement, and keyword lists answer it badly — they miss
paraphrases, fire on negations, and depend on the language the user typed. It goes
to the supervisor model, which sees the same request plus what the video already
contains, and whose answer is normalised against what the request can support.

The rules survive only as the fallback for a model that is absent, unreachable,
or that returns something unusable.
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
    """Settles the facts itself and puts the intent to the supervisor model."""

    def __init__(self, chat: Chat | None = None) -> None:
        self._chat = chat

    def route(self, request: Request) -> Route:
        """Return a usable route for one request, never raising for a model failure."""
        decided = _structural(request)
        if decided is not None:
            return _checked(decided, request)
        if self._chat is not None:
            try:
                return _checked(_ask(self._chat, request), request)
            except Exception:  # a bad route reply must not lose the request
                pass
        return _checked(_fallback(request), request)


def _ask(chat: Chat, request: Request) -> Route:
    """Put the request's intent to the supervisor model as a routing question."""
    text_file = request.referenced_text
    message = prompts.render(
        "route",
        question=request.question,
        video_name=request.video.path.name if request.video is not None else None,
        transcript_state="ready" if request.transcript_ready else "absent",
        scenes=request.scenes,
        events=request.events,
        text_file=text_file.name if text_file is not None else None,
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


def _structural(request: Request) -> Route | None:
    """The routes the request and the store already settle, without a model."""
    if request.referenced_text is not None:
        return Route("text_file", (), "引用了文本文件")
    if request.path is not None:
        return Route("analyze_video", ("asr", "video"), "新引用的视频，建立完整索引")
    if request.video is None:
        return Route("chat", (), "没有在屏的视频")
    return None


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
        agents = ("asr", "video")  # nothing to answer from: produce it first
    return Route(route.kind, agents, route.reason)


def _fallback(request: Request) -> Route:
    """What to do when no model can decide: answer from the video that is here."""
    if request.referenced_text is not None:
        return Route("text_file", (), "按文本文件处理")
    if request.path is not None:
        return Route("analyze_video", ("asr", "video"), "按视频分析处理")
    if request.video is None:
        return Route("chat", (), "按普通任务处理")
    return Route("ask_video", (), "没有可用的路由模型，按视频作答")
