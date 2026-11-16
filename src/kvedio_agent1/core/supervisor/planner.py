"""Turning a question into a short list of tool calls."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..llm import Chat, Message
from . import prompts

MAX_STEPS = 4


@dataclass(frozen=True)
class Step:
    """One tool call the plan asks for."""

    tool: str  # a name the toolbox listed
    arguments: Mapping[str, Any]  # tool-specific; validated by the tool itself
    why: str  # one line the interface shows as the trace


@dataclass(frozen=True)
class Plan:
    """The calls to make, and a line saying where they came from."""

    steps: tuple[Step, ...]
    note: str


def make_plan(
    chat: Chat, question: str, catalogue: str, names: frozenset[str]
) -> Plan:
    """Ask the model for a plan, falling back to a fixed one when it cannot be used.

    A failed call raises: falling back would turn an outage or a bad key into a
    quietly worse answer. Only an unusable *reply* — prose, unknown tools, too
    many steps — degrades to the fixed plan, and the note says so.
    """
    message = prompts.render(
        "plan", catalogue=catalogue, question=question, max_steps=MAX_STEPS
    )
    reply = chat.complete([Message(role="system", content=message)], json_mode=True)
    steps = _parse(reply.text, names)
    if not steps:
        return Plan(_fallback(question), "计划不可用，改用固定检索")
    return Plan(steps, f"按计划执行 {len(steps)} 步")


def _parse(text: str, names: frozenset[str]) -> tuple[Step, ...]:
    payload = _json_object(text)
    raw = payload.get("steps") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return ()
    steps: list[Step] = []
    for item in raw[:MAX_STEPS]:
        if not isinstance(item, dict):
            continue
        tool = item.get("tool")
        arguments = item.get("arguments", {})
        why = item.get("why", "")
        if not isinstance(tool, str) or tool not in names:
            continue
        if not isinstance(arguments, dict):
            continue
        steps.append(
            Step(tool=tool, arguments=arguments, why=str(why)[:120].strip())
        )
    return tuple(steps)


def _json_object(text: str) -> dict[str, Any]:
    """Read the object out of a reply that may be wrapped in prose or a code fence."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _fallback(question: str) -> tuple[Step, ...]:
    """The plan used when the model's plan cannot be read: look wide, then look for the question."""
    return (
        Step(tool="overview", arguments={}, why="先看整体分布"),
        Step(tool="search", arguments={"query": question[:40]}, why="按问题关键词检索"),
    )
