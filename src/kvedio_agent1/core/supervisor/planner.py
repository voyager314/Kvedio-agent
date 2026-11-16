"""Turning a question into a short list of tool calls."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ...tools import Tool
from .. import prompts
from ..llm import Chat, Message

MAX_STEPS = 4


@dataclass(frozen=True)
class Step:
    """One tool call the plan asks for."""

    tool: str  # a name the toolbox listed
    arguments: Mapping[str, Any]  # tool-specific; validated by the tool itself
    why: str = ""  # one line the interface shows as the trace; empty when we wrote the step


@dataclass(frozen=True)
class Plan:
    """The calls to make, and a line saying where they came from."""

    steps: tuple[Step, ...]
    note: str


def make_plan(
    chat: Chat, question: str, tools: Sequence[Tool], *, max_steps: int = MAX_STEPS
) -> Plan:
    """Ask the model for a plan, falling back to a fixed one when it cannot be used.

    A failed call raises: falling back would turn an outage or a bad key into a
    quietly worse answer. Only an unusable *reply* — prose, unknown tools, too
    many steps — degrades to the fixed plan, and the note says so.
    """
    message = prompts.render("plan", tools=tools, question=question, max_steps=max_steps)
    reply = chat.complete([Message(role="system", content=message)], json_mode=True)
    steps = _parse(reply.text, frozenset(tool.name for tool in tools), max_steps)
    if not steps:
        return Plan(_fallback(question, tools), "计划不可用，改用固定检索")
    return Plan(steps, f"按计划执行 {len(steps)} 步")


def _parse(text: str, names: frozenset[str], max_steps: int) -> tuple[Step, ...]:
    payload = _json_object(text)
    raw = payload.get("steps") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return ()
    steps: list[Step] = []
    for item in raw[:max_steps]:
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


def _fallback(question: str, tools: Sequence[Tool]) -> tuple[Step, ...]:
    """The plan used when the model's plan cannot be read: look wide, then look for the question.

    Only tools the box offered are considered, so a recording with pictures but
    no words does not spend a step on a transcript lookup. No ``why`` is attached:
    these steps are ours, not the model's, and the tool's own note is a truthful
    trace.
    """
    names = {tool.name for tool in tools}
    steps: list[Step] = []
    if "overview" in names:
        steps.append(Step(tool="overview", arguments={}))
    if "search" in names:
        steps.append(Step(tool="search", arguments={"query": question[:40]}))
    if not steps and "scenes" in names:
        steps.append(Step(tool="scenes", arguments={}))
    if not steps and "events" in names:
        steps.append(Step(tool="events", arguments={}))
    return tuple(steps)
