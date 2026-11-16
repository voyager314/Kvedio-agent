"""Writing the conclusion, and refusing one that cites what it never saw."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...tools import Evidence
from ..domain import Citation, format_moment
from ..llm import Chat, Message
from . import citations, prompts

REPAIR = "上一版结论的问题：{problem}。请只引用给定片段中出现的时间点，重新回答。"


class UnsupportedCitation(Exception):
    """The answer cited a moment that was not among the evidence it was given.

    Raised instead of storing the answer: a conclusion whose evidence cannot be
    traced is exactly what this system exists not to produce.
    """


@dataclass(frozen=True)
class Conclusion:
    """The answer text and the citations it is allowed to keep."""

    text: str
    citations: tuple[Citation, ...]


def write(
    chat: Chat,
    *,
    question: str,
    evidence: Sequence[Evidence],
    duration_ms: int,
) -> Conclusion:
    """Ask for an answer and check every citation before returning it.

    One repair attempt is allowed, with the offending moments named back to the
    model; a second failure raises. The alternative — storing the text as given —
    would leave the interface showing a timestamp that opens nothing.
    """
    if not evidence:
        raise ValueError("没有可用的证据")
    block = "\n".join(item.render() for item in evidence)
    base = [
        Message(
            role="system",
            content=prompts.render(
                "answer",
                evidence=block,
                question=question,
                duration=format_moment(duration_ms),
            ),
        )
    ]

    reply = chat.complete(base)
    kept, problem = citations.check(reply.text, evidence)
    if problem is None:
        return Conclusion(text=reply.text.strip(), citations=kept)

    repair = Message(role="user", content=REPAIR.format(problem=problem))
    second = chat.complete(
        [*base, Message(role="assistant", content=reply.text), repair]
    )
    kept, problem = citations.check(second.text, evidence)
    if problem is not None:
        raise UnsupportedCitation(problem)
    return Conclusion(text=second.text.strip(), citations=kept)
