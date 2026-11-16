"""Writing the conclusion, and refusing one that cites what it never saw."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...tools import Evidence
from ..domain import Citation
from ..llm import Chat, Message
from . import citations, prompts

REPAIR_SAMPLES = 4  # how many allowed moments the correction shows as examples


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
    system = prompts.render(
        "answer", question=question, evidence=evidence, duration_ms=duration_ms
    )
    base = [Message(role="system", content=system)]

    reply = chat.complete(base)
    verdict = citations.check(reply.text, evidence)
    if verdict.ok:
        return Conclusion(text=reply.text.strip(), citations=verdict.citations)

    repair = prompts.render(
        "repair",
        missing=verdict.missing,
        cited=verdict.cited,
        evidence=evidence,
        limit=REPAIR_SAMPLES,
    )
    second = chat.complete(
        [*base, Message(role="assistant", content=reply.text), Message(role="user", content=repair)]
    )
    verdict = citations.check(second.text, evidence)
    if not verdict.ok:
        raise UnsupportedCitation(_rejection(verdict))
    return Conclusion(text=second.text.strip(), citations=verdict.citations)


def _rejection(verdict: citations.Verdict) -> str:
    """The failure message for the log and the panel; the prompt never carries this text."""
    if not verdict.cited:
        return "结论没有引用任何时间点"
    return "结论引用了未检索到的时间点：" + "、".join(verdict.missing)


def write_absence(chat: Chat, *, question: str) -> str:
    """Say that the recording does not cover the question, in the question's language.

    Retrieval found nothing, so there is no evidence to cite and no citation check
    to run. The model is still asked because only the supervisor speaks to the
    user, and a fixed sentence would answer in the wrong language half the time.
    """
    system = prompts.render("no_evidence", question=question)
    return chat.complete([Message(role="system", content=system)]).text.strip()
