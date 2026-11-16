"""Writing the conclusion, and refusing one that cites what it never saw."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ...tools import Evidence
from .. import prompts
from ..domain import Citation
from ..llm import Chat, Message
from . import citations

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
    limits: Sequence[str] = (),
    on_delta: Callable[[str], None] | None = None,
) -> Conclusion:
    """Ask for an answer and check every citation before returning it.

    ``limits`` names the sources this recording could not supply, so the prompt
    can keep the model from claiming more than the evidence covers. One repair
    attempt is allowed, with the offending moments named back to the model; a
    second failure raises. The alternative — storing the text as given — would
    leave the interface showing a timestamp that opens nothing.
    """
    if not evidence:
        raise ValueError("没有可用的证据")
    system = prompts.render(
        "answer",
        question=question,
        evidence=evidence,
        duration_ms=duration_ms,
        limits=limits,
    )
    base = [Message(role="system", content=system)]

    reply = chat.stream(base, on_delta) if on_delta is not None else chat.complete(base)
    verdict = citations.check(reply.text, evidence)
    if verdict.ok:
        return Conclusion(text=reply.text.strip(), citations=verdict.citations)

    examples = [item for item in evidence if not item.label][:REPAIR_SAMPLES]
    repair = prompts.render(
        "repair", missing=verdict.missing, cited=verdict.cited, examples=examples
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


def write_absence(chat: Chat, *, question: str, limits: Sequence[str] = ()) -> str:
    """Say that the available evidence does not cover the question, in its language.

    Retrieval found nothing, so there is no evidence to cite and no citation check
    to run. ``limits`` distinguishes "the recording does not cover this" from
    "the source that would have covered it was unavailable", which the model must
    not conflate. The model is still asked because only the supervisor speaks to
    the user, and a fixed sentence would answer in the wrong language half the time.
    """
    system = prompts.render("no_evidence", question=question, limits=limits)
    return chat.complete([Message(role="system", content=system)]).text.strip()
