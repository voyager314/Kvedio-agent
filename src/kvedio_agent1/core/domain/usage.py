"""What a model call consumed, and how much of it the provider reported."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class UsageKind(StrEnum):
    """Which capability spent the call; each is billed differently."""

    CHAT = "chat"
    TRANSCRIPTION = "transcription"  # billed by audio seconds, not tokens
    EMBEDDING = "embedding"
    VISION = "vision"


@dataclass(frozen=True)
class Usage:
    """One call's cost as the provider reported it.

    ``prompt_tokens`` and ``completion_tokens`` are zero when the provider did
    not report them, which is not the same as free: ``calls`` in the totals keeps
    the count, so an unreported call is visible rather than invisible.
    """

    kind: UsageKind
    model: str  # the model that answered, as the provider named it
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0  # audio sent, for transcription


@dataclass(frozen=True)
class UsageTotal:
    """The sum of every call that shared a kind and a model."""

    kind: UsageKind
    model: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    seconds: float
