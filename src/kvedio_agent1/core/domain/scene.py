"""What one shot in the recording looks like, in words."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scene:
    """One shot of the recording and one line about what is visible in it.

    The description is an observation, not a quote: it says what the picture
    showed. ``frame_path`` points at the still the description was written from,
    so a reader can check the claim.
    """

    start_ms: int
    end_ms: int
    description: str
    frame_path: str
