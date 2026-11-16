"""How long one node may run before its stage budget is spent.

The budget scales with the recording, because every stage's work does: a
three-hour transcription is not late at the same minute as a ten-minute one. A
floor keeps short clips from receiving an unusably small window, and one scale
lets a slower machine loosen every stage at once. The answer stages do not scale
with the recording; they scale with the request timeout instead, and the
preliminary gets half the final answer's window because it is optional.

The values are deliberately generous. A budget catches a stage that has stopped
making progress, not one that is merely slower than expected.
"""

from __future__ import annotations

from ..config.schema import RuntimeLimits

ANSWER_FLOOR_SECONDS = 600.0
ANSWER_TIMEOUTS = 6.0
PRELIMINARY_FLOOR_SECONDS = 300.0
PRELIMINARY_TIMEOUTS = 3.0

# One node's expected work relative to the recording's length.
_MULTIPLIER: dict[str, float] = {
    "probe": 0.05,
    "extract_audio": 0.5,
    "transcribe": 1.5,
    "build": 0.5,  # coverage map and vectors
    "describe": 1.5,  # scene detection and one call per scene
    "extract": 0.5,  # one call over the whole timeline
}
_DEFAULT_MULTIPLIER = 1.0


def stage_budget(action: str, duration_ms: int, runtime: RuntimeLimits) -> float:
    """Seconds one node may run before the scheduler marks it timed out.

    ``duration_ms`` is the source recording's length, or zero before the probe
    has recorded it. An answer scales with the configured request timeout instead,
    because it costs a fixed number of model calls rather than a slice of media.
    The result is at least the configured floor, multiplied by the configured
    scale, and capped by the configured ceiling when one is set.
    """
    if action == "answer":
        base = max(ANSWER_FLOOR_SECONDS, runtime.request_timeout_seconds * ANSWER_TIMEOUTS)
    elif action == "answer_preliminary":
        base = max(
            PRELIMINARY_FLOOR_SECONDS,
            runtime.request_timeout_seconds * PRELIMINARY_TIMEOUTS,
        )
    else:
        seconds = max(0, duration_ms) / 1000
        base = max(
            float(runtime.stage_budget_floor_seconds),
            seconds * _MULTIPLIER.get(action, _DEFAULT_MULTIPLIER),
        )
    budget = base * runtime.stage_budget_scale
    if runtime.stage_budget_ceiling_seconds:
        budget = min(budget, float(runtime.stage_budget_ceiling_seconds))
    return budget
