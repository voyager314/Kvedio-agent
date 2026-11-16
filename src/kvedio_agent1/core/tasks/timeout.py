"""A stage that ran past the time the pipeline allowed it."""

from __future__ import annotations


class TaskTimeout(Exception):
    """Raised inside a stage when its wall-clock budget is spent.

    Distinct from cancellation: nothing asked the job to stop, the stage simply
    took longer than the budget derived for its kind and the recording's length.
    The scheduler records it as a failed task, so the other branch can still
    answer.
    """

    def __init__(self, action: str, seconds: float) -> None:
        super().__init__(f"{action} 阶段超过预算 {seconds:.0f} 秒")
        self.action = action
        self.seconds = seconds
