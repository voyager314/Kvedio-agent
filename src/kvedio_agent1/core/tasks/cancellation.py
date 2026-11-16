"""A one-way signal from the interface to a run that should stop.

Cancellation is cooperative: nothing is killed. A stage looks at the signal
between units of work — a chunk, a batch, a scene — and raises ``Cancelled``,
which the scheduler records as a cancelled task rather than a failure. Work
already committed stays committed, so a cancelled job resumes like an
interrupted one.
"""

from __future__ import annotations

import threading
from collections.abc import Callable


class Cancelled(Exception):
    """Raised inside a stage that noticed the run was cancelled."""


class Cancellation:
    """A signal that can be raised once and is never cleared."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Stop the calling stage when the signal has been raised."""
        if self._event.is_set():
            raise Cancelled()


def stop_if(cancelled: Callable[[], bool] | None) -> None:
    """Raise ``Cancelled`` when the caller's signal says to stop.

    A stage calls this between units of work — a chunk, a batch, a scene — so the
    run stops within one unit rather than after the whole stage.
    """
    if cancelled is not None and cancelled():
        raise Cancelled()
