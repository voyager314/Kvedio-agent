"""Task scheduling: status rules, cancellation, event delivery, bounded execution."""

from .bus import EventBus
from .cancellation import Cancellation, Cancelled, stop_if
from .scheduler import Handler, Scheduler, TaskContext
from .state import IllegalTransition, check_transition

__all__ = [
    "Cancellation",
    "Cancelled",
    "EventBus",
    "Handler",
    "IllegalTransition",
    "Scheduler",
    "TaskContext",
    "check_transition",
    "stop_if",
]
