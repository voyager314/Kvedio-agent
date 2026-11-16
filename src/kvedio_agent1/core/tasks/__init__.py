"""Task scheduling: status rules, cancellation, event delivery, bounded execution."""

from .budget import stage_budget
from .bus import EventBus
from .cancellation import Cancellation, Cancelled, stop_if
from .scheduler import Handler, Scheduler, TaskContext
from .state import IllegalTransition, check_transition
from .timeout import TaskTimeout

__all__ = [
    "Cancellation",
    "Cancelled",
    "EventBus",
    "Handler",
    "IllegalTransition",
    "Scheduler",
    "TaskContext",
    "TaskTimeout",
    "check_transition",
    "stage_budget",
    "stop_if",
]
