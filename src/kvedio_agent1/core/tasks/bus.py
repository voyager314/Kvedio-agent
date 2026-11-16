"""Delivery of events from the threads that produce them to the observers that render them.

Publishing never blocks and never fails the work: each subscriber owns a bounded
queue, and one that stops draining loses its oldest events instead of stalling
the job that produces them.
"""

from __future__ import annotations

import queue
import threading

from ..domain import Event

DEFAULT_CAPACITY = 8192


class EventBus:
    """Fan-out of published events to every current subscriber."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self._capacity = capacity
        self._lock = threading.Lock()
        self._channels: list[queue.Queue[Event]] = []

    def subscribe(self) -> queue.Queue[Event]:
        """A queue receiving every event published after this call."""
        channel: queue.Queue[Event] = queue.Queue(maxsize=self._capacity)
        with self._lock:
            self._channels.append(channel)
        return channel

    def unsubscribe(self, channel: queue.Queue[Event]) -> None:
        """Stop delivering to a queue; the caller keeps what it has not drained."""
        with self._lock:
            if channel in self._channels:
                self._channels.remove(channel)

    def publish(self, event: Event) -> None:
        """Hand the event to every subscriber without blocking the caller."""
        with self._lock:
            channels = tuple(self._channels)
        for channel in channels:
            try:
                channel.put_nowait(event)
            except queue.Full:
                continue
