"""Grouping the shots of a recording into the events a reader would name.

One call, one reading: events are a whole-recording summary, so they are written
all at once rather than resumed piece by piece. If the model's reply cannot be
used, the shots are grouped as they came instead of failing the job — a rough
timeline still beats no timeline.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..domain import Scene, TimelineEvent, VideoRecord
from ..llm import Chat, Message
from ..prompts import render
from ..storage import Storage

MAX_EVENTS = 40
FALLBACK_GROUP = 5  # scenes per event when the model's answer is unusable
Report = Callable[[float, str | None], None]


@dataclass(frozen=True)
class EventStats:
    """How many events the recording was cut into."""

    events: int


class EventBuilder:
    """Turns scene descriptions into a named timeline for one video."""

    def __init__(
        self, store: Storage, chat: Chat, *, model: str = "", max_events: int = MAX_EVENTS
    ) -> None:
        self._store = store
        self._chat = chat
        self._model = model
        self._max_events = max(1, max_events)

    def key(self, video: VideoRecord, scene_key: str) -> str:
        return hashlib.sha256(f"{scene_key}|{self._model}".encode()).hexdigest()[:16]

    def current(self, video: VideoRecord) -> bool:
        """Whether the events match the shots they were built from."""
        run = self._store.scene_run(video.id)
        if run is None:
            return True  # no shots means nothing to group; the stage is not needed
        last = self._store.event_run(video.id)
        return last is not None and last[0] == self.key(video, run[0])

    def build(self, video: VideoRecord, report: Report) -> EventStats:
        """Read the shots, ask for events, and store the whole list in one go."""
        if self.current(video):
            return EventStats(events=len(self._store.events(video.id)))
        run = self._store.scene_run(video.id)
        if run is None:
            return EventStats(events=0)
        scenes = self._store.scenes(video.id)
        if not scenes:
            return EventStats(events=0)
        key = self.key(video, run[0])
        duration = video.media.duration_ms if video.media is not None else scenes[-1].end_ms
        report(0.1, "抽取事件")
        events = self._extract(scenes, duration)
        self._store.replace_events(video.id, key, events, self._model)
        self._store.save_event_run(video.id, key, len(events))
        report(1.0, f"事件 {len(events)} 个")
        return EventStats(events=len(events))

    def _extract(self, scenes: Sequence[Scene], duration_ms: int) -> list[TimelineEvent]:
        message = render(
            "events",
            duration_ms=duration_ms,
            scenes=scenes,
            max_events=self._max_events,
        )
        reply = self._chat.complete([Message(role="system", content=message)], json_mode=True)
        events = _parse(reply.text, duration_ms, self._max_events)
        return events or _fallback(scenes)


def _parse(text: str, duration_ms: int, limit: int) -> list[TimelineEvent]:
    payload = _json_object(text)
    raw = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    events: list[TimelineEvent] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        start = _seconds(item.get("start"))
        end = _seconds(item.get("end"))
        title = item.get("title")
        summary = item.get("summary", "")
        if start is None or end is None or not isinstance(title, str) or not title.strip():
            continue
        start_ms = max(0, min(start * 1000, duration_ms))
        end_ms = max(0, min(end * 1000, duration_ms))
        if end_ms <= start_ms:
            continue
        events.append(
            TimelineEvent(
                start_ms=start_ms,
                end_ms=end_ms,
                title=" ".join(title.split())[:80],
                summary=" ".join(str(summary).split())[:200],
            )
        )
    return _merge(events)[:limit]


def _merge(events: list[TimelineEvent]) -> list[TimelineEvent]:
    """Sort by start and fold overlaps, so the timeline has no double-booked stretch."""
    ordered = sorted(events, key=lambda item: (item.start_ms, item.end_ms))
    merged: list[TimelineEvent] = []
    for event in ordered:
        if merged and event.start_ms < merged[-1].end_ms:
            last = merged[-1]
            merged[-1] = TimelineEvent(
                start_ms=last.start_ms,
                end_ms=max(last.end_ms, event.end_ms),
                title=last.title,
                summary=last.summary,
            )
            continue
        merged.append(event)
    return merged


def _fallback(scenes: Sequence[Scene]) -> list[TimelineEvent]:
    """Group the shots as they came: no naming, but a timeline a reader can still use."""
    events: list[TimelineEvent] = []
    for start in range(0, len(scenes), FALLBACK_GROUP):
        batch = scenes[start : start + FALLBACK_GROUP]
        first = batch[0].description
        summary = "；".join(scene.description for scene in batch)
        events.append(
            TimelineEvent(
                start_ms=batch[0].start_ms,
                end_ms=batch[-1].end_ms,
                title=first[:40],
                summary=summary[:200],
            )
        )
    return events


def _json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _seconds(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)
