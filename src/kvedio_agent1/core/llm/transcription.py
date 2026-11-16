"""Sending one audio file to a transcription endpoint and reading its segments.

The URL comes from the configuration and is used exactly as written: providers
mount their endpoints at different paths, so nothing is appended here.

Retry policy and error shape come from ``retry``: what is worth a second attempt
is decided once for every model call, not per client.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from ..domain import Segment
from . import retry
from .errors import ModelError


class Transcriber:
    """One transcription endpoint, addressed by the complete URL the user configured."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        model: str,
        timeout: float,
        attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._attempts = max(1, attempts)
        self._sleep = sleep
        self._transport = transport

    def __repr__(self) -> str:
        return f"Transcriber(url={self._url!r}, model={self._model!r})"

    def transcribe(self, audio: Path) -> list[Segment]:
        """Return the segments the endpoint reported, timed from the start of the file.

        Raises ``ModelError`` when the request cannot be completed or the answer
        carries no timing: a segment without timestamps cannot be cited.
        """
        response = retry.send(
            lambda: self._post(audio), attempts=self._attempts, sleep=self._sleep
        )
        return _segments(response)

    def _post(self, audio: Path) -> httpx.Response:
        with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
            return client.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                files={"file": (audio.name, audio.read_bytes(), "audio/wav")},
                data={"model": self._model, "response_format": "verbose_json"},
            )


def _segments(response: httpx.Response) -> list[Segment]:
    payload = _json(response)
    items = payload.get("segments")
    if not isinstance(items, list):
        raise ModelError("转写响应没有分段（缺少 segments）")
    segments: list[Segment] = []
    for item in items:
        if not isinstance(item, dict):
            raise ModelError("转写响应分段格式无法识别")
        start = item.get("start")
        end = item.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ModelError("转写响应缺少分段时间戳")
        text = " ".join(str(item.get("text", "")).split())
        if not text or end <= start:
            continue
        segments.append(
            Segment(start_ms=round(start * 1000), end_ms=round(end * 1000), text=text)
        )
    return segments


def _json(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        raise ModelError("转写响应不是 JSON") from None
    if not isinstance(payload, dict):
        raise ModelError("转写响应不是对象")
    return payload
