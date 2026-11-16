"""Sending one audio file to a transcription endpoint and reading its segments.

The URL comes from the configuration and is used exactly as written: providers
mount their endpoints at different paths, so nothing is appended here.

Only failures that can succeed on a second attempt are retried. A rejected
request is reported at once — repeating it spends money without changing the
answer — and no message ever carries the credential.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from ..domain import Segment

_RETRYABLE = frozenset({408, 409, 425, 429})


class TranscriptionError(Exception):
    """The endpoint refused the file, answered unusably, or stayed unreachable.

    The message names the status and the provider's own message when it sent one.
    It never repeats the request headers, the key, or the audio itself.
    """


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

        Raises ``TranscriptionError`` when the request cannot be completed or the
        answer carries no timing: a segment without timestamps cannot be cited.
        """
        failure: TranscriptionError | None = None
        for attempt in range(1, self._attempts + 1):
            try:
                response = self._post(audio)
            except httpx.TransportError as error:
                failure = TranscriptionError(f"转写请求未送达：{type(error).__name__}")
            else:
                if response.status_code < 400:
                    return _segments(response)
                failure = _refusal(response)
                if not _retryable(response.status_code):
                    raise failure
            if attempt < self._attempts:
                self._sleep(min(30.0, 2.0 ** (attempt - 1)))
        if failure is None:
            raise TranscriptionError("转写请求没有发出")
        raise failure

    def _post(self, audio: Path) -> httpx.Response:
        with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
            return client.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                files={"file": (audio.name, audio.read_bytes(), "audio/wav")},
                data={"model": self._model, "response_format": "verbose_json"},
            )


def _retryable(status: int) -> bool:
    return status in _RETRYABLE or status >= 500


def _segments(response: httpx.Response) -> list[Segment]:
    payload = _json(response)
    items = payload.get("segments")
    if not isinstance(items, list):
        raise TranscriptionError("转写响应没有分段（缺少 segments）")
    segments: list[Segment] = []
    for item in items:
        if not isinstance(item, dict):
            raise TranscriptionError("转写响应分段格式无法识别")
        start = item.get("start")
        end = item.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise TranscriptionError("转写响应缺少分段时间戳")
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
        raise TranscriptionError("转写响应不是 JSON") from None
    if not isinstance(payload, dict):
        raise TranscriptionError("转写响应不是对象")
    return payload


def _refusal(response: httpx.Response) -> TranscriptionError:
    detail = ""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        message = error.get("message") if isinstance(error, dict) else error
        if isinstance(message, str) and message.strip():
            detail = "：" + message.strip().splitlines()[0][:200]
    return TranscriptionError(f"转写失败：HTTP {response.status_code}{detail}")
