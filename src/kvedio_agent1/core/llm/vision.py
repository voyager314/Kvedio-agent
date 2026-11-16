"""Turning one picture into words with a vision-capable chat endpoint.

The image travels inline as a data URI: a local file has no URL the provider could
fetch, and uploading it separately would need a second endpoint and a second key.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from . import retry
from .reply import Reply, parse_reply


class Vision:
    """One vision endpoint, addressed by the complete URL the user configured."""

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
        return f"Vision(url={self._url!r}, model={self._model!r})"

    @property
    def model(self) -> str:
        """The model name, part of every key a scene description is stored under."""
        return self._model

    def describe(self, image: Path, prompt: str) -> str:
        """One description of what the picture shows, as the provider answered it."""
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        response = retry.send(
            lambda: self._post(encoded, prompt),
            attempts=self._attempts,
            sleep=self._sleep,
        )
        reply: Reply = parse_reply(response)
        return " ".join(reply.text.split())

    def _post(self, encoded: str, prompt: str) -> httpx.Response:
        with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
            return client.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "temperature": 0.2,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                                },
                            ],
                        }
                    ],
                },
            )
