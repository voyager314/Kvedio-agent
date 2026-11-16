"""Calling a chat completion endpoint that the configuration points at.

The URL is used exactly as written. Messages are plain turns; the caller decides
what goes into them, so this client holds no prompt and no policy — those belong
to the stage that asks.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from ..domain import Usage, UsageKind
from . import retry
from .reply import Reply, parse_reply


@dataclass(frozen=True)
class Message:
    """One turn of a conversation: the part it plays and the text it carries."""

    role: str  # "system", "user" or "assistant"
    content: str


class Chat:
    """One chat endpoint, addressed by the complete URL the user configured."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        model: str,
        timeout: float,
        attempts: int = 3,
        temperature: float = 0.2,
        sleep: Callable[[float], None] = time.sleep,
        transport: httpx.BaseTransport | None = None,
        usage: Callable[[Usage], None] | None = None,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._attempts = max(1, attempts)
        self._temperature = temperature
        self._sleep = sleep
        self._transport = transport
        self._usage = usage

    def __repr__(self) -> str:
        return f"Chat(url={self._url!r}, model={self._model!r})"

    def complete(self, messages: Sequence[Message], *, json_mode: bool = False) -> Reply:
        """Return the first choice's text.

        ``json_mode`` asks the provider to constrain the answer to a JSON object;
        the caller still validates it, because the constraint is a request rather
        than a guarantee.
        """
        response = retry.send(
            lambda: self._post(messages, json_mode),
            attempts=self._attempts,
            sleep=self._sleep,
        )
        reply = parse_reply(response)
        if self._usage is not None:
            self._usage(
                Usage(
                    kind=UsageKind.CHAT,
                    model=reply.model or self._model,
                    prompt_tokens=reply.prompt_tokens,
                    completion_tokens=reply.completion_tokens,
                )
            )
        return reply

    def _post(self, messages: Sequence[Message], json_mode: bool) -> httpx.Response:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": item.role, "content": item.content} for item in messages],
            "temperature": self._temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
            return client.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
