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

from . import retry
from .errors import ModelError


@dataclass(frozen=True)
class Message:
    """One turn of a conversation: the part it plays and the text it carries."""

    role: str  # "system", "user" or "assistant"
    content: str


@dataclass(frozen=True)
class Reply:
    """What the model answered, and what the answer consumed."""

    text: str
    model: str  # the model that answered, as the provider named it
    prompt_tokens: int  # 0 when the provider reported no usage
    completion_tokens: int


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
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._attempts = max(1, attempts)
        self._temperature = temperature
        self._sleep = sleep
        self._transport = transport

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
        return _reply(response)

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


def _reply(response: httpx.Response) -> Reply:
    try:
        payload = response.json()
    except ValueError:
        raise ModelError("对话响应不是 JSON") from None
    if not isinstance(payload, dict):
        raise ModelError("对话响应不是对象")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ModelError("对话响应没有 choices")
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str):
        raise ModelError("对话响应没有文本")
    usage = payload.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return Reply(
        text=text,
        model=str(payload.get("model") or ""),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )
