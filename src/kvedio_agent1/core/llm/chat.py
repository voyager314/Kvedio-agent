"""Calling a chat completion endpoint that the configuration points at.

The URL is used exactly as written. Messages are plain turns; the caller decides
what goes into them, so this client holds no prompt and no policy — those belong
to the stage that asks.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from ..domain import Usage, UsageKind
from . import retry
from .errors import ModelError
from .reply import Reply, parse_reply


class _Retryable(ModelError):
    """A refusal that happened before any text was emitted, so a retry costs nothing visible."""


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

    def stream(
        self, messages: Sequence[Message], on_delta: Callable[[str], None]
    ) -> Reply:
        """Stream one answer, handing each piece to ``on_delta`` as it arrives.

        A refusal or transport failure before the first piece is retried like any
        other call. Once text has been handed over, a failure is raised instead of
        retried: repeating the request would repeat what the caller already showed.
        """
        state = {"emitted": False}

        def emit(piece: str) -> None:
            state["emitted"] = True
            on_delta(piece)

        for attempt in range(1, self._attempts + 1):
            state["emitted"] = False
            try:
                return self._stream_once(messages, emit)
            except _Retryable:
                if state["emitted"] or attempt == self._attempts:
                    raise
                self._sleep(min(30.0, 2.0 ** (attempt - 1)))
        raise ModelError("请求没有发出")

    def _stream_once(
        self, messages: Sequence[Message], emit: Callable[[str], None]
    ) -> Reply:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": item.role, "content": item.content} for item in messages],
            "temperature": self._temperature,
            "stream": True,
        }
        try:
            with (
                httpx.Client(timeout=self._timeout, transport=self._transport) as client,
                client.stream(
                    "POST",
                    self._url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                ) as response,
            ):
                if response.status_code >= 400:
                    response.read()
                    failure = retry.describe(response)
                    if retry.retryable(response.status_code):
                        raise _Retryable(str(failure))
                    raise failure
                return self._read_stream(response, emit)
        except httpx.TransportError as error:
            raise _Retryable(f"请求未送达：{type(error).__name__}") from error

    def _read_stream(
        self, response: httpx.Response, emit: Callable[[str], None]
    ) -> Reply:
        pieces: list[str] = []
        model = self._model
        prompt_tokens = 0
        completion_tokens = 0
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            if not isinstance(chunk, dict):
                continue
            model = str(chunk.get("model") or model)
            for choice in chunk.get("choices") or []:
                delta = (choice or {}).get("delta") or {}
                piece = delta.get("content")
                if isinstance(piece, str) and piece:
                    pieces.append(piece)
                    emit(piece)
            usage = chunk.get("usage")
            if isinstance(usage, dict):
                prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
        if not pieces:
            raise ModelError("流式响应没有内容")
        reply = Reply(
            text="".join(pieces),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        if self._usage is not None:
            self._usage(
                Usage(
                    kind=UsageKind.CHAT,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
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
