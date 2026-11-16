"""Turning text into vectors with the configured embedding endpoint.

The URL is used as written. Texts go out in batches, and the answer is put back in
request order using the provider's own indices: a provider that answers out of
order would otherwise attach a vector to the wrong sentence.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from ..domain import Usage, UsageKind
from . import retry
from .errors import ModelError


class Embedder:
    """One embedding endpoint, addressed by the complete URL the user configured."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        model: str,
        timeout: float,
        batch_size: int = 32,
        attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        transport: httpx.BaseTransport | None = None,
        usage: Callable[[Usage], None] | None = None,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._batch_size = max(1, batch_size)
        self._attempts = max(1, attempts)
        self._sleep = sleep
        self._transport = transport
        self._usage = usage

    def __repr__(self) -> str:
        return f"Embedder(url={self._url!r}, model={self._model!r})"

    @property
    def model(self) -> str:
        """The model name, part of every key an index is stored under."""
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per text, in the order the texts were given."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        reported = 0
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            response = retry.send(
                lambda batch=batch: self._post(batch),
                attempts=self._attempts,
                sleep=self._sleep,
            )
            batch_vectors, prompt_tokens = _vectors(response, len(batch))
            vectors.extend(batch_vectors)
            reported += prompt_tokens
        if self._usage is not None:
            self._usage(Usage(kind=UsageKind.EMBEDDING, model=self._model, prompt_tokens=reported))
        return vectors

    def _post(self, batch: Sequence[str]) -> httpx.Response:
        with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
            return client.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self._model, "input": list(batch)},
            )


def _vectors(response: httpx.Response, expected: int) -> tuple[list[list[float]], int]:
    try:
        payload: Any = response.json()
    except ValueError:
        raise ModelError("嵌入响应不是 JSON") from None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or len(data) != expected:
        raise ModelError(f"嵌入响应条数不符：期望 {expected}")
    items = [item for item in data if isinstance(item, dict)]
    if len(items) != len(data):
        raise ModelError("嵌入响应格式无法识别")
    if all(isinstance(item.get("index"), int) for item in items):
        items = sorted(items, key=lambda item: item["index"])
    vectors: list[list[float]] = []
    for item in items:
        embedding = item.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ModelError("嵌入响应缺少向量")
        vectors.append([float(value) for value in embedding])
    if len({len(vector) for vector in vectors}) != 1:
        raise ModelError("嵌入向量的维度不一致")
    usage = payload.get("usage") if isinstance(payload, dict) else None
    tokens = int(usage.get("prompt_tokens") or 0) if isinstance(usage, dict) else 0
    return vectors, tokens
