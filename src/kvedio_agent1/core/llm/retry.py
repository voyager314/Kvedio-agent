"""Retrying calls whose failure may be temporary.

One place decides what is worth a second attempt. A rejected request is reported
at once, because repeating it spends money without changing the answer; a
timeout, a rate limit or a server fault is attempted again after a growing pause.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from .errors import ModelError

RETRYABLE = frozenset({408, 409, 425, 429})


def send(
    call: Callable[[], httpx.Response],
    *,
    attempts: int,
    sleep: Callable[[float], None],
) -> httpx.Response:
    """Call until it answers successfully or the attempts run out.

    Transport failures and retryable statuses wait 1s, 2s, 4s… (at most 30s)
    between attempts. Anything else raises immediately with ``describe``.
    """
    failure: ModelError | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            response = call()
        except httpx.TransportError as error:
            failure = ModelError(f"请求未送达：{type(error).__name__}")
        else:
            if response.status_code < 400:
                return response
            failure = describe(response)
            if not retryable(response.status_code):
                raise failure
        if attempt < attempts:
            sleep(min(30.0, 2.0 ** (attempt - 1)))
    if failure is None:
        raise ModelError("请求没有发出")
    raise failure


def describe(response: httpx.Response) -> ModelError:
    """Build the failure for a refusing response from its status and provider message."""
    detail = ""
    try:
        payload: Any = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        message = error.get("message") if isinstance(error, dict) else error
        if isinstance(message, str) and message.strip():
            detail = "：" + message.strip().splitlines()[0][:200]
    return ModelError(f"HTTP {response.status_code}{detail}")


def retryable(status: int) -> bool:
    """Whether a refusal of this status may be worth another attempt."""
    return status in RETRYABLE or status >= 500
