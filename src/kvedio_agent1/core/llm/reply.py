"""Reading a chat completion response, shared by every client that gets one."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from .errors import ModelError


@dataclass(frozen=True)
class Reply:
    """What the model answered, and what the answer consumed."""

    text: str
    model: str  # the model that answered, as the provider named it
    prompt_tokens: int  # 0 when the provider reported no usage
    completion_tokens: int


def parse_reply(response: httpx.Response) -> Reply:
    """The first choice's text, or a failure naming what the response lacked."""
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
