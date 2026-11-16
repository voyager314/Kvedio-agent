import json

import httpx
import pytest

from kvedio_agent1.core.domain import UsageKind
from kvedio_agent1.core.llm import Chat, Message, ModelError

URL = "https://chat.example.invalid/v1/chat/completions"
KEY = "sk-live-do-not-print"


def build(handler, **options) -> Chat:
    return Chat(
        url=URL,
        api_key=KEY,
        model="chat-model",
        timeout=5,
        transport=httpx.MockTransport(handler),
        **options,
    )


def answer(text: str, **usage) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "chat-model-0613",
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": usage or {"prompt_tokens": 12, "completion_tokens": 7},
        },
    )


def test_the_reply_carries_text_model_and_usage():
    reply = build(lambda request: answer("你好")).complete([Message("user", "hi")])
    assert reply.text == "你好"
    assert reply.model == "chat-model-0613"
    assert (reply.prompt_tokens, reply.completion_tokens) == (12, 7)


def test_the_request_uses_the_configured_address_and_roles():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.read())
        return answer("ok")

    build(handler).complete([Message("system", "规则"), Message("user", "问题")])

    assert seen["url"] == URL
    assert seen["auth"] == f"Bearer {KEY}"
    body = seen["body"]
    assert body["model"] == "chat-model"
    assert body["messages"] == [
        {"role": "system", "content": "规则"},
        {"role": "user", "content": "问题"},
    ]
    assert "response_format" not in body


def test_json_mode_is_requested_when_asked():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.read())
        return answer("{}")

    build(handler).complete([Message("user", "plan")], json_mode=True)
    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_a_server_fault_is_retried_once():
    calls: list[int] = []
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(502, text="bad gateway")
        return answer("第二次成功")

    reply = build(handler, sleep=delays.append).complete([Message("user", "hi")])
    assert reply.text == "第二次成功"
    assert len(calls) == 2
    assert delays == [1.0]


def test_a_rejected_key_is_reported_without_retrying_or_leaking():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    with pytest.raises(ModelError) as failure:
        build(handler).complete([Message("user", "hi")])
    message = str(failure.value)
    assert len(calls) == 1
    assert "HTTP 401" in message
    assert "invalid api key" in message
    assert KEY not in message


def test_a_reply_without_choices_is_refused():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "m"})

    with pytest.raises(ModelError, match="choices"):
        build(handler).complete([Message("user", "hi")])


def test_the_repr_names_the_endpoint_and_model_only():
    chat = build(lambda request: answer("ok"))
    assert KEY not in repr(chat)
    assert URL in repr(chat)

def test_each_reply_reports_its_usage():
    seen = []

    build(lambda request: answer("你好"), usage=seen.append).complete([Message("user", "hi")])

    assert len(seen) == 1
    assert seen[0].kind is UsageKind.CHAT
    assert seen[0].model == "chat-model-0613"
    assert (seen[0].prompt_tokens, seen[0].completion_tokens) == (12, 7)

def sse(*pieces: str, usage: dict | None = None) -> httpx.Response:
    body = b""
    for piece in pieces:
        chunk = {"model": "chat-model-stream", "choices": [{"delta": {"content": piece}}]}
        body += b"data: " + json.dumps(chunk).encode() + b"\n\n"
    if usage is not None:
        tail = {"model": "chat-model-stream", "choices": [], "usage": usage}
        body += b"data: " + json.dumps(tail).encode() + b"\n\n"
    body += b"data: [DONE]\n\n"
    return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


def test_streaming_hands_over_each_piece():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.read())["stream"] is True
        return sse("预算", "还差", "二十万")

    reply = build(handler).stream([Message("user", "hi")], seen.append)

    assert reply.text == "预算还差二十万"
    assert "".join(seen) == reply.text
    assert reply.model == "chat-model-stream"


def test_a_stream_failure_before_any_text_is_retried():
    calls: list[int] = []
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, text="busy")
        return sse("ok")

    reply = build(handler, sleep=delays.append).stream([Message("user", "x")], lambda piece: None)

    assert reply.text == "ok"
    assert len(calls) == 2
    assert delays == [1.0]


def test_streaming_reports_the_usage_of_the_final_chunk():
    seen = []
    usage = {"prompt_tokens": 5, "completion_tokens": 7}

    def handler(request: httpx.Request) -> httpx.Response:
        return sse("ok", usage=usage)

    build(handler, usage=seen.append).stream([Message("user", "x")], lambda piece: None)

    assert seen[0].prompt_tokens == 5
    assert seen[0].completion_tokens == 7
