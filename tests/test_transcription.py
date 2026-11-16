import json
from pathlib import Path

import httpx
import pytest

from kvedio_agent1.core.domain import Segment
from kvedio_agent1.core.llm import Transcriber, TranscriptionError

URL = "https://asr.example.invalid/v1/audio/transcriptions"
KEY = "sk-live-do-not-print"


@pytest.fixture
def audio(tmp_path) -> Path:
    target = tmp_path / "chunk.wav"
    target.write_bytes(b"RIFF....WAVE")
    return target


def build(handler, **options) -> Transcriber:
    return Transcriber(
        url=URL,
        api_key=KEY,
        model="asr-model",
        timeout=5,
        transport=httpx.MockTransport(handler),
        **options,
    )


def test_segments_are_converted_to_milliseconds(audio):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "segments": [
                    {"start": 0.0, "end": 1.25, "text": " 你好 "},
                    {"start": 1.25, "end": 2.0, "text": "   "},
                ]
            },
        )

    assert build(handler).transcribe(audio) == [Segment(start_ms=0, end_ms=1250, text="你好")]


def test_the_request_uses_the_configured_address_and_model(audio):
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode("utf-8", errors="replace")
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = body
        return httpx.Response(200, json={"segments": [{"start": 0, "end": 1, "text": "hi"}]})

    build(handler).transcribe(audio)

    assert seen["url"] == URL
    assert seen["auth"] == f"Bearer {KEY}"
    assert 'name="model"' in seen["body"]
    assert "verbose_json" in seen["body"]
    assert 'name="file"' in seen["body"]


def test_a_transient_failure_is_retried(audio):
    attempts: list[int] = []
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"segments": [{"start": 0, "end": 1, "text": "hi"}]})

    segments = build(handler, sleep=delays.append).transcribe(audio)

    assert segments == [Segment(start_ms=0, end_ms=1000, text="hi")]
    assert len(attempts) == 2
    assert delays == [1.0]


def test_a_rejected_request_is_not_retried_and_hides_the_key(audio):
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(400, json={"error": {"message": "unsupported format"}})

    with pytest.raises(TranscriptionError) as failure:
        build(handler).transcribe(audio)
    message = str(failure.value)
    assert len(attempts) == 1
    assert "HTTP 400" in message
    assert "unsupported format" in message
    assert KEY not in message


def test_an_unreachable_endpoint_is_reported_after_the_last_attempt(audio):
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectError("refused")

    with pytest.raises(TranscriptionError, match="未送达"):
        build(handler, attempts=2, sleep=lambda _: None).transcribe(audio)
    assert len(attempts) == 2


def test_an_answer_without_segments_is_refused(audio):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "whole transcript"})

    with pytest.raises(TranscriptionError, match="segments"):
        build(handler).transcribe(audio)


def test_a_segment_without_timing_is_refused(audio):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"segments": [{"text": "no timing"}]})

    with pytest.raises(TranscriptionError, match="时间戳"):
        build(handler).transcribe(audio)


def test_an_empty_segment_list_is_a_silent_chunk_not_a_failure(audio):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"segments": []})

    assert build(handler).transcribe(audio) == []


def test_the_repr_names_the_endpoint_and_model_only():
    transcriber = build(lambda request: httpx.Response(200, json={}))
    assert KEY not in repr(transcriber)
    assert URL in repr(transcriber)


def test_json_bodies_are_not_echoed_into_errors(audio):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=json.dumps({"error": "boom"}).encode())

    with pytest.raises(TranscriptionError) as failure:
        build(handler, attempts=1).transcribe(audio)
    assert "boom" in str(failure.value)
