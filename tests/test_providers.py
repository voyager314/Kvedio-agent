from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from kvedio_agent.platform.costs import ProviderGate
from kvedio_agent.platform.providers import (
    Completion,
    LiteLLMChat,
    OpenAICompatibleASR,
    OpenAICompatibleVision,
    ProviderError,
    _parse_transcript,
)
from kvedio_agent.platform.settings import ModelEntry
from kvedio_agent.platform.storage import Store


def test_transcript_normalization_applies_segment_offset_and_word_timing() -> None:
    result = _parse_transcript(
        {"segments": [{"start": 0.5, "end": 1.5, "text": " hello ",
                       "words": [{"word": "hello", "start": 0.5, "end": 1.5}]}]},
        10_000,
    )
    assert result[0].start_ms == 10_500
    assert result[0].end_ms == 11_500
    assert result[0].text == "hello"
    assert result[0].words[0].start_ms == 10_500


def test_provider_errors_are_bounded_and_secret_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ASR_TOKEN", "sentinel-secret")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    provider = OpenAICompatibleASR(
        ModelEntry(model="asr", api_base="https://provider.example/v1",
                   api_key_env="ASR_TOKEN", roles=["asr"])
    )

    def fail(*args: object, **kwargs: object) -> httpx.Response:
        request = httpx.Request("POST", "https://provider.example/v1/audio/transcriptions")
        return httpx.Response(429, request=request, text=json.dumps({"error": "secret"}))

    monkeypatch.setattr(httpx, "post", fail)
    with pytest.raises(ProviderError) as raised:
        provider.transcribe(audio, 0)
    assert raised.value.category == "provider_rate_limited_or_unavailable"
    assert "sentinel-secret" not in str(raised.value)


def test_chat_adapter_suppresses_raw_exception_and_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHAT_TOKEN", "sentinel-secret")
    model = ModelEntry(
        model="provider/model", api_base="https://provider.example/v1",
        api_key_env="CHAT_TOKEN", roles=["chat"],
    )
    adapter = LiteLLMChat(model)

    def fail(**kwargs: object) -> None:
        raise RuntimeError("authorization=sentinel-secret")

    monkeypatch.setattr("litellm.completion", fail)
    with pytest.raises(ProviderError) as raised:
        adapter.complete("system", "prompt")
    assert "sentinel-secret" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_chat_completion_returns_provider_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_TOKEN", "chat-secret")
    model = ModelEntry(
        model="provider/model", api_key_env="CHAT_TOKEN", roles=["chat"],
        input_cost_per_million=1, output_cost_per_million=2,
    )

    class Usage:
        prompt_tokens = 12
        completion_tokens = 7

    class Message:
        content = "answer"

    class Response:
        choices = [type("Choice", (), {"message": Message()})()]
        usage = Usage()
        model = "resolved/model"

    monkeypatch.setattr("litellm.completion", lambda **kwargs: Response())
    result = LiteLLMChat(model).complete("system", "prompt")
    assert isinstance(result, Completion)
    assert result.model == "resolved/model"
    assert result.input_tokens == 12
    assert result.output_tokens == 7
    assert result.actual_cost_usd == pytest.approx(0.000026)


def test_chat_adapter_accepts_tool_call_without_text_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHAT_TOKEN", "chat-secret")
    model = ModelEntry(model="provider/model", api_key_env="CHAT_TOKEN", roles=["chat"])

    class Function:
        name = "search_evidence"
        arguments = {"query": "launch"}

    class Message:
        content = None
        tool_calls = [type("Call", (), {"id": "call-1", "function": Function()})()]

    class Response:
        choices = [type("Choice", (), {"message": Message()})()]
        usage = None
        model = "resolved/model"

    captured: dict[str, object] = {}

    def complete(**kwargs: object) -> Response:
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr("litellm.completion", complete)
    result = LiteLLMChat(model).complete(
        "system", "prompt", [{"type": "function", "function": {"name": "search_evidence"}}]
    )
    assert result.content == ""
    assert result.tool_calls[0]["name"] == "search_evidence"
    assert result.tool_calls[0]["arguments"] == '{"query": "launch"}'
    assert captured["tool_choice"] == "auto"


def test_visual_observation_preserves_frame_source_timestamps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VISION_TOKEN", "vision-secret")
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpeg")
    model = ModelEntry(
        model="vision/model", api_base="https://provider.example/v1",
        api_key_env="VISION_TOKEN", roles=["vision"],
    )

    def response(*args: object, **kwargs: object) -> httpx.Response:
        request = httpx.Request("POST", "https://provider.example/v1/chat/completions")
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": "A slide"}}]},
        )

    monkeypatch.setattr(httpx, "post", response)
    result = OpenAICompatibleVision(model).describe([(frame, 1250)], 1000, 1500)
    assert result.frame_times_ms == (1250,)
    assert result.start_ms == 1000


def test_vision_usage_is_persisted_as_actual_cost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VISION_TOKEN", "vision-secret")
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"image")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    store = Store(tmp_path / "workspace")
    source_id = store.register_source(source, 6, source.stat().st_mtime_ns, "digest")
    job_id = store.create_job(source_id, "objective", {})
    model = ModelEntry(
        model="vision/model", api_base="https://provider.example/v1",
        api_key_env="VISION_TOKEN", roles=["vision"],
        input_cost_per_million=1, output_cost_per_million=2,
    )

    def response(*args: object, **kwargs: object) -> httpx.Response:
        assert kwargs["json"]["max_tokens"] == 2048  # type: ignore[index]
        request = httpx.Request("POST", "https://provider.example/v1/chat/completions")
        return httpx.Response(
            200, request=request,
            json={"choices": [{"message": {"content": "A slide"}}],
                  "usage": {"prompt_tokens": 20, "completion_tokens": 5}},
        )

    monkeypatch.setattr(httpx, "post", response)
    provider = OpenAICompatibleVision(
        model, gate=ProviderGate(store, model, None), job_id=job_id
    )
    provider.describe([(frame, 1250)], 1000, 1500)
    summary = store.usage_summary(job_id)
    assert summary["actual_cost_usd"] == pytest.approx(0.00003)
    assert summary["estimated_only_calls"] == 0
