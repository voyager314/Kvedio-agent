"""Vendor-neutral ASR and vision contracts with a redacted HTTP adapter."""

from __future__ import annotations

import base64
import hashlib
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
import litellm

from kvedio_agent.platform.costs import ProviderGate, UsageReservation, call_key
from kvedio_agent.platform.settings import ModelEntry


@dataclass(frozen=True)
class WordTiming:
    """One recognized word and its source-relative timing in milliseconds."""

    text: str  # Recognized token; non-empty when emitted.
    start_ms: int  # Source-relative start in milliseconds; non-negative.
    end_ms: int  # Source-relative end in milliseconds; greater than start_ms.


@dataclass(frozen=True)
class TranscriptSegment:
    """Normalized ASR segment with optional word and speaker evidence."""

    start_ms: int  # Source-relative start in milliseconds; non-negative.
    end_ms: int  # Source-relative end in milliseconds; greater than start_ms.
    text: str  # Recognized speech; empty only when the provider emits no text.
    confidence: float | None = None  # Provider confidence in [0, 1], or None when absent.
    speaker_id: str | None = None  # Provider speaker label, or None when unavailable.
    words: tuple[WordTiming, ...] = ()  # Ordered word timings; empty when not supplied.


@dataclass(frozen=True)
class VisualObservation:
    """Normalized observation tied to a source-relative visual interval."""

    start_ms: int  # Source-relative interval start in milliseconds; non-negative.
    end_ms: int  # Source-relative interval end in milliseconds; greater than start_ms.
    summary: str  # Provider description of the supplied visual evidence.
    ocr_text: str | None = None  # Extracted on-screen text, or None when not supplied.
    confidence: float | None = None  # Provider confidence in [0, 1], or None when absent.
    frame_times_ms: tuple[int, ...] = ()  # Source-relative frame times in milliseconds.
    frame_paths: tuple[str, ...] = ()  # Local generated frame paths; empty when unavailable.


@dataclass(frozen=True)
class Completion:
    """Text completion and provider-reported usage; absent usage remains unknown."""

    content: str  # Generated text; empty only if the provider returns empty content.
    model: str | None = None  # Resolved provider model name, or None when not reported.
    input_tokens: int | None = None  # Provider-reported input token count; None if absent.
    output_tokens: int | None = None  # Provider-reported output token count; None if absent.
    actual_cost_usd: float | None = None  # Derived USD cost; None if pricing or usage is absent.
    tool_calls: tuple[dict[str, Any], ...] = ()  # Provider function calls, empty when absent.


class ProviderError(RuntimeError):
    """Stable provider failure category safe for user-facing output."""

    def __init__(self, category: str, retryable: bool) -> None:
        super().__init__(category)
        self.category = category
        self.retryable = retryable


class ChatProvider(Protocol):
    """Generate bounded text analysis from structured evidence context."""

    def complete(
        self, system: str, prompt: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion: ...


class LiteLLMChat:
    """Call a user-configured chat model while returning only safe errors."""

    def __init__(
        self,
        model: ModelEntry,
        timeout_seconds: float = 120.0,
        gate: ProviderGate | None = None,
        job_id: str | None = None,
    ) -> None:
        if not model.resolve_key():
            raise ProviderError("missing_provider_configuration", retryable=False)
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.gate = gate
        self.job_id = job_id

    def complete(
        self, system: str, prompt: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion:
        """Return one model response or a bounded provider error category."""

        reservation = self._reserve("chat", system, prompt)
        kwargs: dict[str, Any] = {
            "model": self.model.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "timeout": self.timeout_seconds,
            "max_tokens": 2048,
        }
        if key := self.model.resolve_key():
            kwargs["api_key"] = key
        if self.model.api_base:
            kwargs["api_base"] = self.model.api_base
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            response = litellm.completion(**kwargs)
            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise ProviderError("provider_invalid_response", retryable=False)
            completion = _completion(response, content, self.model)
            self._settle(reservation, completion)
            return completion
        except ProviderError as exc:
            self._settle(reservation, None, failed=True, retryable=exc.retryable)
            raise
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            retryable = status == 429 or (isinstance(status, int) and status >= 500)
            category = (
                "provider_rate_limited_or_unavailable"
                if retryable
                else "provider_request_failed"
            )
            self._settle(reservation, None, failed=True)
            raise ProviderError(category, retryable=retryable) from None

    def _reserve(
        self, stage: str, system: str, prompt: str
    ) -> UsageReservation | None:
        if self.gate is None or self.job_id is None:
            return None
        fingerprint = hashlib.sha256((system + "\0" + prompt).encode("utf-8")).hexdigest()
        return self.gate.reserve(
            self.job_id, stage, f"{stage}:{fingerprint}:{self.model.model}",
            self.gate.estimate_chat(system + "\n" + prompt, output_tokens=2048),
        )

    def _settle(
        self, reservation: UsageReservation | None, response: Any,
        failed: bool = False, retryable: bool = False
    ) -> None:
        if reservation is None or self.gate is None:
            return
        input_tokens = response.input_tokens if isinstance(response, Completion) else None
        output_tokens = response.output_tokens if isinstance(response, Completion) else None
        actual_cost = response.actual_cost_usd if isinstance(response, Completion) else None
        self.gate.settle(
            reservation, actual_cost, input_tokens, output_tokens,
            failed=failed, retryable=retryable
        )


class ASRProvider(Protocol):
    """Transcribe one bounded audio artifact."""

    def transcribe(self, audio_path: Path, start_ms: int) -> list[TranscriptSegment]: ...


class VisionProvider(Protocol):
    """Describe bounded keyframe evidence."""

    def describe(
        self, frame_paths: list[Path], start_ms: int, end_ms: int
    ) -> VisualObservation: ...


class OpenAICompatibleASR:
    """Call an OpenAI-compatible transcription endpoint without leaking secrets."""

    def __init__(
        self,
        model: ModelEntry,
        timeout_seconds: float = 120.0,
        gate: ProviderGate | None = None,
        job_id: str | None = None,
    ) -> None:
        if not model.api_base or not model.resolve_key():
            raise ProviderError("missing_provider_configuration", retryable=False)
        self.model = model
        self.api_base = model.api_base.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.gate = gate
        self.job_id = job_id

    def transcribe(self, audio_path: Path, start_ms: int) -> list[TranscriptSegment]:
        """Upload one audio artifact and normalize timestamped response segments."""

        reservation = self._reserve(audio_path, start_ms)
        try:
            with audio_path.open("rb") as audio:
                response = httpx.post(
                    self.api_base + "/audio/transcriptions",
                    headers={"Authorization": "Bearer " + (self.model.resolve_key() or "")},
                    data={"model": self.model.model, "response_format": "verbose_json"},
                    files={"file": (audio_path.name, audio, "audio/wav")},
                    timeout=self.timeout_seconds,
                )
            if response.status_code == 429 or response.status_code >= 500:
                raise ProviderError("provider_rate_limited_or_unavailable", retryable=True)
            if response.status_code >= 400:
                raise ProviderError("provider_request_rejected", retryable=False)
            result = _parse_transcript(response.json(), start_ms)
            self._settle(reservation, audio_path)
            return result
        except ProviderError as exc:
            self._settle(reservation, audio_path, failed=True, retryable=exc.retryable)
            raise
        except (OSError, httpx.TimeoutException):
            self._settle(reservation, audio_path, failed=True)
            raise ProviderError("provider_timeout", retryable=True) from None
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            self._settle(reservation, audio_path, failed=True)
            raise ProviderError("provider_invalid_response", retryable=False) from None

    def _reserve(self, path: Path, start_ms: int) -> UsageReservation | None:
        if self.gate is None or self.job_id is None:
            return None
        duration_seconds = _wav_duration_seconds(path)
        return self.gate.reserve(
            self.job_id,
            "asr",
            call_key("asr", f"{start_ms}:{_file_digest(path)}", self.model),
            self.gate.estimate_audio(duration_seconds),
        )

    def _settle(
        self, reservation: UsageReservation | None, path: Path,
        failed: bool = False, retryable: bool = False
    ) -> None:
        if reservation and self.gate:
            self.gate.settle(
                reservation, None, audio_seconds=_wav_duration_seconds(path),
                failed=failed, retryable=retryable
            )


class OpenAICompatibleVision:
    """Call a vision-capable chat endpoint with local image references."""

    def __init__(
        self,
        model: ModelEntry,
        timeout_seconds: float = 120.0,
        gate: ProviderGate | None = None,
        job_id: str | None = None,
    ) -> None:
        if not model.api_base or not model.resolve_key():
            raise ProviderError("missing_provider_configuration", retryable=False)
        self.model = model
        self.api_base = model.api_base.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.gate = gate
        self.job_id = job_id

    def describe(
        self, frames: list[tuple[Path, int]], start_ms: int, end_ms: int
    ) -> VisualObservation:
        """Send a bounded image set and normalize a textual visual observation."""

        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Describe important visible events and on-screen text."}
        ]
        for frame, timestamp_ms in frames:
            mime = mimetypes.guess_type(frame.name)[0] or "image/jpeg"
            encoded = base64.b64encode(frame.read_bytes()).decode("ascii")
            content.append({"type": "text", "text": f"Source timestamp: {timestamp_ms} ms"})
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
            )
        payload = {
            "model": self.model.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 2048,
        }
        reservation = self._reserve(frames, start_ms)
        try:
            response = httpx.post(
                self.api_base + "/chat/completions",
                headers={"Authorization": "Bearer " + (self.model.resolve_key() or "")},
                json=payload,
                timeout=self.timeout_seconds,
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise ProviderError("provider_rate_limited_or_unavailable", retryable=True)
            if response.status_code >= 400:
                raise ProviderError("provider_request_rejected", retryable=False)
            data = response.json()
            summary = str(data["choices"][0]["message"]["content"])
            self._settle(reservation, frames, data)
            return VisualObservation(
                start_ms, end_ms, summary,
                frame_times_ms=tuple(timestamp for _, timestamp in frames),
                frame_paths=tuple(str(path) for path, _ in frames),
            )
        except ProviderError as exc:
            self._settle(reservation, frames, failed=True, retryable=exc.retryable)
            raise
        except httpx.TimeoutException:
            self._settle(reservation, frames, failed=True)
            raise ProviderError("provider_timeout", retryable=True) from None
        except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError):
            self._settle(reservation, frames, failed=True)
            raise ProviderError("provider_invalid_response", retryable=False) from None

    def _reserve(
        self, frames: list[tuple[Path, int]], start_ms: int
    ) -> UsageReservation | None:
        if self.gate is None or self.job_id is None:
            return None
        return self.gate.reserve(
            self.job_id,
            "vision",
            call_key(
                "vision",
                f"{start_ms}:" + ":".join(_file_digest(p) for p, _ in frames),
                self.model,
            ),
            self.gate.estimate_vision(
                "Describe important visible events and on-screen text.",
                len(frames),
                output_tokens=2048,
            ),
        )

    def _settle(
        self, reservation: UsageReservation | None,
        frames: list[tuple[Path, int]], response: dict[str, Any] | None = None,
        failed: bool = False, retryable: bool = False,
    ) -> None:
        if reservation and self.gate:
            usage = response.get("usage", {}) if response else {}
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
            actual_cost = None
            if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                if (
                    self.model.input_cost_per_million is not None
                    and self.model.output_cost_per_million is not None
                ):
                    actual_cost = (
                        input_tokens * self.model.input_cost_per_million
                        + output_tokens * self.model.output_cost_per_million
                    ) / 1_000_000
            self.gate.settle(
                reservation, actual_cost, input_tokens, output_tokens,
                failed=failed, retryable=retryable,
            )


def _parse_transcript(payload: dict[str, Any], offset_ms: int) -> list[TranscriptSegment]:
    """Normalize common verbose transcription response shapes."""

    segments = payload.get("segments") or []
    if not isinstance(segments, list):
        raise ValueError("segments is not a list")
    normalized: list[TranscriptSegment] = []
    for item in segments:
        if not isinstance(item, dict):
            raise ValueError("segment is not an object")
        start = offset_ms + int(float(item["start"]) * 1000)
        end = offset_ms + int(float(item["end"]) * 1000)
        words = tuple(
            WordTiming(str(word["word"]), offset_ms + int(float(word["start"]) * 1000),
                       offset_ms + int(float(word["end"]) * 1000))
            for word in (item.get("words") or [])
        )
        normalized.append(
            TranscriptSegment(
                start,
                end,
                str(item.get("text", "")).strip(),
                float(item["confidence"]) if item.get("confidence") is not None else None,
                str(item["speaker"]) if item.get("speaker") is not None else None,
                words,
            )
        )
    return normalized


def _completion(response: Any, content: str, configured: ModelEntry) -> Completion:
    """Normalize provider response usage and derive cost only from complete pricing data."""

    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    model_name = getattr(response, "model", None) or configured.model
    actual_cost = None
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        if (
            configured.input_cost_per_million is not None
            and configured.output_cost_per_million is not None
        ):
            actual_cost = (
                input_tokens * configured.input_cost_per_million
                + output_tokens * configured.output_cost_per_million
            ) / 1_000_000
    message = response.choices[0].message
    raw_calls = getattr(message, "tool_calls", None) or []
    calls = tuple(
        {
            "id": str(call.id),
            "name": str(call.function.name),
            "arguments": call.function.arguments,
        }
        for call in raw_calls
    )
    return Completion(content, model_name, input_tokens, output_tokens, actual_cost, calls)


def _file_digest(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as content:
        while block := content.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _wav_duration_seconds(path: Path) -> float:
    import wave

    with wave.open(str(path), "rb") as audio:
        rate = audio.getframerate()
        return audio.getnframes() / rate if rate else 0.0
