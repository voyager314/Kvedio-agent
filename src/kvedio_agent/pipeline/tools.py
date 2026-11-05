"""Bounded, job-scoped tools for evidence lookup and operational inspection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kvedio_agent.platform.storage import Store


class ToolCallError(ValueError):
    """Safe validation error for a malformed or unauthorized tool call."""


@dataclass(frozen=True)
class ToolSpec:
    """Description and bounded handler for one model-visible tool."""

    name: str  # Allowlisted provider-visible function name.
    description: str  # User-data-only capability description.
    parameters: dict[str, Any]  # JSON Schema object; additional properties are forbidden.
    handler: Callable[[dict[str, Any]], dict[str, Any]]  # Read-only handler scoped to one job.


class JobToolRegistry:
    """Expose read-only evidence and job state for exactly one persisted job."""

    def __init__(self, store: Store, job_id: str) -> None:
        self.store = store
        self.job_id = job_id
        self._specs = self._build_specs()

    def schemas(self) -> list[dict[str, Any]]:
        """Return provider-compatible function schemas without executable details."""

        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in self._specs.values()
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate and execute one allowlisted tool call for the bound job."""

        spec = self._specs.get(name)
        if spec is None:
            raise ToolCallError("unknown tool")
        if not isinstance(arguments, dict):
            raise ToolCallError("tool arguments must be an object")
        allowed = set(spec.parameters.get("properties", {}))
        if set(arguments) - allowed:
            raise ToolCallError("unsupported tool arguments")
        try:
            return spec.handler(arguments)
        except (KeyError, TypeError, ValueError):
            raise ToolCallError("invalid tool arguments") from None

    def _build_specs(self) -> dict[str, ToolSpec]:
        return {
            "search_evidence": ToolSpec(
                "search_evidence",
                "Find timestamped transcript or visual evidence by terms.",
                {
                    "type": "object", "required": ["query"],
                    "properties": {
                        "query": {"type": "string", "minLength": 1},
                        "modality": {"type": "string", "enum": ["transcript", "visual"]},
                    },
                    "additionalProperties": False,
                },
                self._search_evidence,
            ),
            "get_transcript": ToolSpec(
                "get_transcript", "Get transcript evidence overlapping a time range.",
                _range_schema(), self._get_transcript,
            ),
            "get_visual_evidence": ToolSpec(
                "get_visual_evidence", "Get visual evidence overlapping a time range.",
                _range_schema(), self._get_visual,
            ),
            "get_job_progress": ToolSpec(
                "get_job_progress", "Get durable status and stage progress for this job.",
                {"type": "object", "properties": {}, "additionalProperties": False},
                self._get_progress,
            ),
            "get_usage_summary": ToolSpec(
                "get_usage_summary", "Get actual, estimated, reserved, and unknown usage.",
                {"type": "object", "properties": {}, "additionalProperties": False},
                lambda _: self.store.usage_summary(self.job_id),
            ),
        }

    def _search_evidence(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments["query"]
        if not isinstance(query, str) or not query.strip():
            raise ToolCallError("query must be non-empty")
        modality = arguments.get("modality")
        rows = self.store.search_evidence(
            self.job_id, query.split(), 12,
            modality=modality if isinstance(modality, str) else None,
        )
        return {"evidence": _evidence_rows(rows)}

    def _get_transcript(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"evidence": _evidence_rows(self._range(arguments, "transcript"))}

    def _get_visual(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"evidence": _evidence_rows(self._range(arguments, "visual"))}

    def _get_progress(self, _: dict[str, Any]) -> dict[str, Any]:
        job = next((item for item in self.store.job_history() if item["id"] == self.job_id), None)
        if job is None:
            raise ToolCallError("job does not exist")
        stages = [item for item in self.store.job_stages(self.job_id)]
        return {"status": job["status"], "current_stage": job["current_stage"], "stages": stages}

    def _range(self, arguments: dict[str, Any], modality: str) -> list[dict[str, Any]]:
        start_ms = arguments["start_ms"]
        end_ms = arguments["end_ms"]
        if not isinstance(start_ms, int) or not isinstance(end_ms, int):
            raise ToolCallError("time bounds must be integers in milliseconds")
        if start_ms < 0 or end_ms <= start_ms:
            raise ToolCallError("time range must be positive and ordered")
        return [
            row for row in self.store.evidence_for_range(self.job_id, start_ms, end_ms)
            if row["modality"] == modality
        ]


def _range_schema() -> dict[str, Any]:
    return {
        "type": "object", "required": ["start_ms", "end_ms"],
        "properties": {
            "start_ms": {"type": "integer", "minimum": 0},
            "end_ms": {"type": "integer", "exclusiveMinimum": 0},
        },
        "additionalProperties": False,
    }


def _evidence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": row["id"], "start_ms": row["start_ms"],
            "end_ms": row["end_ms"], "modality": row["modality"],
            "content": row.get("content") or "",
        }
        for row in rows
    ]
