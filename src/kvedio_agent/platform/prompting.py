"""Strict Jinja2 rendering for prompt templates shipped with the application."""

from __future__ import annotations

from importlib import resources
from typing import Any

from jinja2 import Environment, StrictUndefined


class PromptRenderer:
    """Load named prompt templates and reject missing variables before provider calls."""

    def __init__(self) -> None:
        self._environment = Environment(undefined=StrictUndefined, autoescape=False)

    def render(self, name: str, **values: Any) -> str:
        """Render a known template with strict variable checking and normalized whitespace."""

        if not name.replace("_", "").isalnum() or name.startswith("_"):
            raise ValueError("invalid prompt template name")
        resource = resources.files("kvedio_agent.prompts").joinpath(f"{name}.jinja2")
        try:
            template = self._environment.from_string(resource.read_text(encoding="utf-8"))
        except (FileNotFoundError, ModuleNotFoundError):
            raise ValueError(f"unknown prompt template: {name}") from None
        return template.render(**values).strip()
