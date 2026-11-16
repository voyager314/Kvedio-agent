"""Prompt templates, kept as text so wording changes without touching logic.

Placeholders use ``$name`` rather than braces, because every prompt here contains
JSON and braces would have to be escaped twice.
"""

from __future__ import annotations

from importlib import resources
from string import Template


def render(name: str, **values: object) -> str:
    """Fill a template shipped inside the package.

    Raises ``KeyError`` when a placeholder has no value: a prompt that silently
    keeps ``$evidence`` in front of the model is worse than a failed call.
    """
    text = (
        resources.files("kvedio_agent1")
        .joinpath(f"prompts/{name}.md")
        .read_text(encoding="utf-8")
    )
    return Template(text).substitute(values)
