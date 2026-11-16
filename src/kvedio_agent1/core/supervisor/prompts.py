"""Prompt templates, rendered with Jinja2 from files shipped in the package.

The templates are English on purpose: models follow English instructions most
reliably. Only the supervisor speaks to the user, so the answer templates carry a
language rule that binds the reply to the question's language — this module never
decides the output language, and no prompt text lives in Python.
"""

from __future__ import annotations

from importlib import resources

from jinja2 import Environment, FunctionLoader, StrictUndefined

from ..domain import format_moment


def _moment(value: object) -> str:
    """The one timestamp format: the templates write it, the citation parser reads it."""
    return format_moment(int(value))  # type: ignore[arg-type]


def _load(name: str) -> str | None:
    """Read one template from the package; ``None`` makes Jinja report a missing template."""
    path = resources.files("kvedio_agent1").joinpath("prompts", name)
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


_ENVIRONMENT = Environment(
    loader=FunctionLoader(_load),
    undefined=StrictUndefined,
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)
_ENVIRONMENT.filters["moment"] = _moment


def render(name: str, /, **values: object) -> str:
    """Render one template with the given values.

    Raises ``jinja2.UndefinedError`` when a template refers to a value that was not
    passed: a prompt that silently keeps ``{{ evidence }}`` in front of the model is
    worse than a failed call.
    """
    template = _ENVIRONMENT.get_template(f"{name}.j2")
    return template.render(**values).strip()
