from __future__ import annotations

from pathlib import Path

import pytest

from kvedio_agent.platform.prompting import PromptRenderer


def test_prompt_renderer_renders_structured_template() -> None:
    rendered = PromptRenderer().render(
        "content_location",
        start_ms=1000,
        end_ms=2000,
        evidence="[e1] 1000..1500 ms (transcript): hello",
    )
    assert "1000..2000 ms" in rendered
    assert "e1" in rendered
    assert "evidence_id" in rendered


def test_prompt_renderer_rejects_missing_variables() -> None:
    with pytest.raises(Exception):
        PromptRenderer().render("final_answer", objective="find it")


def test_prompt_template_directory_contains_all_runtime_templates() -> None:
    root = Path(__file__).parents[1] / "src" / "kvedio_agent" / "prompts"
    names = {path.stem for path in root.glob("*.jinja2")}
    assert names == {
        "content_location",
        "final_answer",
        "follow_up",
        "segment_extraction",
        "timeline_reduction",
        "tool_selection",
    }
