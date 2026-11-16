from importlib import resources

import pytest
from jinja2 import UndefinedError

from kvedio_agent1.core import prompts
from kvedio_agent1.core.domain import Scene, format_moment
from kvedio_agent1.core.supervisor import citations
from kvedio_agent1.tools import Evidence
from kvedio_agent1.tools.transcript import TOOLS

EVIDENCE = (
    Evidence(start_ms=0, end_ms=5_000, text="今天先过一遍项目进度"),
    Evidence(start_ms=5_000, end_ms=12_000, text="预算方面还有二十万的缺口"),
)


def test_the_plan_lists_every_tool_with_its_arguments():
    text = prompts.render("plan", tools=TOOLS, question="预算够吗", max_steps=3)

    for tool in TOOLS:
        assert f"### {tool.name}" in text
    assert "`query`" in text
    assert "`start`" in text
    assert "at most 3 steps" in text
    assert "预算够吗" in text


def test_the_answer_carries_question_duration_and_evidence():
    text = prompts.render(
        "answer",
        question="预算够吗",
        evidence=EVIDENCE,
        duration_ms=20_000,
        limits=(),
        previous=None,
    )

    assert "预算够吗" in text
    assert "[00:05] 预算方面还有二十万的缺口" in text
    assert "00:20" in text


def test_the_answer_binds_the_reply_to_the_question_language():
    text = prompts.render(
        "answer", question="q", evidence=EVIDENCE, duration_ms=0, limits=(), previous=None
    )
    assert "language of the question" in text.lower()


def test_the_answer_names_unavailable_evidence():
    text = prompts.render(
        "answer",
        question="q",
        evidence=EVIDENCE,
        duration_ms=0,
        limits=("画面描述不可用（视觉模型失败）",),
        previous=None,
    )
    assert "Evidence not available" in text
    assert "画面描述不可用（视觉模型失败）" in text


def test_the_answer_receives_the_earlier_conclusion_when_updating():
    text = prompts.render(
        "answer",
        question="q",
        evidence=EVIDENCE,
        duration_ms=0,
        limits=(),
        previous="先前的结论 [00:01]。",
    )
    assert "Earlier conclusion" in text
    assert "先前的结论 [00:01]。" in text


def test_the_chat_prompt_carries_the_file_and_the_evidence():
    text = prompts.render(
        "chat",
        question="q",
        text_file="notes.md",
        text="文件正文",
        video_name="talk.mp4",
        evidence=(Evidence(0, 5_000, "预算不足"),),
        duration_ms=5_000,
    )

    assert "notes.md" in text
    assert "文件正文" in text
    assert "[00:00] 预算不足" in text
    assert "No file or video was supplied" not in text


def test_the_chat_prompt_says_when_nothing_was_supplied():
    text = prompts.render(
        "chat",
        question="q",
        text_file=None,
        text="",
        video_name=None,
        evidence=(),
        duration_ms=0,
    )

    assert "No file or video was supplied" in text


def test_the_route_prompt_names_both_agents_and_the_supervisor_tools():
    text = prompts.render(
        "route",
        question="q",
        video_name="talk.mp4",
        transcript_state="absent",
        scenes=0,
        events=0,
        text_file=None,
        no_video=False,
    )

    assert "asr_agent" in text
    assert "video_agent" in text
    assert "read_text_file" in text
    assert "Integrating the two agents" in text


def test_the_absence_prompt_binds_the_language_too():
    text = prompts.render("no_evidence", question="what did they decide", limits=())
    assert "language of the question" in text.lower()


def test_the_repair_names_invented_moments_and_shows_allowed_ones():
    text = prompts.render(
        "repair", missing=("[09:00]",), cited=2, examples=[EVIDENCE[0]]
    )
    assert "not in the evidence: [09:00]" in text
    assert "[00:00] 今天先过一遍项目进度" in text
    assert "预算方面" not in text  # only the first example is shown


def test_the_repair_says_so_when_nothing_was_cited():
    text = prompts.render("repair", missing=(), cited=0, examples=[EVIDENCE[0]])
    assert "cited no moment at all" in text


def test_one_timestamp_format_is_shared_by_prompt_and_parser():
    text = prompts.render(
        "answer", question="q", evidence=EVIDENCE, duration_ms=0, limits=(), previous=None
    )
    assert citations.find(text), "the prompt must show a citation the parser accepts"

    shown = text.split("for example ")[1].split(".")[0]
    moments = citations.find(shown)
    assert moments and moments[0].start_ms == 12 * 60_000 + 34_000


def test_the_moment_filter_matches_the_parser_format():
    text = prompts.render(
        "answer", question="q", evidence=EVIDENCE, duration_ms=3_725_000, limits=(), previous=None
    )
    assert format_moment(3_725_000) in text


def test_the_coverage_prompt_describes_one_stretch_only():
    text = prompts.render("coverage", start_ms=0, end_ms=180_000, segments=EVIDENCE)
    assert "[00:05] 预算方面还有二十万的缺口" in text
    assert "03:00" in text
    assert "language of the transcript" in text.lower()


def test_a_missing_value_fails_loudly():
    with pytest.raises(UndefinedError):
        prompts.render("plan", tools=TOOLS)


def test_templates_carry_no_chinese():
    """The prompts are English by decision; only a rendered value may be Chinese."""
    root = resources.files("kvedio_agent1").joinpath("prompts")
    offenders: list[str] = []
    for item in root.iterdir():
        if not item.name.endswith(".j2"):
            continue
        for number, line in enumerate(item.read_text(encoding="utf-8").splitlines(), start=1):
            if any("\u4e00" <= character <= "\u9fff" for character in line):
                offenders.append(f"{item.name}:{number}")
    assert offenders == []

def test_the_scene_prompt_carries_the_speech_and_the_stretch():
    text = prompts.render("scene", start_ms=0, end_ms=5_000, segments=EVIDENCE[:1])
    assert "[00:00] 今天先过一遍项目进度" in text
    assert "00:05" in text


def test_the_scene_prompt_says_when_nothing_was_said():
    text = prompts.render("scene", start_ms=0, end_ms=5_000, segments=())
    assert "No speech was transcribed" in text

def test_the_events_prompt_lists_every_shot_and_the_ceiling():
    scenes = [
        Scene(0, 5_000, "白板上写着预算", "/tmp/0.jpg"),
        Scene(5_000, 9_000, "镜头转向财务表", "/tmp/1.jpg"),
    ]
    text = prompts.render("events", duration_ms=9_000, scenes=scenes, max_events=7)
    assert "[00:00-00:05] 白板上写着预算" in text
    assert "[00:05-00:09] 镜头转向财务表" in text
    assert "at most 7" in text
