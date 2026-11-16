from pathlib import Path

import pytest

from kvedio_agent1.interface.submission import Command, Reference, parse_submission

BASE = Path("D:/work")


def test_a_reference_carries_the_question_verbatim():
    parsed = parse_submission("@videos/talk.mp4 讲了什么？", BASE)
    assert isinstance(parsed, Reference)
    assert parsed.path == BASE / "videos/talk.mp4"
    assert parsed.question == "讲了什么？"


def test_a_quoted_path_may_contain_spaces():
    parsed = parse_submission('@"my talks/week 1.mp4" 总结一下', BASE)
    assert isinstance(parsed, Reference)
    assert parsed.path.name == "week 1.mp4"
    assert parsed.question == "总结一下"


def test_an_absolute_path_is_taken_as_given(tmp_path):
    absolute = tmp_path / "a.mp4"
    parsed = parse_submission(f"@{absolute} 讲了什么", BASE)
    assert isinstance(parsed, Reference)
    assert parsed.path == absolute


def test_a_reference_without_a_question_is_refused():
    assert parse_submission("@videos/talk.mp4", BASE) is None


def test_a_plain_sentence_is_not_a_reference():
    assert parse_submission("帮我写一封邮件", BASE) is None


def test_a_slash_line_is_a_command_with_arguments():
    parsed = parse_submission("/model asr dashscope", BASE)
    assert isinstance(parsed, Command)
    assert parsed.name == "model"
    assert parsed.arguments == ("asr", "dashscope")


def test_a_command_name_is_case_folded():
    assert isinstance(parse_submission("/HELP", BASE), Command)
    assert parse_submission("/HELP", BASE).name == "help"


def test_an_unclosed_quote_is_refused():
    assert parse_submission('@"week 1.mp4 讲了什么', BASE) is None


@pytest.mark.parametrize("text", ["", "   ", "@", "@   "])
def test_empty_input_produces_nothing(text):
    assert parse_submission(text, BASE) is None
