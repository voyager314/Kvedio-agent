from datetime import UTC, datetime
from pathlib import Path

from kvedio_agent1.core.domain import VideoRecord
from kvedio_agent1.core.llm import Reply
from kvedio_agent1.core.supervisor import Request, Router


def on_screen() -> VideoRecord:
    now = datetime.now(UTC)
    return VideoRecord(1, Path("talk.mp4"), 1, 1, "fp", None, now, now)


def analysed() -> Request:
    """A request about a video whose transcript and scenes are already there."""
    return Request(
        "谁提到了预算", video=on_screen(), transcript_ready=True, scenes=2, events=1
    )


class ScriptedChat:
    """Plays back one routing reply and counts the calls."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    def complete(self, messages, *, json_mode: bool = False) -> Reply:
        self.calls += 1
        return Reply(
            text=self._text, model="stub", prompt_tokens=0, completion_tokens=0
        )


def test_a_text_file_is_settled_without_a_model():
    chat = ScriptedChat("{}")

    route = Router(chat).route(Request("总结", path=Path("notes.md")))

    assert route.kind == "text_file"
    assert route.agents == ()
    assert chat.calls == 0


def test_a_new_video_is_settled_without_a_model():
    chat = ScriptedChat("{}")

    # The accompanying text is asked afterwards; it does not decide what gets
    # analysed, so a reference always builds the full index.
    for question in ("画面里在做什么", "讲了什么", "总结一下", "帮我整理成笔记"):
        route = Router(chat).route(Request(question, path=Path("talk.mp4")))
        assert route.kind == "analyze_video", question
        assert route.agents == ("asr", "video"), question
    assert chat.calls == 0


def test_a_request_without_a_video_is_settled_without_a_model():
    chat = ScriptedChat("{}")

    route = Router(chat).route(Request("你好"))

    assert route.kind == "chat"
    assert route.agents == ()
    assert chat.calls == 0


def test_the_model_decides_that_a_video_question_is_a_text_task():
    # The wording is full of picture words; the decision is still the model's.
    chat = ScriptedChat('{"route": "chat", "agents": [], "reason": "要一份笔记"}')

    request = Request("画面里在做什么", video=on_screen(), transcript_ready=True, scenes=2)
    route = Router(chat).route(request)

    assert route.kind == "chat"
    assert chat.calls == 1


def test_the_model_decides_that_a_writing_request_is_a_video_question():
    chat = ScriptedChat('{"route": "ask_video", "agents": [], "reason": "问会议内容"}')

    route = Router(chat).route(Request("帮我写一封邮件", video=on_screen(), scenes=2))

    assert route.kind == "ask_video"
    assert route.agents == ()
    assert chat.calls == 1


def test_a_ready_video_needs_no_new_branch_when_the_model_asks_for_none():
    chat = ScriptedChat('{"route": "ask_video", "agents": [], "reason": "已有产物"}')

    assert Router(chat).route(analysed()).agents == ()


def test_a_video_route_without_evidence_asks_for_both_branches():
    chat = ScriptedChat('{"route": "ask_video", "agents": [], "reason": "x"}')

    route = Router(chat).route(Request("那后面呢", video=on_screen()))

    assert route.kind == "ask_video"
    assert route.agents == ("asr", "video")


def test_an_unreadable_route_reply_falls_back_to_the_video():
    chat = ScriptedChat("我不知道")

    route = Router(chat).route(analysed())

    assert route.kind == "ask_video"
    assert route.agents == ()


def test_without_a_model_a_video_is_answered_rather_than_abandoned():
    route = Router().route(analysed())

    assert route.kind == "ask_video"
    assert route.agents == ()


def test_chat_never_carries_agents():
    chat = ScriptedChat('{"route": "chat", "agents": ["asr"], "reason": "x"}')

    route = Router(chat).route(analysed())

    assert route.kind == "chat"
    assert route.agents == ()
