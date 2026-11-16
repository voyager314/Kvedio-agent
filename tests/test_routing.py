from datetime import UTC, datetime
from pathlib import Path

from kvedio_agent1.core.domain import VideoRecord
from kvedio_agent1.core.llm import Reply
from kvedio_agent1.core.supervisor import Request, Router


def on_screen() -> VideoRecord:
    now = datetime.now(UTC)
    return VideoRecord(1, Path("talk.mp4"), 1, 1, "fp", None, now, now)


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


def test_a_text_file_is_routed_without_a_model():
    chat = ScriptedChat("{}")

    route = Router(chat).route(Request("总结", path=Path("notes.md")))

    assert route.kind == "text_file"
    assert route.agents == ()
    assert chat.calls == 0


def test_a_new_video_is_analysed_on_both_branches_whatever_the_wording():
    router = Router()

    # The accompanying text is asked afterwards; it does not decide what gets
    # analysed, so a reference always builds the full index.
    for question in ("画面里在做什么", "讲了什么", "总结一下", "帮我整理成笔记"):
        route = router.route(Request(question, path=Path("talk.mp4")))
        assert route.kind == "analyze_video", question
        assert route.agents == ("asr", "video"), question


def test_a_question_against_ready_products_asks_for_no_new_branch():
    route = Router().route(
        Request(
            "谁提到了预算",
            video=on_screen(),
            transcript_ready=True,
            scenes=2,
            events=1,
        )
    )

    assert route.kind == "ask_video"
    assert route.agents == ()


def test_a_text_task_on_a_video_is_handled_by_the_supervisor():
    route = Router().route(
        Request(
            "帮我写一封邮件",
            video=on_screen(),
            transcript_ready=True,
            scenes=2,
        )
    )

    assert route.kind == "chat"
    assert route.agents == ()


def test_a_text_artefact_over_a_ready_video_is_handled_by_the_supervisor():
    route = Router().route(
        Request(
            "把视频里提到的 Redis 整理成笔记",
            video=on_screen(),
            transcript_ready=True,
            scenes=2,
        )
    )

    assert route.kind == "chat"
    assert route.agents == ()


def test_a_request_without_a_video_is_a_chat():
    route = Router().route(Request("你好"))

    assert route.kind == "chat"
    assert route.agents == ()


def test_an_ambiguous_request_goes_to_the_model():
    chat = ScriptedChat('{"route": "chat", "agents": [], "reason": "普通问题"}')

    route = Router(chat).route(
        Request("那后面呢", video=on_screen(), transcript_ready=True, scenes=2)
    )

    assert route.kind == "chat"
    assert route.agents == ()
    assert chat.calls == 1


def test_an_unreadable_route_reply_falls_back_to_the_video():
    chat = ScriptedChat("我不知道")

    route = Router(chat).route(
        Request("那后面呢", video=on_screen(), transcript_ready=True, scenes=2)
    )

    assert route.kind == "ask_video"
    assert route.agents == ()


def test_a_video_route_without_evidence_asks_for_both_branches():
    chat = ScriptedChat('{"route": "ask_video", "agents": [], "reason": "x"}')

    route = Router(chat).route(Request("那后面呢", video=on_screen()))

    assert route.kind == "ask_video"
    assert route.agents == ("asr", "video")


def test_chat_never_carries_agents():
    chat = ScriptedChat('{"route": "chat", "agents": ["asr"], "reason": "x"}')

    route = Router(chat).route(
        Request("那后面呢", video=on_screen(), transcript_ready=True, scenes=2)
    )

    assert route.kind == "chat"
    assert route.agents == ()
