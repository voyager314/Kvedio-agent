import json
from dataclasses import replace
from pathlib import Path

import pytest

from kvedio_agent1.core.domain import (
    Chunk,
    DependencyMode,
    MediaInfo,
    NewTask,
    Scene,
    Segment,
    TaskStatus,
)
from kvedio_agent1.core.llm import Message, Reply
from kvedio_agent1.core.storage import Storage
from kvedio_agent1.core.supervisor import Answering, UnsupportedCitation
from kvedio_agent1.core.supervisor import citations as rules
from kvedio_agent1.core.supervisor.planner import MAX_STEPS, make_plan
from kvedio_agent1.core.supervisor.synthesis import write
from kvedio_agent1.core.tasks import EventBus
from kvedio_agent1.tools import Evidence
from kvedio_agent1.tools.transcript import TOOLS

SEGMENTS = [
    Segment(start_ms=0, end_ms=5_000, text="今天先过一遍项目进度"),
    Segment(start_ms=5_000, end_ms=12_000, text="预算方面还有二十万的缺口"),
    Segment(start_ms=12_000, end_ms=20_000, text="人力上需要再补两个后端"),
]


class ScriptedChat:
    """Plays back replies in order and keeps every message it was given."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[list[Message]] = []

    def complete(self, messages, *, json_mode: bool = False) -> Reply:
        self.calls.append(list(messages))
        return Reply(
            text=self._replies.pop(0), model="stub", prompt_tokens=0, completion_tokens=0
        )

    def stream(self, messages, on_delta):
        reply = self.complete(messages)
        for piece in reply.text:
            on_delta(piece)
        return reply


def plan(*steps: tuple[str, dict]) -> str:
    return json.dumps(
        {"steps": [{"tool": tool, "arguments": args, "why": "查一下"} for tool, args in steps]}
    )


def evidence(start: int, end: int, text: str) -> Evidence:
    return Evidence(start_ms=start, end_ms=end, text=text)


# --- citation rules -------------------------------------------------------


def test_moments_and_ranges_are_read_from_the_text():
    spans = rules.find("开场 [00:10]，讨论 [01:05-02:00]")
    assert [(span.start_ms, span.end_ms) for span in spans] == [
        (10_000, 11_000),
        (65_000, 120_000),
    ]


def test_a_moment_inside_the_evidence_is_accepted():
    verdict = rules.check("预算有缺口 [01:30]", [evidence(60_000, 120_000, "预算有缺口")])
    assert verdict.ok
    assert verdict.citations[0].quote == "预算有缺口"
    assert (verdict.citations[0].start_ms, verdict.citations[0].end_ms) == (90_000, 91_000)


def test_a_moment_outside_the_evidence_is_refused():
    verdict = rules.check("预算有缺口 [03:00]", [evidence(0, 60_000, "预算")])
    assert verdict.citations == ()
    assert verdict.missing == ("[03:00]",)
    assert verdict.ok is False


def test_a_range_may_not_end_outside_the_evidence():
    verdict = rules.check("这段讨论 [00:10-00:50]", [evidence(0, 30_000, "讨论")])
    assert verdict.citations == ()
    assert verdict.missing == ("[00:10]",)
    assert verdict.ok is False


def test_an_answer_without_any_moment_is_refused():
    verdict = rules.check("大家都同意", [evidence(0, 1_000, "同意")])
    assert verdict.cited == 0
    assert verdict.missing == ()
    assert verdict.ok is False


# --- planning -------------------------------------------------------------


def test_a_usable_plan_is_kept_in_order():
    chat = ScriptedChat([plan(("overview", {}), ("search", {"query": "预算"}))])
    result = make_plan(chat, "预算够吗", TOOLS)
    assert [step.tool for step in result.steps] == ["overview", "search"]
    assert "2 步" in result.note


def test_a_plan_in_a_code_fence_is_still_read():
    chat = ScriptedChat([f"```json\n{plan(('overview', {}))}\n```"])
    result = make_plan(chat, "讲了什么", TOOLS)
    assert [step.tool for step in result.steps] == ["overview"]


def test_a_plan_with_unknown_tools_falls_back():
    chat = ScriptedChat([plan(("download", {}))])
    result = make_plan(chat, "讲了什么", TOOLS)
    assert [step.tool for step in result.steps] == ["overview", "search"]
    assert "固定检索" in result.note


def test_prose_instead_of_json_falls_back():
    chat = ScriptedChat(["我先看看 transcript。"])
    result = make_plan(chat, "讲了什么", TOOLS)
    assert result.steps
    assert "固定检索" in result.note


def test_a_plan_is_truncated_to_the_step_ceiling():
    steps = [("overview", {})] * (MAX_STEPS + 3)
    chat = ScriptedChat([plan(*steps)])
    result = make_plan(chat, "问题", TOOLS)
    assert len(result.steps) == MAX_STEPS


def test_a_fallback_plan_uses_only_the_tools_the_video_can_supply():
    chat = ScriptedChat(["这不是 JSON"])
    offered = [tool for tool in TOOLS if tool.name == "scenes"]

    result = make_plan(chat, "画面里有什么", offered)

    assert [step.tool for step in result.steps] == ["scenes"]
    assert "固定检索" in result.note


# --- synthesis ------------------------------------------------------------


def test_an_answer_that_cites_the_evidence_is_returned_as_written():
    chat = ScriptedChat(["预算还差二十万 [00:10]。"])
    conclusion = write(
        chat,
        question="预算够吗",
        evidence=[evidence(5_000, 12_000, "预算还差二十万")],
        duration_ms=20_000,
    )
    assert conclusion.text == "预算还差二十万 [00:10]。"
    assert len(conclusion.citations) == 1
    assert len(chat.calls) == 1


def test_one_repair_is_allowed_when_a_moment_was_invented():
    chat = ScriptedChat(["结论 [09:00]。", "结论 [00:10]。"])
    conclusion = write(
        chat,
        question="问题",
        evidence=[evidence(5_000, 12_000, "证据原文")],
        duration_ms=20_000,
    )
    assert conclusion.text == "结论 [00:10]。"
    repair = chat.calls[1][-1]
    assert repair.role == "user"
    assert "09:00" in repair.content


def test_a_second_unsupported_answer_is_refused():
    chat = ScriptedChat(["结论 [09:00]。", "还是 [09:00]。"])
    with pytest.raises(UnsupportedCitation, match="09:00"):
        write(
            chat,
            question="问题",
            evidence=[evidence(5_000, 12_000, "证据原文")],
            duration_ms=20_000,
        )


# --- the whole loop -------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    opened = Storage(tmp_path / "kvedio.db")
    yield opened
    opened.close()


def prepare(store: Storage, question: str = "预算够吗"):
    video = store.register_video(
        Path("talk.mp4"), size_bytes=1, mtime_ns=1, fingerprint="fp-supervisor"
    )
    store.record_media(
        video.id,
        MediaInfo(
            duration_ms=20_000,
            container="mp4",
            video_codec="h264",
            width=320,
            height=180,
            fps=25.0,
            audio_codec="aac",
            sample_rate=48_000,
            channels=2,
        ),
    )
    transcript = store.open_transcript(video.id, "key", "asr", "transcribe")
    store.save_chunk(
        transcript.id,
        video.id,
        "transcribe",
        "key",
        Chunk(index=0, start_ms=0, end_ms=20_000),
        SEGMENTS,
    )
    store.finish_transcript(transcript.id)
    job = store.create_job(
        video.id, question, [NewTask(id="answer", agent="supervisor", action="answer")]
    )
    return video, job


def test_the_loop_plans_searches_answers_and_records_citations(store):
    video, job = prepare(store)
    chat = ScriptedChat(
        [
            plan(("overview", {}), ("search", {"query": "预算"})),
            "预算还差二十万 [00:05]。",
        ]
    )
    bus = EventBus()
    channel = bus.subscribe()
    reports: list[float] = []

    answer = Answering(store, bus, chat, "chat-model").answer(
        job, lambda fraction, message: reports.append(fraction)
    )

    assert answer.text == "预算还差二十万 [00:05]。"
    assert answer.model == "chat-model"
    assert answer.citations[0].quote == "预算方面还有二十万的缺口"
    assert len(chat.calls) == 2
    assert reports[-1] >= 0.9

    events = _drain(channel)
    notes = [event.payload["message"] for event in events if "message" in event.payload]
    deltas = [event.payload["delta"] for event in events if "delta" in event.payload]
    assert any("按计划执行 2 步" in note for note in notes)
    assert any("search(预算)" in note for note in notes)
    assert "".join(deltas) == answer.text


def test_the_loop_says_so_when_nothing_matched(store):
    video, job = prepare(store, question="天气怎么样")
    chat = ScriptedChat([plan(("search", {"query": "天气"})), "这段录像没有提到天气。"])
    bus = EventBus()

    answer = Answering(store, bus, chat, "chat-model").answer(job, lambda fraction, message: None)

    assert answer.text == "这段录像没有提到天气。"
    assert answer.citations == ()
    assert len(chat.calls) == 2
    assert "language of the question" in chat.calls[1][0].content.lower()


def test_the_loop_refuses_to_answer_without_any_evidence(store):
    video, job = prepare(store)
    second = store.register_video(
        Path("other.mp4"), size_bytes=2, mtime_ns=2, fingerprint="fp-none"
    )
    empty_job = store.create_job(
        second.id, "问题", [NewTask(id="answer", agent="supervisor", action="answer")]
    )
    chat = ScriptedChat([])

    with pytest.raises(ValueError, match="转录或画面描述"):
        Answering(store, EventBus(), chat, "chat-model").answer(
            empty_job, lambda fraction, message: None
        )


def test_a_picture_only_video_is_answered_and_cited_as_a_scene(store):
    video = store.register_video(
        Path("silent.mp4"), size_bytes=3, mtime_ns=3, fingerprint="fp-visual"
    )
    store.record_media(
        video.id,
        MediaInfo(
            duration_ms=5_000,
            container="mp4",
            video_codec="h264",
            width=320,
            height=180,
            fps=25.0,
            audio_codec=None,
            sample_rate=None,
            channels=None,
        ),
    )
    store.save_scene(video.id, "sk", 0, Scene(0, 4_000, "白板上写着预算", "/tmp/frame.jpg"))
    job = store.create_job(
        video.id, "画面里有什么", [NewTask(id="answer", agent="supervisor", action="answer")]
    )
    chat = ScriptedChat([plan(("scenes", {})), "白板上写着预算 [00:01]。"])

    answer = Answering(store, EventBus(), chat, "chat-model").answer(
        job, lambda fraction, message: None
    )

    assert answer.citations[0].kind == "scene"
    assert answer.citations[0].quote == "白板上写着预算"
    assert "证据范围" not in answer.text
    assert "该视频没有音轨" in chat.calls[1][0].content


def test_a_failed_vision_run_is_named_in_the_conclusion(store):
    video = store.register_video(
        Path("lecture.mp4"), size_bytes=4, mtime_ns=4, fingerprint="fp-gap"
    )
    store.record_media(
        video.id,
        MediaInfo(
            duration_ms=20_000,
            container="mp4",
            video_codec="h264",
            width=320,
            height=180,
            fps=25.0,
            audio_codec="aac",
            sample_rate=48_000,
            channels=2,
        ),
    )
    transcript = store.open_transcript(video.id, "key", "asr", "transcribe")
    store.save_chunk(
        transcript.id,
        video.id,
        "transcribe",
        "key",
        Chunk(index=0, start_ms=0, end_ms=20_000),
        SEGMENTS,
    )
    store.finish_transcript(transcript.id)
    job = store.create_job(
        video.id,
        "预算够吗",
        [
            NewTask(id="vision", agent="vision", action="describe"),
            NewTask(
                id="answer",
                agent="supervisor",
                action="answer",
                depends_on=("vision",),
                dependency_mode=DependencyMode.PARTIAL,
            ),
        ],
    )
    tasks = {task.id: task for task in store.tasks_for_job(job.id)}
    store.save_task(
        replace(tasks["vision"], status=TaskStatus.FAILED, error="视觉模型不可用")
    )
    chat = ScriptedChat([plan(("search", {"query": "预算"})), "预算还差二十万 [00:05]。"])

    answer = Answering(store, EventBus(), chat, "chat-model").answer(
        job, lambda fraction, message: None
    )

    assert "预算还差二十万" in answer.text
    assert answer.text.endswith(
        "证据范围：本次作答有证据缺失——画面描述不可用（视觉模型不可用）。"
    )
    assert "Evidence not available" in chat.calls[1][0].content


def _drain(channel) -> list:
    import queue

    collected = []
    while True:
        try:
            collected.append(channel.get_nowait())
        except queue.Empty:
            return collected

def test_a_map_line_cannot_be_cited():
    verdict = rules.check("[00:00]", [evidence(0, 10_000, "整段在谈预算").__class__(
        start_ms=0, end_ms=10_000, text="整段在谈预算", label="map"
    )])
    assert verdict.missing == ("[00:00]",)
    assert verdict.ok is False


def test_a_scene_line_can_be_cited_when_nobody_was_talking():
    verdict = rules.check("[00:00]", [evidence(0, 10_000, "白板上写着预算").__class__(
        start_ms=0, end_ms=10_000, text="白板上写着预算", label="scene"
    )])
    assert verdict.ok is True
    assert verdict.citations[0].kind == "scene"
    assert verdict.citations[0].quote == "白板上写着预算"


def test_speech_wins_over_the_picture_at_the_same_moment():
    scene = evidence(0, 10_000, "白板上写着预算").__class__(
        start_ms=0, end_ms=10_000, text="白板上写着预算", label="scene"
    )
    spoken = evidence(0, 10_000, "预算还差二十万")
    verdict = rules.check("[00:02]", [scene, spoken])
    assert verdict.ok is True
    assert verdict.citations[0].kind == "transcript"
    assert verdict.citations[0].quote == "预算还差二十万"

def test_a_scene_and_speech_at_the_same_moment_are_both_kept(store):
    video, job = prepare(store)
    store.save_scene(video.id, "sk", 0, Scene(0, 5_000, "白板上写着预算", "/tmp/frame.jpg"))
    chat = ScriptedChat(
        [
            plan(("scenes", {}), ("overview", {})),
            "预算写在白板上 [00:00]。",
        ]
    )

    answer = Answering(store, EventBus(), chat, "chat-model").answer(
        job, lambda fraction, message: None
    )

    # The citation is anchored on the speech at that moment, not on the picture line.
    assert answer.citations[0].quote == "今天先过一遍项目进度"
