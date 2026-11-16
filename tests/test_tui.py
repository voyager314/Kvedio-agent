import asyncio
from pathlib import Path

from textual.widgets import DataTable, Input, Static

from kvedio_agent1.core.asr import AsrStage
from kvedio_agent1.core.config import load_or_create
from kvedio_agent1.core.domain import JobStatus, Segment
from kvedio_agent1.core.pipeline import Pipeline, Stages
from kvedio_agent1.core.storage import Storage
from kvedio_agent1.core.tasks import EventBus
from kvedio_agent1.core.workspace import Workspace
from kvedio_agent1.interface.tui import Workbench


def spoken(audio: Path) -> list[Segment]:
    return [Segment(start_ms=0, end_ms=600, text="开场白")]


def build(store, workspace, bus, loaded, answerer) -> Pipeline:
    asr = AsrStage(
        store,
        workspace,
        spoken,
        model="asr-model",
        address="https://asr.example.invalid/v1/audio/transcriptions",
        chunk_ms=300_000,
        concurrency=1,
    )
    stages = Stages(
        runtime=loaded.config.runtime,
        workspace=workspace,
        asr=asr,
        answering=answerer,
    )
    return Pipeline(store, bus, stages)


def test_a_reference_travels_from_the_input_line_to_the_evidence_panel(
    home, tmp_path, sample_video, make_answerer
):
    loaded = load_or_create()
    store = Storage(tmp_path / "kvedio.db")
    bus = EventBus()
    pipeline = build(
        store, Workspace(tmp_path / "workspace"), bus, loaded, make_answerer(store)
    )

    async def scenario() -> None:
        app = Workbench(loaded, store, pipeline, bus)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#prompt", Input).value = f"@{sample_video} 讲了什么"
            await pilot.press("enter")
            for _ in range(160):
                await pilot.pause(0.05)
                jobs = store.jobs()
                if jobs and jobs[0].status is not JobStatus.RUNNING:
                    break
            jobs = store.jobs()
            assert jobs, "作业没有落库"
            assert jobs[0].status is JobStatus.SUCCEEDED, jobs[0].error
            for _ in range(40):
                evidence = str(app.query_one("#evidence", Static).render())
                if "引用 1 处" in evidence:
                    break
                await pilot.pause(0.05)
            assert app.query_one("#tasks", DataTable).row_count == 4
            assert "转录" in str(app.query_one("#status", Static).render())
            assert "引用 1 处" in str(app.query_one("#evidence", Static).render())
            assert "开场白" in str(app.query_one("#evidence", Static).render())
            timeline = str(app.query_one("#timeline", Static).render())
            assert "转录：1 段" in timeline
            assert store.answer(jobs[0].id) is not None

    asyncio.run(scenario())
    store.close()


def test_a_plain_line_asks_the_video_on_screen(home, tmp_path, sample_video, make_answerer):
    loaded = load_or_create()
    store = Storage(tmp_path / "kvedio.db")
    bus = EventBus()
    pipeline = build(
        store, Workspace(tmp_path / "workspace"), bus, loaded, make_answerer(store)
    )

    async def scenario() -> None:
        app = Workbench(loaded, store, pipeline, bus)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#prompt", Input).value = f"@{sample_video} 讲了什么"
            await pilot.press("enter")
            for _ in range(160):
                await pilot.pause(0.05)
                if store.jobs() and store.jobs()[0].status is not JobStatus.RUNNING:
                    break
            for _ in range(40):
                if "引用 1 处" in str(app.query_one("#evidence", Static).render()):
                    break
                await pilot.pause(0.05)

            app.query_one("#prompt", Input).value = "再具体说说"
            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
                jobs = store.jobs()
                if len(jobs) == 2 and jobs[0].status is not JobStatus.RUNNING:
                    break
            jobs = store.jobs()
            assert len(jobs) == 2
            assert jobs[0].status is JobStatus.SUCCEEDED, jobs[0].error
            assert [task.id for task in store.tasks_for_job(jobs[0].id)] == ["answer"]
            for _ in range(40):
                if app.query_one("#tasks", DataTable).row_count == 1:
                    break
                await pilot.pause(0.05)
            assert app.query_one("#tasks", DataTable).row_count == 1

    asyncio.run(scenario())
    store.close()


def test_a_line_without_a_video_explains_what_to_do(home, tmp_path, make_answerer):
    loaded = load_or_create()
    store = Storage(tmp_path / "kvedio.db")
    bus = EventBus()
    pipeline = build(
        store, Workspace(tmp_path / "workspace"), bus, loaded, make_answerer(store)
    )

    async def scenario() -> None:
        app = Workbench(loaded, store, pipeline, bus)
        async with app.run_test() as pilot:
            await pilot.pause()
            prompt = app.query_one("#prompt", Input)
            prompt.value = "帮我写一封邮件"
            await pilot.press("enter")
            await pilot.pause()
            assert prompt.value == ""
            assert store.jobs() == []

    asyncio.run(scenario())
    store.close()
