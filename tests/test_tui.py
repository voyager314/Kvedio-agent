import asyncio
from pathlib import Path

from textual.widgets import DataTable, Input, Static

from kvedio_agent1.core.asr import AsrStage
from kvedio_agent1.core.config import load_or_create
from kvedio_agent1.core.domain import JobStatus, Segment
from kvedio_agent1.core.pipeline import Pipeline
from kvedio_agent1.core.storage import Storage
from kvedio_agent1.core.tasks import EventBus
from kvedio_agent1.core.workspace import Workspace
from kvedio_agent1.interface.tui import Workbench


def spoken(audio: Path) -> list[Segment]:
    return [Segment(start_ms=0, end_ms=600, text="开场白")]


def build(store: Storage, workspace: Workspace, bus: EventBus, loaded):
    asr = AsrStage(
        store,
        workspace,
        spoken,
        model="asr-model",
        address="https://asr.example.invalid/v1/audio/transcriptions",
        chunk_ms=300_000,
        concurrency=1,
    )
    return Pipeline(store, bus, loaded.config.runtime, workspace, asr)


def test_a_reference_travels_from_the_input_line_to_the_task_panel(home, tmp_path, sample_video):
    loaded = load_or_create()
    store = Storage(tmp_path / "kvedio.db")
    bus = EventBus()
    pipeline = build(store, Workspace(tmp_path / "workspace"), bus, loaded)

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
                timeline = str(app.query_one("#timeline", Static).render())
                if "转录：1 段" in timeline:
                    break
                await pilot.pause(0.05)
            table = app.query_one("#tasks", DataTable)
            assert table.row_count == 3
            assert "转录" in str(app.query_one("#status", Static).render())
            assert sample_video.name in str(app.query_one("#evidence", Static).render())
            assert "转录：1 段" in str(app.query_one("#timeline", Static).render())
            assert "asr-model" in str(app.query_one("#timeline", Static).render())

    asyncio.run(scenario())
    store.close()


def test_a_bad_reference_keeps_the_line_and_explains_the_usage(home, tmp_path):
    loaded = load_or_create()
    store = Storage(tmp_path / "kvedio.db")
    bus = EventBus()
    pipeline = build(store, Workspace(tmp_path / "workspace"), bus, loaded)

    async def scenario() -> None:
        app = Workbench(loaded, store, pipeline, bus)
        async with app.run_test() as pilot:
            await pilot.pause()
            prompt = app.query_one("#prompt", Input)
            prompt.value = "帮我写一封邮件"
            await pilot.press("enter")
            await pilot.pause()
            assert prompt.value == "帮我写一封邮件"
            assert store.jobs() == []

    asyncio.run(scenario())
    store.close()
