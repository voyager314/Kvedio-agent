import asyncio

from textual.widgets import DataTable, Input, Static

from kvedio_agent1.core.config import load_or_create
from kvedio_agent1.core.domain import JobStatus
from kvedio_agent1.core.pipeline import Pipeline
from kvedio_agent1.core.storage import Storage
from kvedio_agent1.core.tasks import EventBus
from kvedio_agent1.interface.tui import Workbench


def test_a_reference_travels_from_the_input_line_to_the_task_panel(home, tmp_path, sample_video):
    loaded = load_or_create()
    store = Storage(tmp_path / "kvedio.db")
    bus = EventBus()
    pipeline = Pipeline(store, bus, loaded.config.runtime)

    async def scenario() -> None:
        app = Workbench(loaded, store, pipeline, bus)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#prompt", Input).value = f"@{sample_video} 讲了什么"
            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
                jobs = store.jobs()
                if jobs and jobs[0].status is not JobStatus.RUNNING:
                    break
            jobs = store.jobs()
            assert jobs, "作业没有落库"
            assert jobs[0].status is JobStatus.SUCCEEDED, jobs[0].error
            assert app.query_one("#tasks", DataTable).row_count == 1
            assert "转录" in str(app.query_one("#status", Static).render())
            assert sample_video.name in str(app.query_one("#evidence", Static).render())

    asyncio.run(scenario())
    store.close()


def test_a_bad_reference_keeps_the_line_and_explains_the_usage(home, tmp_path):
    loaded = load_or_create()
    store = Storage(tmp_path / "kvedio.db")
    bus = EventBus()
    pipeline = Pipeline(store, bus, loaded.config.runtime)

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
