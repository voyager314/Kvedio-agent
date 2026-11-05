"""Stateful Textual workbench for durable video jobs and evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, Static

from kvedio_agent.storage import Store


def job_label(job: dict[str, Any]) -> str:
    """Format a persisted job for the workbench list."""

    return f"{job['id'][:8]}  {job['status']:<12}  {Path(job['source_path']).name}"


def event_lines(events: list[dict[str, Any]]) -> str:
    """Format durable stage events as concise chronological milestones."""

    return "\n".join(
        f"{event['created_at'][:19]}  {event['kind']}: {event['message']}"
        for event in events[-30:]
    )


def evidence_lines(evidence: list[dict[str, Any]]) -> str:
    """Format timestamped evidence for the selected job."""

    return "\n".join(
        f"[{row['start_ms']}..{row['end_ms']} ms] {row['modality']}: "
        f"{(row['content'] or '')[:240]}"
        for row in evidence
    )


class WorkbenchApp(App[None]):
    """Render durable jobs and evidence while delegating commands to the CLI."""

    CSS = """
    Screen { layout: vertical; }
    #body { height: 1fr; }
    #jobs { width: 34%; border: solid $accent; }
    #details { width: 66%; }
    #events, #evidence { height: 1fr; border: solid $panel; padding: 1; overflow-y: auto; }
    #ask { dock: bottom; }
    """

    def __init__(self, store: Store) -> None:
        super().__init__()
        self.store = store
        self.selected_job_id: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            yield ListView(id="jobs")
            with Vertical(id="details"):
                yield Static("Select a job", id="events")
                yield Static("No evidence selected", id="evidence")
        yield Input(placeholder="Ask a follow-up for the selected job", id="ask")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_jobs()

    def refresh_jobs(self) -> None:
        """Reload durable jobs and preserve the selected item when possible."""

        jobs = self.store.job_history()
        view = self.query_one("#jobs", ListView)
        view.clear()
        for job in jobs:
            view.append(ListItem(Label(job_label(job)), name=job["id"]))
        if jobs and self.selected_job_id is None:
            self.selected_job_id = jobs[0]["id"]
        self.refresh_details()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.name:
            self.selected_job_id = event.item.name
            self.refresh_details()

    def refresh_details(self) -> None:
        """Render events and evidence for the selected durable job."""

        if self.selected_job_id is None:
            return
        events = self.store.job_events(self.selected_job_id)
        evidence = self.store.job_evidence(self.selected_job_id)
        self.query_one("#events", Static).update(event_lines(events) or "No stage events")
        self.query_one("#evidence", Static).update(
            evidence_lines(evidence) or "No evidence"
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Acknowledge follow-up input without running network work in the UI thread."""

        if self.selected_job_id:
            self.query_one("#events", Static).update(
                f"Follow-up queued for {self.selected_job_id[:8]}: {event.value}"
            )
        event.input.value = ""


def run_workbench(store: Store) -> None:
    """Run the stateful workbench for the current local workspace."""

    WorkbenchApp(store).run()
