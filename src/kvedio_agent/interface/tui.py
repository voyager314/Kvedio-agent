"""Stateful Textual workbench for durable video jobs and evidence."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, Static

from kvedio_agent.pipeline.analyze import AnalysisResult
from kvedio_agent.platform.ingestion import UploadProgress
from kvedio_agent.platform.storage import Store


def parse_file_references(value: str) -> tuple[list[Path], str]:
    """Resolve @-referenced local files and return the remaining prompt text.

    Quoted paths may contain spaces; unquoted paths end at whitespace. Relative
    paths use the process working directory, and every reference must exist.
    """

    matches = list(re.finditer(r'(?<!\S)@"([^"\r\n]+)"|(?<!\S)@(\S+)', value))
    paths = [
        Path(match.group(1) or match.group(2)).expanduser().resolve(strict=True)
        for match in matches
    ]
    if any(not path.is_file() for path in paths):
        raise ValueError("Every @ reference must point to an existing file.")
    objective = value
    for match in reversed(matches):
        objective = f"{objective[:match.start()]}{objective[match.end():]}"
    return paths, re.sub(r"\s+", " ", objective).strip()


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


def usage_lines(summary: dict[str, Any]) -> str:
    """Format actual spend, estimates, active reservations, and unknown calls distinctly."""

    return (
        f"Actual: ${summary['actual_cost_usd']:.4f}  "
        f"Estimated only: ${summary['estimated_only_cost_usd']:.4f} "
        f"({summary['estimated_only_calls']} calls)  "
        f"Reserved: ${summary['reserved_cost_usd']:.4f}  "
        f"Unknown: {summary['unknown_calls']} calls"
    )


class WorkbenchApp(App[None]):
    """Render durable jobs and evidence while delegating commands to the CLI."""

    CSS = """
    Screen { layout: vertical; }
    #body { height: 1fr; }
    #jobs { width: 34%; border: solid $accent; }
    #details { width: 66%; }
    #events, #evidence { height: 1fr; border: solid $panel; padding: 1; overflow-y: auto; }
    #usage { height: 3; padding: 1; }
    #ask { dock: bottom; }
    """

    def __init__(
        self, store: Store,
        follow_up: Callable[[str, str], AnalysisResult] | None = None,
        upload: Callable[[Path], UploadProgress] | None = None,
        analyze: Callable[[Path, str], str] | None = None,
    ) -> None:
        super().__init__()
        self.store = store
        self.follow_up = follow_up
        self.upload = upload
        self.analyze = analyze
        self.uploaded_video_path: Path | None = None
        self.selected_job_id: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            yield ListView(id="jobs")
            with Vertical(id="details"):
                yield Static("Select a job", id="events")
                yield Static("No evidence selected", id="evidence")
                yield Static("No usage data", id="usage")
        yield Input(placeholder='MP4 path or @"path with spaces.mp4"', id="upload")
        yield Input(placeholder='Objective and optional @"file path" references', id="objective")
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
        usage = self.store.usage_summary(self.selected_job_id)
        self.query_one("#events", Static).update(event_lines(events) or "No stage events")
        self.query_one("#evidence", Static).update(
            evidence_lines(evidence) or "No evidence"
        )
        self.query_one("#usage", Static).update(usage_lines(usage))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Submit a follow-up in a worker and render its structured result."""

        if event.input.id == "upload":
            upload = self.upload
            if event.value.strip() and upload is not None:
                try:
                    references, _ = parse_file_references(event.value.strip())
                    source = references[0] if references else Path(event.value.strip()).expanduser()
                    if source.suffix.casefold() != ".mp4":
                        raise ValueError("The upload source must be an MP4 file.")
                    self._run_upload(source, upload)
                except (OSError, ValueError) as exc:
                    self.query_one("#events", Static).update(str(exc))
            event.input.value = ""
            return
        if event.input.id == "objective":
            analyze = self.analyze
            if event.value.strip() and analyze is not None:
                try:
                    references, objective = parse_file_references(event.value)
                    videos = [path for path in references if path.suffix.casefold() == ".mp4"]
                    if len(videos) > 1:
                        raise ValueError("Reference at most one MP4 video per analysis.")
                    video = videos[0] if videos else self.uploaded_video_path
                    if video is None:
                        raise ValueError("Upload a video or reference one with @path.")
                    if not objective:
                        raise ValueError("Enter an analysis objective after the video reference.")
                    attachments = [path for path in references if path != video]
                    if attachments:
                        contents = []
                        for path in attachments:
                            if path.stat().st_size > 65_536:
                                raise ValueError(f"Referenced file exceeds 64 KiB: {path}")
                            try:
                                contents.append(
                                    f"--- {path} ---\n{path.read_text(encoding='utf-8')}"
                                )
                            except (OSError, UnicodeError):
                                raise ValueError(
                                    f"Referenced file is not readable UTF-8 text: {path}"
                                ) from None
                        objective += "\n\nReferenced files:\n" + "\n".join(contents)
                    self._run_analysis(video, objective, analyze)
                except (OSError, ValueError) as exc:
                    self.query_one("#events", Static).update(str(exc))
            event.input.value = ""
            return
        follow_up = self.follow_up
        if self.selected_job_id and event.value.strip() and follow_up is not None:
            try:
                references, question = parse_file_references(event.value)
                if references:
                    question += "\n\nUser referenced local files: " + ", ".join(
                        str(path) for path in references
                    )
                self._run_follow_up(self.selected_job_id, question, follow_up)
            except (OSError, ValueError) as exc:
                self.query_one("#events", Static).update(str(exc))
        event.input.value = ""

    @work(thread=True, exclusive=True)
    def _run_upload(
        self, source: Path, upload: Callable[[Path], UploadProgress]
    ) -> None:
        try:
            result = upload(source)
            message = (
                f"Upload {result.upload_id}: {result.acknowledged_chunks}/"
                f"{result.total_chunks} chunks"
            )
            if result.target_path:
                message += f"\nStored video: {result.target_path}"
                self.uploaded_video_path = result.target_path
            self.app.call_from_thread(self.query_one("#events", Static).update, message)
        except Exception:
            self.app.call_from_thread(
                self.query_one("#events", Static).update,
                "Upload failed; verify the MP4 path and resume ID.",
            )

    @work(thread=True, exclusive=True)
    def _run_analysis(
        self, video: Path, objective: str, analyze: Callable[[Path, str], str]
    ) -> None:
        try:
            job_id = analyze(video, objective)
            self.app.call_from_thread(
                self.query_one("#events", Static).update,
                f"Analysis started for job {job_id}",
            )
            self.app.call_from_thread(self.refresh_jobs)
        except Exception:
            self.app.call_from_thread(
                self.query_one("#events", Static).update,
                "Analysis failed; check model bindings and provider status.",
            )

    @work(thread=True, exclusive=True)
    def _run_follow_up(
        self, job_id: str, question: str,
        follow_up: Callable[[str, str], AnalysisResult],
    ) -> None:
        try:
            result = follow_up(job_id, question)
            rendered = "\n".join(
                f"{claim.text} [{', '.join(claim.evidence_ids)}]"
                for claim in result.claims
            ) or "No supported factual claims."
            if result.uncertainties:
                rendered += "\nUncertainty: " + "; ".join(result.uncertainties)
            self.app.call_from_thread(self.query_one("#events", Static).update, rendered)
            self.app.call_from_thread(self.refresh_details)
        except Exception:
            self.app.call_from_thread(
                self.query_one("#events", Static).update,
                "Follow-up failed; check provider settings and job evidence.",
            )


def run_workbench(
    store: Store,
    follow_up: Callable[[str, str], AnalysisResult],
    upload: Callable[[Path], UploadProgress],
    analyze: Callable[[Path, str], str],
) -> None:
    """Run the stateful workbench for the current local workspace."""

    WorkbenchApp(store, follow_up, upload, analyze).run()
