"""Stateful Textual workbench for durable video jobs and evidence."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, Static

from kvedio_agent.interface.commands import (
    COMMANDS,
    command_label,
    command_spec,
    command_suggestions,
    parse_command,
)
from kvedio_agent.pipeline.analyze import AnalysisResult
from kvedio_agent.platform.config import Config
from kvedio_agent.platform.ingestion import UploadProgress
from kvedio_agent.platform.settings import Settings, render_reference
from kvedio_agent.platform.storage import Store


def parse_file_references(value: str) -> tuple[list[Path], str]:
    """Resolve @-referenced local files and return the remaining prompt text.

    Quoted paths may contain spaces; unquoted paths end at whitespace. Relative
    paths use the process working directory, and every reference must exist.
    """

    if re.search(r'(?<!\S)@"[^"\r\n]*$', value):
        raise ValueError("Close the quote around the referenced file path.")
    if re.search(r'(?<!\S)@(?=\s|$)', value):
        raise ValueError("Enter a file path after @.")
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


def referenced_text(paths: list[Path], video: Path | None = None) -> str:
    """Return bounded UTF-8 references as prompt context, excluding the video.

    paths: Existing files referenced by the user; the list is not mutated.
    video: MP4 selected as the analysis source, or None when no video is selected.
    Returns text blocks in reference order; an empty string means no text files.
    Raises ValueError for oversized or non-UTF-8 content without exposing file bytes.
    """

    blocks: list[str] = []
    for path in paths:
        if path == video:
            continue
        try:
            if path.stat().st_size > 65_536:
                raise ValueError("Referenced text files must not exceed 64 KiB.")
            content = path.read_text(encoding="utf-8")
        except OSError:
            raise ValueError("A referenced file could not be read.") from None
        except UnicodeError:
            raise ValueError(
                "Referenced files must be MP4 videos or readable UTF-8 text."
            ) from None
        blocks.append(f"--- {path} ---\n{content}")
    return "\n\nReferenced files:\n" + "\n".join(blocks) if blocks else ""


def file_reference_context(value: str) -> tuple[int, str, bool] | None:
    """Return the trailing @ path token's start, typed path, and quote state."""

    match = re.search(r'(?<!\S)@(?:"([^"\r\n]*)|([^\s]*))$', value)
    if match is None:
        return None
    path = match.group(1) if match.group(1) is not None else match.group(2) or ""
    return match.start(), path, match.group(1) is not None


def file_path_suggestions(value: str, limit: int = 12) -> list[tuple[str, str]]:
    """Return local file and directory completions for a trailing @ path.

    value: Complete prompt text ending in the path token; it is not mutated.
    limit: Maximum suggestions, a positive number of local entries.
    Returns display paths and replacement paths; an empty list means no matches.
    """

    context = file_reference_context(value)
    if context is None or limit <= 0:
        return []
    _, typed_path, _ = context
    typed = Path(typed_path).expanduser()
    has_separator = typed_path.endswith(("/", "\\"))
    directory = typed if has_separator else typed.parent
    name_prefix = "" if has_separator else typed.name
    if not directory.is_absolute():
        directory = Path.cwd() / directory
    try:
        entries = sorted(
            (item for item in directory.iterdir() if item.is_dir() or item.is_file()),
            key=lambda item: (not item.is_dir(), item.name.casefold()),
        )
    except OSError:
        return []
    suggestions: list[tuple[str, str]] = []
    for item in entries:
        if not item.name.casefold().startswith(name_prefix.casefold()):
            continue
        try:
            relative = item.relative_to(Path.cwd())
            displayed = str(relative)
        except ValueError:
            displayed = str(item)
        if item.is_dir():
            displayed += "/"
        replacement = str(item) + ("/" if item.is_dir() else "")
        suggestions.append((displayed, replacement))
        if len(suggestions) == limit:
            break
    return suggestions


def resolve_job(job_history: list[dict[str, Any]], selector: str) -> dict[str, Any]:
    """Resolve one job by a unique ID prefix.

    job_history: Current job records returned by Store; it is not mutated.
    selector: Full job ID or a non-empty unique prefix.
    Returns the matching job; raises ValueError when the prefix is absent or ambiguous.
    """

    matches = [job for job in job_history if job["id"].startswith(selector)]
    if len(matches) != 1:
        raise ValueError("Job ID is missing or ambiguous.")
    return matches[0]


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
    #prompt { dock: bottom; }
    #suggestions { height: auto; max-height: 8; padding: 0 1; border: solid $panel; display: none; }
    #suggestions.visible { display: block; }
    """

    def __init__(
        self, store: Store,
        follow_up: Callable[[str, str], AnalysisResult] | None = None,
        upload: Callable[[Path], UploadProgress] | None = None,
        analyze: Callable[[Path, str], str] | None = None,
        settings: Settings | None = None,
        resume: Callable[[str], str] | None = None,
        locate: Callable[[str, int, int], str] | None = None,
    ) -> None:
        """Bind terminal actions to existing workspace callbacks.

        store: Durable local job, evidence, and upload store; retained by reference.
        follow_up: Existing job/question handler, or None when unavailable.
        upload: Existing MP4 upload handler, or None when unavailable.
        analyze: Existing video/objective handler, or None when unavailable.
        settings: Validated user settings for display, or None when unavailable.
        resume: Existing job resume handler, or None when unavailable.
        locate: Existing time-range evidence handler, or None when unavailable.
        Returns None; the supplied callbacks are not invoked during construction.
        """

        super().__init__()
        self.store = store
        self.follow_up = follow_up
        self.upload = upload
        self.analyze = analyze
        self.settings = settings
        self.resume = resume
        self.locate = locate
        self.uploaded_video_path: Path | None = None
        self.selected_job_id: str | None = None
        self.suggestion_index = 0
        self._suggestion_values: list[tuple[str, str]] = []
        self.operation_running = False
        self.exit_requested = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            yield ListView(id="jobs")
            with Vertical(id="details"):
                yield Static("Select a job", id="events")
                yield Static("No evidence selected", id="evidence")
                yield Static("No usage data", id="usage")
        yield Static(id="suggestions")
        yield Input(
            placeholder='Ask a question, use / for commands, or @"file path" for context',
            id="prompt",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_jobs()
        self.query_one("#prompt", Input).focus()

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
            self.query_one("#events", Static).update("Select a job")
            self.query_one("#evidence", Static).update("No evidence selected")
            self.query_one("#usage", Static).update("No usage data")
            return
        events = self.store.job_events(self.selected_job_id)
        evidence = self.store.job_evidence(self.selected_job_id)
        usage = self.store.usage_summary(self.selected_job_id)
        self.query_one("#events", Static).update(event_lines(events) or "No stage events")
        self.query_one("#evidence", Static).update(
            evidence_lines(evidence) or "No evidence"
        )
        self.query_one("#usage", Static).update(usage_lines(usage))

    def on_input_changed(self, event: Input.Changed) -> None:
        """Refresh command or file suggestions for the unified prompt."""

        if event.input.id == "prompt":
            self._update_suggestions(event.value)

    def on_key(self, event: events.Key) -> None:
        """Handle prompt navigation keys before Textual submits the input."""

        if event.key in {"ctrl+c", "ctrl+d"}:
            event.stop()
            self._request_exit(force=True)
            return
        if not self._suggestion_values:
            return
        if event.key in {"up", "down"}:
            event.stop()
            delta = -1 if event.key == "up" else 1
            self.suggestion_index = (self.suggestion_index + delta) % len(self._suggestion_values)
            self._render_suggestions()
        elif event.key == "tab":
            event.stop()
            self._accept_suggestion()
        elif event.key == "escape":
            event.stop()
            self._clear_suggestions()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Route the unified prompt to a command, analysis, or follow-up."""

        if event.input.id != "prompt":
            return
        value = event.value.strip()
        event.input.value = ""
        self._clear_suggestions()
        if not value:
            return
        try:
            parsed = parse_command(value)
            if parsed is not None:
                self._execute_command(*parsed)
            else:
                self._submit_prompt(value)
        except (OSError, ValueError) as exc:
            self._show_message(str(exc))

    def _update_suggestions(self, value: str) -> None:
        """Refresh the visible completion list for one prompt value.

        value: Current prompt text; it is not mutated.
        Returns None; command suggestions take precedence over file suggestions.
        """

        self.suggestion_index = 0
        if value.startswith("/") and " " not in value and "\t" not in value:
            self._suggestion_values = [
                (
                    f"{command_label(command)}  {command.description}",
                    f"/{command.name} ",
                )
                for command in command_suggestions(value)
            ]
        elif file_reference_context(value) is not None:
            self._suggestion_values = [
                (path, replacement)
                for path, replacement in file_path_suggestions(value)
            ]
        else:
            self._suggestion_values = []
        self._render_suggestions()

    def _render_suggestions(self) -> None:
        """Render the selected completion list or hide its panel when empty."""

        widget = self.query_one("#suggestions", Static)
        if not self._suggestion_values:
            widget.remove_class("visible")
            widget.update("")
            return
        widget.add_class("visible")
        lines = []
        for index, (label, _) in enumerate(self._suggestion_values):
            marker = "> " if index == self.suggestion_index else "  "
            lines.append(f"{marker}{label}")
        widget.update(Text("\n".join(lines)))

    def _clear_suggestions(self) -> None:
        self._suggestion_values = []
        self.suggestion_index = 0
        self._render_suggestions()

    def _accept_suggestion(self) -> None:
        """Insert the selected command or file path into the prompt."""

        input_widget = self.query_one("#prompt", Input)
        _, insertion = self._suggestion_values[self.suggestion_index]
        current = input_widget.value
        context = file_reference_context(current)
        if current.startswith("/") and context is None:
            input_widget.value = insertion
        else:
            if context is None:
                return
            start, _, quoted = context
            replacement = insertion
            if " " in replacement or quoted:
                replacement = f'"{replacement}"'
            input_widget.value = current[:start] + "@" + replacement
        input_widget.cursor_position = len(input_widget.value)
        self._clear_suggestions()

    def _submit_prompt(self, value: str, force_analysis: bool = False) -> None:
        """Route natural input or an explicit analyze request to an existing callback.

        value: Submitted question or analysis objective, including @ references.
        force_analysis: Start a new job even when an existing job is selected.
        Returns None; input validation failures are raised as ValueError.
        """

        if self.operation_running:
            raise ValueError("Wait for the current operation to finish before submitting another.")
        references, text = parse_file_references(value)
        videos = [path for path in references if path.suffix.casefold() == ".mp4"]
        if len(videos) > 1:
            raise ValueError("Reference at most one MP4 video per analysis.")
        if self.selected_job_id is not None and not force_analysis:
            if videos:
                raise ValueError("MP4 references select a source for new analysis, not follow-up.")
            if not text and not references:
                raise ValueError("Enter a follow-up question.")
            question = text + referenced_text(references)
            if self.follow_up is None:
                raise ValueError("Follow-up analysis is not available in this workbench.")
            self.operation_running = True
            self.exit_requested = False
            self._run_follow_up(self.selected_job_id, question, self.follow_up)
            return
        video = videos[0] if videos else self.uploaded_video_path
        if video is None:
            raise ValueError("Reference an MP4 video with @path before asking for analysis.")
        if not text:
            raise ValueError("Enter an analysis objective after the video reference.")
        if self.analyze is None:
            raise ValueError("Analysis is not available in this workbench.")
        self.operation_running = True
        self.exit_requested = False
        self._run_analysis(video, text + referenced_text(references, video), self.analyze)

    def _execute_command(self, name: str, arguments: str) -> None:
        """Dispatch a parsed command to its terminal-only handler.

        name: Case-folded registered command name or alias.
        arguments: Remaining user text after the command name; it is not mutated.
        Returns None; malformed or unknown commands raise ValueError for display.
        """

        command = command_spec(name)
        if command is None:
            raise ValueError(f"Unknown command: /{name}")
        argument_commands = {
            "upload", "analyze", "ask", "resume", "locate", "show", "cancel", "delete"
        }
        if arguments and command.name not in argument_commands:
            raise ValueError(f"Usage: {command_label(command)}")
        if command.name == "help":
            self._show_message("\n".join(
                f"{command_label(item)}  {item.description}"
                for item in COMMANDS
            ))
        elif command.name == "exit":
            self._request_exit()
        elif command.name == "model":
            self._show_models()
        elif command.name == "jobs":
            self.refresh_jobs()
            self._show_message("Job list refreshed.")
        elif command.name == "uploads":
            self._show_uploads()
        elif command.name == "upload":
            self._command_upload(arguments)
        elif command.name == "analyze":
            self._command_analyze(arguments)
        elif command.name == "ask":
            if not arguments or not self.selected_job_id:
                raise ValueError("Select a job and provide a follow-up question.")
            self._submit_prompt(arguments)
        elif command.name == "resume":
            if not arguments or self.resume is None:
                raise ValueError("Usage: /resume <job-id>")
            self._run_command(self.resume, arguments, "Resume started")
        elif command.name == "show":
            self._show_job(arguments or self.selected_job_id)
        elif command.name == "locate":
            self._command_locate(arguments)
        elif command.name == "cancel":
            self._command_cancel(arguments or self.selected_job_id)
        elif command.name == "delete":
            self._command_delete(arguments or self.selected_job_id)
        elif command.name == "config":
            self._show_config()
        elif command.name == "reference":
            self._show_message(render_reference())

    def _command_upload(self, value: str) -> None:
        """Schedule the existing uploader for one referenced or literal MP4 path.

        value: Non-empty path argument, optionally expressed as an @ reference.
        Returns None; invalid paths or missing upload wiring raise ValueError.
        """

        if self.operation_running:
            raise ValueError("Wait for the current operation to finish before submitting another.")
        if self.upload is None or not value:
            raise ValueError("Usage: /upload <@video.mp4>")
        references, _ = parse_file_references(value)
        if len(references) > 1:
            raise ValueError("Upload one MP4 file at a time.")
        source = references[0] if references else Path(value).expanduser().resolve(strict=True)
        if source.suffix.casefold() != ".mp4":
            raise ValueError("The upload source must be an MP4 file.")
        self.operation_running = True
        self.exit_requested = False
        self._run_upload(source, self.upload)

    def _command_analyze(self, value: str) -> None:
        """Start a new analysis from an explicit objective command.

        value: Objective and optional local file references; it is not mutated.
        Returns None; the existing analysis callback performs the work.
        """

        if not value:
            raise ValueError("Usage: /analyze <objective>")
        self._submit_prompt(value, force_analysis=True)

    def _command_locate(self, value: str) -> None:
        """Schedule evidence lookup for two source-relative millisecond bounds.

        value: Two whitespace-separated non-negative millisecond values.
        Returns None; malformed ranges are reported by the existing callback.
        """

        if self.operation_running:
            raise ValueError("Wait for the current operation to finish before submitting another.")
        locate = self.locate
        job_id = self.selected_job_id
        if locate is None or not job_id:
            raise ValueError("Select a job before using /locate.")
        fields = value.split()
        if len(fields) != 2:
            raise ValueError("Usage: /locate <start-ms> <end-ms>")
        try:
            start_ms, end_ms = (int(field) for field in fields)
        except ValueError:
            raise ValueError("Time bounds must be integer milliseconds.") from None
        if start_ms < 0 or end_ms <= start_ms:
            raise ValueError("Time bounds must be non-negative and ordered.")
        self._run_command(
            lambda _: locate(job_id, start_ms, end_ms),
            "",
            "Evidence loaded",
        )

    def _command_cancel(self, selector: str | None) -> None:
        """Mark the selected job or one unique ID prefix as cancelled.

        selector: Full job ID, unique prefix, or None to use the selected job.
        Returns None; a missing or ambiguous job raises ValueError.
        """

        if self.operation_running:
            raise ValueError("Wait for the current operation to finish before cancelling a job.")
        if not selector:
            raise ValueError("Select a job or provide a job ID.")
        job = resolve_job(self.store.job_history(), selector)
        self.store.set_job_status(job["id"], "cancelled", "cancel")
        self.refresh_jobs()
        self._show_message(f"Cancelled {job['id'][:8]}")

    def _command_delete(self, selector: str | None) -> None:
        """Delete generated data for the selected job or one unique ID prefix.

        selector: Full job ID, unique prefix, or None to use the selected job.
        Returns None; source video files remain untouched.
        """

        if self.operation_running:
            raise ValueError("Wait for the current operation to finish before deleting a job.")
        if not selector:
            raise ValueError("Select a job or provide a job ID.")
        job = resolve_job(self.store.job_history(), selector)
        self.store.delete_job(job["id"])
        self.selected_job_id = None
        self.refresh_jobs()
        self._show_message("Deleted job data; the source video was not modified.")

    def _show_job(self, selector: str | None) -> None:
        """Select one job and render its timestamped evidence.

        selector: Full job ID, unique prefix, or None to use the selected job.
        Returns None; a missing or ambiguous job raises ValueError.
        """

        if not selector:
            raise ValueError("Select a job or provide a job ID.")
        job = resolve_job(self.store.job_history(), selector)
        self.selected_job_id = job["id"]
        self.refresh_details()
        self._show_message(evidence_lines(self.store.job_evidence(job["id"])) or "No evidence")

    def _show_uploads(self) -> None:
        rows = self.store.upload_history()
        self._show_message("\n".join(
            f"{row['id']}  {row['status']}  {Path(row['source_path']).name}"
            for row in rows
        ) or "No uploads")

    def _show_models(self) -> None:
        """Display configured model IDs and endpoint hosts without credentials."""

        if self.settings is None:
            raise ValueError("Model settings are unavailable in this workbench.")
        rows = []
        for capability, entry in (("video", self.settings.video), ("asr", self.settings.asr)):
            if entry is None:
                rows.append(f"{capability}: not configured")
            else:
                host = urlsplit(entry.base_url).hostname if entry.base_url else None
                endpoint = host or "provider default"
                rows.append(f"{capability}: {entry.model} ({endpoint})")
        self._show_message("\n".join(rows))

    def _show_config(self) -> None:
        """Display validated runtime limits without provider credentials."""

        if self.settings is None:
            raise ValueError("Runtime settings are unavailable in this workbench.")
        values = Config.from_settings(self.settings).model_dump().items()
        self._show_message("\n".join(
            f"{key}: {value}" for key, value in values
        ))

    def _show_message(self, message: str) -> None:
        self.query_one("#events", Static).update(message)

    def _request_exit(self, force: bool = False) -> None:
        """Exit immediately or require a second request while work is running.

        force: True bypasses the slash-command confirmation for terminal exit keys.
        Returns None; active synchronous provider calls may continue in their threads.
        """

        if self.operation_running and not self.exit_requested and not force:
            self.exit_requested = True
            self._show_message(
                "An operation is running and may continue after exit. "
                "Press Ctrl-C or run /exit again to close the workbench."
            )
            return
        self.exit()

    def _run_command(self, callback: Callable[[str], str], value: str, prefix: str) -> None:
        """Run a synchronous existing command callback on a worker thread.

        callback: Existing operation accepting value and returning display text.
        value: Validated command argument passed without mutation.
        prefix: Short status label rendered with the callback result.
        Returns None; callback failures become a generic terminal message.
        """

        if self.operation_running:
            self._show_message(
                "Wait for the current operation to finish before running another command."
            )
            return
        self.operation_running = True
        self.exit_requested = False

        def run() -> None:
            try:
                message = callback(value)
                rendered = f"{prefix}: {message}" if message else prefix
                self.app.call_from_thread(self._show_message, rendered)
            except Exception:
                self.app.call_from_thread(
                    self._show_message,
                    "Command failed; check the selected job and settings.",
                )
            finally:
                self.app.call_from_thread(self._finish_operation)

        self.run_worker(run, name="prompt-command", exclusive=True, thread=True)

    def _finish_operation(self) -> None:
        """Clear the interaction guard after an operation worker completes."""

        self.operation_running = False
        self.exit_requested = False

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
        finally:
            self.app.call_from_thread(self._finish_operation)

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
        finally:
            self.app.call_from_thread(self._finish_operation)

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
        finally:
            self.app.call_from_thread(self._finish_operation)


def run_workbench(
    store: Store,
    follow_up: Callable[[str, str], AnalysisResult],
    upload: Callable[[Path], UploadProgress],
    analyze: Callable[[Path, str], str],
    settings: Settings | None = None,
    resume: Callable[[str], str] | None = None,
    locate: Callable[[str, int, int], str] | None = None,
) -> None:
    """Run the stateful terminal workbench.

    store: Durable local job and upload state.
    follow_up: Existing job/question handler.
    upload: Existing local chunk uploader.
    analyze: Existing video/objective workflow.
    settings: Validated model settings for read-only display, or None.
    resume: Existing job resume workflow, or None.
    locate: Existing timestamped evidence lookup, or None.
    Returns None after Textual exits; callbacks are passed to the app unchanged.
    """

    WorkbenchApp(store, follow_up, upload, analyze, settings, resume, locate).run()
