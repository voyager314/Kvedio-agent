"""Terminal commands and prompted workbench for local video analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from prompt_toolkit import PromptSession
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from kvedio_agent.pipeline.analyze import AnalysisService
from kvedio_agent.pipeline.context import EvidenceIndex, Scratchpad
from kvedio_agent.pipeline.jobs import StageScheduler, WorkUnit
from kvedio_agent.pipeline.media import MediaProcessor, fingerprint
from kvedio_agent.platform.config import Config
from kvedio_agent.platform.costs import ProviderGate
from kvedio_agent.platform.providers import (
    LiteLLMChat,
    OpenAICompatibleASR,
    OpenAICompatibleVision,
)
from kvedio_agent.platform.settings import (
    Role,
    RuntimeState,
    Settings,
    ensure_settings,
    load_state,
    save_state,
    settings_path,
)
from kvedio_agent.platform.storage import Store

app = typer.Typer(name="kvedio-agent", help="Local long-video analysis workbench")
console = Console()
store = Store()


def _load_settings() -> Settings:
    """Load settings after first-run initialization or exit with a safe error."""

    try:
        settings = ensure_settings(console)
    except ValueError:
        console.print("[red]Invalid settings file. Check ~/.kvedio_ai/settings.json.[/red]")
        raise typer.Exit(code=1) from None
    if settings is None:
        raise typer.Exit(code=0)
    resolved = settings.with_state(load_state())
    store.recover_interrupted_work()
    store.cleanup_expired(Config.from_settings(resolved).retention_days)
    return resolved


def _print_models(settings: Settings) -> None:
    """Render configured models and role bindings without resolving secrets."""

    table = Table(title="Configured models")
    table.add_column("Name")
    table.add_column("Roles")
    table.add_column("Model")
    table.add_column("Key")
    for name, entry in settings.models.items():
        key_state = "configured" if entry.resolve_key() else "missing"
        table.add_row(name, ", ".join(entry.roles), entry.model, key_state)
    console.print(table)
    console.print(f"Settings: {settings_path()}")


def _job(job_id: str) -> dict[str, Any]:
    matches = [row for row in store.job_history() if row["id"].startswith(job_id)]
    if len(matches) != 1:
        raise typer.BadParameter("Job ID is missing or ambiguous.")
    return matches[0]


def _print_jobs() -> None:
    table = Table(title="Video jobs")
    table.add_column("Job")
    table.add_column("Status")
    table.add_column("Stage")
    table.add_column("Video")
    for job in store.job_history():
        table.add_row(
            job["id"][:8], job["status"], job["current_stage"] or "",
            Path(job["source_path"]).name,
        )
    console.print(table)


def _model(settings: Settings, role: Role) -> tuple[str, Any] | None:
    try:
        return settings.resolve(role)
    except LookupError:
        return None


def _chat(settings: Settings, job_id: str | None = None) -> LiteLLMChat:
    selected = _model(settings, "chat")
    if selected is None:
        raise typer.BadParameter("Bind a chat model first with `kvedio-agent bind chat NAME`.")
    gate = None
    if job_id is not None:
        gate = ProviderGate(store, selected[1], Config.from_settings(settings).max_job_cost_usd)
    return LiteLLMChat(selected[1], selected[1].timeout_seconds, gate, job_id)


def _evidence_context(
    settings: Settings, job_id: str, scratchpad: Scratchpad
) -> list[dict[str, Any]]:
    config = Config.from_settings(settings)
    history = store.job_history()
    job = next((row for row in history if row["id"] == job_id), None)
    if job is None:
        raise typer.BadParameter("Job does not exist.")
    source = Path(job["source_path"])
    source_digest = fingerprint(source).digest
    manifest_path = MediaProcessor(
        store, Config.from_settings(settings)
    ).manifest_path(source_digest)
    if not manifest_path.exists():
        raise typer.BadParameter("Media manifest is missing; resume the job to rebuild it.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    asr_binding = _model(settings, "asr")
    vision_binding = _model(settings, "vision")
    if not asr_binding and any(item.get("audio") for item in manifest["segments"]):
        raise typer.BadParameter("Bind an ASR model with `kvedio-agent bind asr NAME`.")
    asr_entry = asr_binding[1] if asr_binding else None
    vision_entry = vision_binding[1] if vision_binding else None
    total = len(manifest["segments"])
    scheduler = StageScheduler(store, config.max_asr_workers, max_attempts=3)
    modality_count = int(asr_entry is not None) + int(vision_entry is not None)
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"), console=console,
    ) as progress:
        task_id = progress.add_task(
            "Analyzing media segments", total=total * modality_count
        )
        if asr_entry:
            asr_units = [
                WorkUnit(segment["id"], segment["start_ms"], segment["end_ms"], segment)
                for segment in manifest["segments"]
                if segment.get("audio")
                and not store.evidence_exists(
                    job_id, "transcript", segment["start_ms"], segment["end_ms"]
                )
            ]

            def transcribe(unit: WorkUnit) -> Any:
                provider = OpenAICompatibleASR(
                    asr_entry,
                    asr_entry.timeout_seconds,
                    ProviderGate(store, asr_entry, config.max_job_cost_usd),
                    job_id,
                )
                return provider.transcribe(Path(unit.payload["audio"]), unit.start_ms)

            def save_transcript(unit: WorkUnit, transcript: Any) -> None:
                for item in transcript:
                    store.add_evidence(
                        job_id, "transcript", item.start_ms, item.end_ms,
                        {"provider_model": asr_entry.model}, item.text,
                        confidence=item.confidence,
                    )
                progress.advance(task_id)
                if progress.tasks[0].completed % 5 == 0:
                    console.print(
                        f"Milestone: {int(progress.tasks[0].completed)} work units complete"
                    )

            scheduler.run(job_id, "transcribe", asr_units, transcribe, save_transcript)

        if vision_entry:
            vision_units = [
                WorkUnit(segment["id"], segment["start_ms"], segment["end_ms"], segment)
                for segment in manifest["segments"]
                if segment.get("keyframes")
                and not store.evidence_exists(
                    job_id, "visual", segment["start_ms"], segment["end_ms"]
                )
            ]

            def inspect(unit: WorkUnit) -> Any:
                provider = OpenAICompatibleVision(
                    vision_entry,
                    vision_entry.timeout_seconds,
                    ProviderGate(store, vision_entry, config.max_job_cost_usd),
                    job_id,
                )
                return provider.describe(
                    list(zip(
                        [Path(path) for path in unit.payload["keyframes"]],
                        unit.payload["frame_times_ms"],
                    )),
                    unit.start_ms,
                    unit.end_ms,
                )

            def save_observation(unit: WorkUnit, observation: Any) -> None:
                store.add_evidence(
                    job_id, "visual", unit.start_ms, unit.end_ms,
                    {"provider_model": vision_entry.model}, observation.summary,
                    confidence=observation.confidence,
                )
                progress.advance(task_id)

            scheduler.max_workers = config.max_vision_workers
            scheduler.run(job_id, "inspect_visuals", vision_units, inspect, save_observation)

    service = AnalysisService(store, EvidenceIndex(store), _chat(settings, job_id))
    scratchpad.current_stage = "understand_segments"
    completed_ranges = set(scratchpad.completed_ranges)
    for segment in manifest["segments"]:
        start_ms = segment["start_ms"]
        end_ms = segment["end_ms"]
        if (start_ms, end_ms) in completed_ranges:
            continue
        transcript = store.evidence_for_range(job_id, start_ms, end_ms)
        if not transcript:
            scratchpad.failed_ranges.append((start_ms, end_ms, "missing_segment_evidence"))
            scratchpad.quality_flags.append(f"No provider evidence for {start_ms}..{end_ms} ms")
            scratchpad.save(store, job_id)
            continue
        summary_exists = any(
            item["start_ms"] == start_ms and item["end_ms"] == end_ms
            for item in store.evidence_for_modality(job_id, "segment_summary")
        )
        fact = None
        if not summary_exists:
            fact = service.understand_segment(
                job_id, start_ms, end_ms, config.retention_days
            )
        else:
            summary = next(
                item for item in store.evidence_for_modality(job_id, "segment_summary")
                if item["start_ms"] == start_ms and item["end_ms"] == end_ms
            )
            provenance = json.loads(summary["provenance_json"])
            refs = provenance.get("evidence_ids", [])
            if not refs or any(store.evidence_by_id(ref) is None for ref in refs):
                fact = service.understand_segment(
                    job_id, start_ms, end_ms, config.retention_days
                )
            else:
                fact = service.segment_fact_from_summary(summary, provenance)
        scratchpad.record_segment(
            start_ms,
            end_ms,
            list(fact.evidence_ids),
            [item for item in transcript if item["modality"] == "transcript"],
            [item for item in transcript if item["modality"] == "visual"],
            list(fact.entities),
            fact.uncertainty,
        )
        scratchpad.save(store, job_id)

    scratchpad.current_stage = "synthesize_timeline"
    scratchpad.save(store, job_id)
    chapters = service.synthesize_timeline(job_id, config.retention_days)
    scratchpad.timeline_facts = [chapter.model_dump() for chapter in chapters]
    scratchpad.current_stage = "ready_for_question"
    scratchpad.save(store, job_id)

    completed = progress.tasks[0].completed
    if completed < total:
        progress.advance(task_id, total - completed)
    console.print(f"Milestone: processed {total} media segments.")
    return EvidenceIndex(store).retrieve(job_id, "")


def _render_result(result: dict[str, Any]) -> None:
    console.print(Panel(str(result.get("quality_assessment", "")), title="Quality"))
    console.print(Panel("\n".join(result.get("plan", [])), title="Execution plan"))
    for claim in result.get("claims", []):
        ranges = []
        for evidence_id in claim["evidence_ids"]:
            row = store.evidence_by_id(evidence_id)
            if row:
                ranges.append(f"{row['start_ms']}..{row['end_ms']} ms")
        console.print(Panel(
            f"{claim['text']}\nEvidence: {', '.join(ranges)}\n"
            f"Confidence: {claim['confidence']}", title="Conclusion",
        ))
    if result.get("uncertainties"):
        console.print(Panel("\n".join(result["uncertainties"]), title="Uncertainties"))


def _run_analysis(
    video: Path, objective: str, settings: Settings, job_id: str | None = None
) -> tuple[str, str]:
    config = Config.from_settings(settings)
    provider_identity = {
        role: selected[1].model
        for role in ("chat", "asr", "vision")
        if (selected := _model(settings, role)) is not None
    }
    source_id, resolved_job_id, media = MediaProcessor(store, config).preprocess(
        video, objective, provider_identity
    )
    if job_id is not None and resolved_job_id != job_id:
        raise typer.BadParameter("The requested job no longer matches the source and objective.")
    job_id = resolved_job_id
    console.print(Panel(
        f"{video.name} · {media.duration_ms / 1000:.1f}s · job {job_id[:8]}",
        title="Video prepared",
    ))
    prior = store.checkpoint(job_id)
    scratchpad = Scratchpad.from_dict(prior) if prior else Scratchpad(objective=objective)
    scratchpad.objective = objective
    _evidence_context(settings, job_id, scratchpad)
    conversation_id = store.latest_conversation(job_id) or store.create_conversation(
        job_id, objective
    )
    result = AnalysisService(
        store, EvidenceIndex(store), _chat(settings, job_id)
    ).analyze(job_id, objective)
    store.add_message(conversation_id, "user", objective)
    evidence_ids = [item for claim in result.claims for item in claim.evidence_ids]
    store.add_message(conversation_id, "assistant", result.model_dump_json(), evidence_ids)
    store.set_job_status(job_id, "completed", "answer")
    _render_result(result.model_dump())
    return job_id, conversation_id


@app.command("models")
def models_command() -> None:
    """List configured models and role bindings."""

    _print_models(_load_settings())


@app.command("reference")
def reference_command() -> None:
    """Display the packaged provider reference document."""

    from kvedio_agent.platform.settings import render_reference

    console.print(render_reference())


@app.command("config")
def config_command() -> None:
    """Show non-secret runtime configuration."""

    for key, value in Config.from_settings(_load_settings()).model_dump().items():
        console.print(f"{key}: {value}")


@app.command("bind")
def bind_command(role: Role, model: str) -> None:
    """Persist a model binding for one role."""

    settings = _load_settings()
    try:
        settings.resolve(role, model)
    except LookupError as exc:
        raise typer.BadParameter(str(exc)) from None
    state = load_state()
    state.active[role] = model
    save_state(RuntimeState(active=state.active))
    console.print(f"Bound {role} to {model}")


@app.command("jobs")
def jobs_command() -> None:
    """List recent recoverable and completed video jobs."""

    _print_jobs()


@app.command("analyze")
def analyze_command(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    objective: Annotated[str, typer.Option("--objective", "-q")],
) -> None:
    """Preprocess and analyze a local video against an objective."""

    job_id, session_id = _run_analysis(video, objective, _load_settings())
    console.print(f"Job: {job_id}\nSession: {session_id}")


@app.command("resume")
def resume_command(job_id: str) -> None:
    """Resume a local video analysis job from its persisted media manifest."""

    job = _job(job_id)
    source = Path(job["source_path"])
    if not source.is_file():
        raise typer.BadParameter("The referenced source video is missing.")
    settings = _load_settings()
    store.set_job_status(job["id"], "interrupted", "resume")
    new_job_id, session_id = _run_analysis(source, job["objective"], settings, job["id"])
    console.print(f"Resumed job {new_job_id}; session {session_id}")


@app.command("show")
def show_command(job_id: str) -> None:
    """Show timestamped evidence for a job."""

    job = _job(job_id)
    table = Table(title=f"Evidence for {Path(job['source_path']).name}")
    table.add_column("Time")
    table.add_column("Modality")
    table.add_column("Content")
    for row in store.job_evidence(job["id"]):
        table.add_row(
            f"{row['start_ms']}..{row['end_ms']} ms", row["modality"],
            (row["content"] or "")[:160],
        )
    console.print(table)


@app.command("ask")
def ask_command(job_id: str, question: str, session_id: str | None = None) -> None:
    """Ask a follow-up in a persisted session or create a new one."""

    job = _job(job_id)
    settings = _load_settings()
    selected = session_id or store.latest_conversation(job["id"])
    if selected is None:
        selected = store.create_conversation(job["id"], question)
    result = AnalysisService(
        store, EvidenceIndex(store), _chat(settings, job["id"])
    ).follow_up(
        job["id"], selected, question
    )
    _render_result(result.model_dump())
    console.print(f"Session: {selected}")


@app.command("cancel")
def cancel_command(job_id: str) -> None:
    """Mark a job as cancelled for recovery and future worker checks."""

    job = _job(job_id)
    store.set_job_status(job["id"], "cancelled", "cancel")
    console.print(f"Cancelled {job_id}")


@app.command("delete")
def delete_command(job_id: str) -> None:
    """Delete generated job data while preserving the source video."""

    _job(job_id)
    store.delete_job(job_id)
    console.print("Deleted job data; the source video was not modified.")


@app.command("workbench")
def workbench_command() -> None:
    """Open the stateful terminal workbench."""

    _load_settings()
    from kvedio_agent.interface.tui import run_workbench

    run_workbench(store)


@app.callback(invoke_without_command=True)
def main(
    video: Annotated[Path | None, typer.Argument(help="Optional local video path")] = None,
) -> None:
    """Start the local long-video analysis workbench."""

    if video is None:
        workbench_command()
        return
    objective = PromptSession[str]().prompt("Analysis objective: ").strip()
    job_id, session_id = _run_analysis(video, objective, _load_settings())
    console.print(f"Job: {job_id}\nSession: {session_id}")
