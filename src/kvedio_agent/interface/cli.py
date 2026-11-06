"""Terminal commands and prompted workbench for local video analysis."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
from prompt_toolkit import PromptSession
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from kvedio_agent.pipeline.analyze import AnalysisResult, AnalysisService
from kvedio_agent.pipeline.context import EvidenceIndex, Scratchpad
from kvedio_agent.pipeline.jobs import StageScheduler, WorkUnit
from kvedio_agent.pipeline.media import MediaProcessor, fingerprint
from kvedio_agent.platform.config import Config
from kvedio_agent.platform.costs import ProviderGate
from kvedio_agent.platform.ingestion import LocalChunkIngestor, UploadProgress
from kvedio_agent.platform.providers import (
    LiteLLMChat,
    OpenAICompatibleASR,
    OpenAICompatibleVideo,
)
from kvedio_agent.platform.settings import Settings, ensure_settings, settings_path
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
    store.recover_interrupted_work()
    store.cleanup_expired(Config.from_settings(settings).retention_days)
    return settings


def _print_models(settings: Settings) -> None:
    """Render the configured connections without resolving secrets."""

    table = Table(title="Configured models")
    table.add_column("Capability")
    table.add_column("Model")
    table.add_column("Endpoint")
    table.add_column("Key")
    for capability, entry in (("video", settings.video), ("asr", settings.asr)):
        if entry is None:
            table.add_row(capability, "not configured", "", "")
            continue
        table.add_row(
            capability, entry.model, entry.base_url or "provider default",
            "configured" if entry.resolve_key() else "missing",
        )
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
    table.add_column("Actual USD")
    table.add_column("Est. only USD")
    table.add_column("Reserved USD")
    for job in store.job_history():
        usage = store.usage_summary(job["id"])
        table.add_row(
            job["id"][:8], job["status"], job["current_stage"] or "",
            Path(job["source_path"]).name,
            f"{usage['actual_cost_usd']:.4f}",
            f"{usage['estimated_only_cost_usd']:.4f}",
            f"{usage['reserved_cost_usd']:.4f}",
        )
    console.print(table)


def _text(settings: Settings, job_id: str | None = None) -> LiteLLMChat:
    """Open the text surface of the video connection for one job's reasoning calls."""

    entry = settings.video
    gate = None
    if job_id is not None:
        gate = ProviderGate(store, entry, Config.from_settings(settings).max_job_cost_usd)
    return LiteLLMChat(entry, entry.timeout_seconds, gate, job_id)


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
    asr_entry = settings.asr
    video_entry = settings.video
    total = len(manifest["segments"])
    scheduler = StageScheduler(store, config.max_global_workers)
    modality_count = 1 + int(asr_entry is not None)
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

            scheduler.run(
                job_id, "transcribe", asr_units, transcribe, save_transcript,
                max_workers=min(config.max_asr_workers, config.max_global_workers),
                max_attempts=asr_entry.max_attempts,
            )

        video_units = [
            WorkUnit(segment["id"], segment["start_ms"], segment["end_ms"], segment)
            for segment in manifest["segments"]
            if segment.get("clip")
            and not store.evidence_exists(
                job_id, "visual", segment["start_ms"], segment["end_ms"]
            )
        ]

        def observe(unit: WorkUnit) -> Any:
            provider = OpenAICompatibleVideo(
                video_entry,
                video_entry.timeout_seconds,
                ProviderGate(store, video_entry, config.max_job_cost_usd),
                job_id,
            )
            return provider.observe(
                Path(unit.payload["clip"]), unit.start_ms, unit.end_ms, config.max_clip_events
            )

        def save_observation(unit: WorkUnit, events: Any) -> None:
            for event in events:
                store.add_evidence(
                    job_id, "visual", event.start_ms, event.end_ms,
                    {"provider_model": video_entry.model, "entities": list(event.entities)},
                    event.text,
                )
            progress.advance(task_id)

        scheduler.run(
            job_id, "observe_clips", video_units, observe, save_observation,
            max_workers=min(config.max_video_workers, config.max_global_workers),
            max_attempts=video_entry.max_attempts,
        )

    service = AnalysisService(store, EvidenceIndex(store), _text(settings, job_id))
    scratchpad.current_stage = "understand_segments"
    completed_ranges = set(scratchpad.completed_ranges)
    milestone_ranges: list[tuple[int, int]] = []
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
        milestone_ranges.append((start_ms, end_ms))
        if len(milestone_ranges) == 5:
            milestone_start = milestone_ranges[0][0]
            milestone_end = milestone_ranges[-1][1]
            milestone_summary = service.summarize_milestone(
                job_id, milestone_start, milestone_end
            )
            console.print(Panel(
                milestone_summary,
                title=f"Key developments {milestone_start}..{milestone_end} ms",
            ))
            milestone_ranges.clear()

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


def _clip_observer(
    settings: Settings, job_id: str
) -> Callable[[int, int], list[dict[str, Any]]]:
    """Return a hook that watches one range again and records what it reports.

    Observing a range costs a provider call, so the hook runs only when a follow-up
    asks for one. What it reports is stored as evidence before it is returned, which
    is what makes a second question about the same range a database read.
    """

    config = Config.from_settings(settings)

    def observe(start_ms: int, end_ms: int) -> list[dict[str, Any]]:
        job = next((row for row in store.job_history() if row["id"] == job_id), None)
        if job is None:
            raise ValueError("job does not exist")
        source = Path(job["source_path"])
        manifest_path = MediaProcessor(store, config).manifest_path(
            fingerprint(source).digest
        )
        if not manifest_path.exists():
            raise ValueError("media manifest is missing")
        clip_path = store.root / "artifacts" / fingerprint(source).digest / "clips"
        clip_path.mkdir(parents=True, exist_ok=True)
        target = clip_path / f"{start_ms:012d}-{end_ms:012d}.mp4"
        if not target.exists():
            MediaProcessor(store, config).write_clip(source, start_ms, end_ms, target)
        entry = settings.video
        provider = OpenAICompatibleVideo(
            entry, entry.timeout_seconds,
            ProviderGate(store, entry, config.max_job_cost_usd), job_id,
        )
        events = provider.observe(target, start_ms, end_ms, config.max_clip_events)
        rows: list[dict[str, Any]] = []
        for event in events:
            evidence_id = store.add_evidence(
                job_id, "visual", event.start_ms, event.end_ms,
                {"provider_model": entry.model, "entities": list(event.entities)},
                event.text,
            )
            rows.append({
                "evidence_id": evidence_id, "start_ms": event.start_ms,
                "end_ms": event.end_ms, "modality": "visual", "content": event.text,
            })
        return rows

    return observe


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
    provider_identity: dict[str, str] = {"video": settings.video.model}
    if settings.asr is not None:
        provider_identity["asr"] = settings.asr.model
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
        store, EvidenceIndex(store), _text(settings, job_id)
    ).analyze(job_id, objective)
    store.add_message(conversation_id, "user", objective)
    evidence_ids = [item for claim in result.claims for item in claim.evidence_ids]
    store.add_message(conversation_id, "assistant", result.model_dump_json(), evidence_ids)
    store.set_job_status(job_id, "completed", "answer")
    _render_result(result.model_dump())
    return job_id, conversation_id


@app.command("models")
def models_command() -> None:
    """List the configured provider connections."""

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


@app.command("jobs")
def jobs_command() -> None:
    """List recent recoverable and completed video jobs."""

    _print_jobs()


@app.command("uploads")
def uploads_command() -> None:
    """List local chunk transfers and their durable status."""

    table = Table(title="Video uploads")
    table.add_column("Upload")
    table.add_column("Status")
    table.add_column("Size")
    table.add_column("Source")
    for upload in store.upload_history():
        table.add_row(
            upload["id"], upload["status"], f"{upload['size_bytes']} bytes",
            Path(upload["source_path"]).name,
        )
    console.print(table)


@app.command("upload")
def upload_command(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    upload_id: Annotated[str | None, typer.Option("--upload-id")] = None,
) -> None:
    """Copy an MP4 into the local workspace with resumable verified chunks."""

    try:
        result = LocalChunkIngestor(store).ingest(video, upload_id)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from None
    console.print(
        f"Upload {result.upload_id}: {result.acknowledged_chunks}/"
        f"{result.total_chunks} chunks"
    )
    if result.completed and result.target_path:
        console.print(f"Stored video: {result.target_path}")
    else:
        console.print(f"Resume with: kvedio-agent upload {video} --upload-id {result.upload_id}")


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
        store, EvidenceIndex(store), _text(settings, job["id"]),
        clip_observer=_clip_observer(settings, job["id"]),
        max_clip_ms=Config.from_settings(settings).max_segment_duration_seconds * 1000,
    ).follow_up(
        job["id"], selected, question
    )
    _render_result(result.model_dump())
    console.print(f"Session: {selected}")


@app.command("locate")
def locate_command(job_id: str, start_ms: int, end_ms: int) -> None:
    """Show transcript and visual evidence overlapping a source-time range."""

    job = _job(job_id)
    settings = _load_settings()
    result = AnalysisService(
        store, EvidenceIndex(store), _text(settings, job["id"])
    ).locate(job["id"], start_ms, end_ms)
    table = Table(title=f"Evidence {result.start_ms}..{result.end_ms} ms")
    table.add_column("Evidence")
    table.add_column("Time")
    table.add_column("Modality")
    table.add_column("Content")
    for item in result.matches:
        table.add_row(
            item.evidence_id, f"{item.start_ms}..{item.end_ms} ms",
            item.modality, item.content,
        )
    console.print(table)
    if result.uncertainty:
        console.print(Panel(result.uncertainty, title="Uncertainty"))


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

    settings = _load_settings()
    from kvedio_agent.interface.tui import run_workbench

    def follow_up(job_id: str, question: str) -> AnalysisResult:
        selected = store.latest_conversation(job_id)
        if selected is None:
            selected = store.create_conversation(job_id, question)
        return AnalysisService(
            store, EvidenceIndex(store), _text(settings, job_id),
            clip_observer=_clip_observer(settings, job_id),
            max_clip_ms=Config.from_settings(settings).max_segment_duration_seconds * 1000,
        ).follow_up(job_id, selected, question)

    def upload(video: Path) -> UploadProgress:
        return LocalChunkIngestor(store).ingest(video)

    def analyze(video: Path, objective: str) -> str:
        job_id, _ = _run_analysis(video, objective, settings)
        return job_id

    run_workbench(store, follow_up, upload, analyze)


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
