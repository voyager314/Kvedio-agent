"""Terminal entry point for the local video analysis workbench foundation."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from kvedio_agent.config import Config
from kvedio_agent.settings import (
    ROLES,
    Role,
    Settings,
    ensure_settings,
    load_state,
    save_state,
    settings_path,
)

app = typer.Typer(name="kvedio-agent", help="Local long-video analysis workbench")
console = Console()


def _load_settings() -> Settings:
    """Load settings after first-run initialization or exit with a safe error."""

    try:
        settings = ensure_settings(console)
    except ValueError:
        console.print("[red]Invalid settings file. Check ~/.kvedio_ai/settings.json.[/red]")
        raise typer.Exit(code=1) from None
    if settings is None:
        raise typer.Exit(code=0)
    return settings.with_state(load_state())


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


@app.command("models")
def models_command() -> None:
    """List configured models and role bindings."""

    _print_models(_load_settings())


@app.command("config")
def config_command() -> None:
    """Show non-secret runtime configuration."""

    settings = _load_settings()
    for key, value in Config.from_settings(settings).model_dump().items():
        console.print(f"{key}: {value}")


@app.command("bind")
def bind_command(
    role: Annotated[Role, typer.Argument(help="chat, asr, or vision")],
    model: Annotated[str, typer.Argument(help="Configured model name")],
) -> None:
    """Persist a model binding for one role."""

    settings = _load_settings()
    try:
        settings.resolve(role, model)
    except LookupError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    state = load_state()
    state.active[role] = model
    save_state(state)
    console.print(f"Bound {role} to {model}")


@app.callback(invoke_without_command=True)
def main(
    video: Annotated[
        str | None, typer.Argument(help="Video path for a future analysis job")
    ] = None,
) -> None:
    """Start the local long-video analysis workbench."""

    if video:
        console.print("Video analysis is not initialized yet; use the available setup commands.")
        raise typer.Exit(code=2)
    _load_settings()
    console.print(f"Roles: {', '.join(ROLES)}")

