"""Package entry point for the local long-video analysis CLI."""

from __future__ import annotations


def main() -> None:
    """Start the terminal application."""

    from kvedio_agent.interface.cli import app

    app()

