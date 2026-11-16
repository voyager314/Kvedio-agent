import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "src" / "kvedio_agent1"
NOT_SOURCE = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    ".workbuddy",
    ".agents",
    ".claude",
    "__pycache__",
    "build",
    "dist",
    "temp",
}


def source_files(pattern: str) -> list[Path]:
    found: list[Path] = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [name for name in dirs if name not in NOT_SOURCE]
        found.extend(Path(root) / name for name in files if Path(name).match(pattern))
    return found


def test_tests_live_only_under_tests():
    strays = [path for path in source_files("test_*.py") if path.parent != REPO / "tests"]
    assert strays == []


def test_every_package_has_a_row_in_the_layout_table():
    listed = (REPO / "AGENTS.md").read_text(encoding="utf-8")
    packages = sorted(
        path.name for path in PACKAGE.iterdir() if path.is_dir() and path.name != "__pycache__"
    )
    assert [name for name in packages if f"`{name}/`" not in listed] == []
