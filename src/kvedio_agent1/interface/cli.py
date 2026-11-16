"""Process entry: subcommand parsing, dependency assembly, control handoff.

The only place where the pieces are wired together: the configuration is read
once, the database opened once, and the workbench handed collaborators that are
already built. Anything that prevents starting is reported here as one line on
stderr with a non-zero status, rather than as a traceback over the panel layout.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ..core.config import (
    ConfigError,
    Role,
    database_path,
    load_or_create,
    read_active,
    resolved_api_key,
    select,
)
from ..core.media import decoder_version
from ..core.pipeline import Pipeline
from ..core.storage import Storage, StorageError
from ..core.tasks import EventBus
from .tui import Workbench
from .tui.format import binding_label, role_label


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested subcommand and return the process status."""
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "config":
            return _show_config()
        if arguments.command == "doctor":
            return _doctor()
        return _workbench()
    except (ConfigError, StorageError) as error:
        print(f"无法启动：{error}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kvedio-agent1", description="本地长视频理解工作台")
    subcommands = parser.add_subparsers(dest="command")
    subcommands.add_parser("workbench", help="打开工作台（默认）")
    subcommands.add_parser("config", help="显示配置位置与当前绑定")
    subcommands.add_parser("doctor", help="检查配置、数据库与解码器")
    return parser


def _workbench() -> int:
    loaded = load_or_create()
    store = Storage(database_path())
    try:
        store.abandon_interrupted()
        bus = EventBus()
        pipeline = Pipeline(store, bus, loaded.config.runtime)
        Workbench(loaded, store, pipeline, bus).run()
    finally:
        store.close()
    return 0


def _show_config() -> int:
    loaded = load_or_create()
    state = read_active()
    suffix = "（本次按模板创建，请填写密钥）" if loaded.created else ""
    print(f"配置：{loaded.path}{suffix}")
    for role in Role:
        selection = select(loaded.config, role, state)
        configured = _configured(selection)
        print(f"\n{role_label(role)}（{role.value}）：{binding_label(selection, configured)}")
        if selection.fell_back:
            print(f"  注意：{selection.requested} 不在配置中，已回退")
        bindings = loaded.config.roles.for_role(role)
        if not bindings:
            print("  未声明任何绑定")
        for binding in bindings:
            current = selection.binding is not None and selection.binding.name == binding.name
            usable = "已配置" if resolved_api_key(binding) is not None else "未配置"
            mark = "*" if current else " "
            print(f" {mark} {binding.name}  {binding.model}  {binding.api_base}  {usable}")
            if binding.env_var is not None:
                print(f"     密钥取自环境变量 {binding.env_var}")
    return 0


def _doctor() -> int:
    checks: list[tuple[str, bool, str]] = []
    loaded = None
    try:
        loaded = load_or_create()
        checks.append(("配置", True, str(loaded.path)))
    except ConfigError as error:
        checks.append(("配置", False, str(error)))

    path = database_path()
    try:
        opened = Storage(path)
        opened.close()
        checks.append(("数据库", True, str(path)))
    except StorageError as error:
        checks.append(("数据库", False, str(error)))

    checks.append(("解码器", True, decoder_version()))
    for name, passed, detail in checks:
        print(f"[{'ok' if passed else 'fail'}] {name}  {detail}")

    if loaded is not None:
        state = read_active()
        for role in Role:
            selection = select(loaded.config, role, state)
            configured = _configured(selection)
            print(
                f"[{'ok' if configured else 'warn'}] {role_label(role)}  "
                f"{binding_label(selection, configured)}"
            )
    return 0 if all(passed for _, passed, _ in checks) else 1


def _configured(selection) -> bool:
    return selection.binding is not None and resolved_api_key(selection.binding) is not None
