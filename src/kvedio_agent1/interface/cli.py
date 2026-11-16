"""Process entry: subcommand parsing, dependency assembly, control handoff.

The only place where the pieces are wired together: the configuration is read
once, the database opened once, and the workbench handed collaborators that are
already built. Anything that prevents starting is reported here as one line on
stderr with a non-zero status, rather than as a traceback over the panel layout.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from ..core.asr import AsrStage
from ..core.config import (
    AppConfig,
    ConfigError,
    Role,
    database_path,
    home_dir,
    is_usable,
    load_or_create,
    resolved_api_key,
)
from ..core.domain import Usage
from ..core.llm import Chat, Embedder, Transcriber, Vision
from ..core.media import decoder_version
from ..core.memory import MemoryBuilder
from ..core.pipeline import Pipeline, Stages
from ..core.storage import Storage, StorageError
from ..core.supervisor import Answering
from ..core.tasks import EventBus
from ..core.vision import EventBuilder, VisionBuilder
from ..core.workspace import Workspace
from .tui import Workbench
from .tui.format import role_label


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
        workspace = Workspace(
            loaded.config.runtime.workspace_dir or home_dir() / "workspace"
        )
        chat = _chat(loaded.config, store.record_usage)
        embedder = _embedder(loaded.config, store.record_usage)
        stages = Stages(
            runtime=loaded.config.runtime,
            workspace=workspace,
            asr=_asr_stage(loaded.config, store, workspace, store.record_usage),
            memory=_memory_stage(loaded.config, store, chat, embedder),
            vision=_vision_stage(loaded.config, store, workspace, store.record_usage),
            events=_events_stage(loaded.config, store, chat),
            answering=_answering_stage(loaded.config, store, bus, chat, embedder),
        )
        Workbench(loaded, store, Pipeline(store, bus, stages), bus).run()
    finally:
        store.close()
    return 0


def _chat(
    config: AppConfig, usage: Callable[[Usage], None] | None = None
) -> Chat | None:
    """The conversation client, or ``None`` when the role has no model or no usable key."""
    binding = config.supervisor
    key = resolved_api_key(binding) if binding is not None else None
    if binding is None or key is None:
        return None
    return Chat(
        url=binding.url,
        api_key=key,
        model=binding.model,
        timeout=config.runtime.request_timeout_seconds,
        usage=usage,
    )


def _embedder(
    config: AppConfig, usage: Callable[[Usage], None] | None = None
) -> Embedder | None:
    """The embedding client, or ``None``; without it retrieval stays keyword-only."""
    binding = config.embedding
    key = resolved_api_key(binding) if binding is not None else None
    if binding is None or key is None:
        return None
    return Embedder(
        url=binding.url,
        api_key=key,
        model=binding.model,
        timeout=config.runtime.request_timeout_seconds,
        usage=usage,
    )


def _memory_stage(
    config: AppConfig, store: Storage, chat: Chat | None, embedder: Embedder | None
) -> MemoryBuilder | None:
    """The memory builder, or ``None`` when neither half can be produced."""
    if chat is None and embedder is None:
        return None
    model = config.supervisor.model if chat is not None and config.supervisor else ""
    return MemoryBuilder(
        store,
        chat,
        embedder,
        model=model,
        block_ms=config.runtime.coverage_block_seconds * 1000,
    )


def _vision_stage(
    config: AppConfig,
    store: Storage,
    workspace: Workspace,
    usage: Callable[[Usage], None] | None = None,
) -> VisionBuilder | None:
    """The scene describer, or ``None`` when no vision model can call out."""
    binding = config.video
    key = resolved_api_key(binding) if binding is not None else None
    if binding is None or key is None:
        return None
    return VisionBuilder(
        store,
        Vision(
            url=binding.url,
            api_key=key,
            model=binding.model,
            timeout=config.runtime.request_timeout_seconds,
            usage=usage,
        ),
        workspace,
        max_ms=config.runtime.scene_max_seconds * 1000,
    )


def _events_stage(
    config: AppConfig, store: Storage, chat: Chat | None
) -> EventBuilder | None:
    """The event extractor, or ``None`` when no conversation model can call out."""
    if chat is None or config.supervisor is None:
        return None
    return EventBuilder(store, chat, model=config.supervisor.model)


def _answering_stage(
    config: AppConfig,
    store: Storage,
    bus: EventBus,
    chat: Chat | None,
    embedder: Embedder | None,
) -> Answering | None:
    """The supervisor, or ``None`` when no conversation model can call out.

    A job then fails at the answer step naming the missing model, which is more
    honest than returning a transcript that nobody was asked to read.
    """
    if chat is None or config.supervisor is None:
        return None
    return Answering(store, bus, chat, config.supervisor.model, embedder)


def _asr_stage(
    config: AppConfig,
    store: Storage,
    workspace: Workspace,
    usage: Callable[[Usage], None] | None = None,
) -> AsrStage | None:
    """Build the transcription stage from the configured role.

    Returns ``None`` when the role has no model or no usable key: the job then
    fails at transcription with a message naming the missing piece, rather than
    silently producing a video without words.
    """
    binding = config.asr
    key = resolved_api_key(binding) if binding is not None else None
    if binding is None or key is None:
        return None
    transcriber = Transcriber(
        url=binding.url,
        api_key=key,
        model=binding.model,
        timeout=config.runtime.request_timeout_seconds,
    )
    return AsrStage(
        store,
        workspace,
        transcriber.transcribe,
        model=binding.model,
        address=binding.url,
        chunk_ms=config.runtime.asr_chunk_seconds * 1000,
        concurrency=config.runtime.asr_concurrency,
        usage=usage,
    )


def _show_config() -> int:
    loaded = load_or_create()
    suffix = "（本次按模板创建，请填写密钥）" if loaded.created else ""
    print(f"配置：{loaded.path}{suffix}")
    for role in Role:
        binding = loaded.config.for_role(role)
        usable = "已配置" if is_usable(binding) else "未配置"
        print(f"\n{role_label(role)}（{role.value}）：{usable}")
        if binding is None:
            print("  未声明模型")
            continue
        print(f"  模型  {binding.model}")
        print(f"  地址  {binding.url}")
        if binding.env_var is not None:
            print(f"  密钥  环境变量 {binding.env_var}")
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
        for role in Role:
            binding = loaded.config.for_role(role)
            usable = is_usable(binding)
            model = binding.model if binding is not None else "未配置模型"
            detail = model if usable else f"{model}（缺少密钥）"
            print(f"[{'ok' if usable else 'warn'}] {role_label(role)}  {detail}")
    return 0 if all(passed for _, passed, _ in checks) else 1
