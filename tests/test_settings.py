from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from kvedio_agent import settings as settings_module
from kvedio_agent.config import Config
from kvedio_agent.settings import ModelEntry, Settings, _strip_jsonc


def test_json_comments_preserve_urls_and_string_comment_markers() -> None:
    text = '{"url":"https://provider.example/v1","note":"a // b"} // trailing\n'
    assert json.loads(_strip_jsonc(text)) == {
        "url": "https://provider.example/v1",
        "note": "a // b",
    }


def test_model_entry_resolves_only_environment_named_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROVIDER_TOKEN", "sensitive-value")
    entry = ModelEntry(model="provider/model", roles=["chat"], api_key_env="PROVIDER_TOKEN")
    assert entry.resolve_key() == "sensitive-value"
    assert "sensitive-value" not in entry.model_dump_json()


def test_model_entry_rejects_plaintext_or_legacy_api_key() -> None:
    with pytest.raises(ValueError, match="api_key is not accepted"):
        ModelEntry.model_validate(
            {"model": "provider/model", "roles": ["chat"], "api_key": "sentinel-secret"}
        )


def test_settings_resolve_requires_explicit_binding() -> None:
    settings = Settings(
        models={"talk": ModelEntry(model="provider/model", roles=["chat"])}
    )
    with pytest.raises(LookupError, match="no model is bound"):
        settings.resolve("chat")
    assert settings.resolve("chat", "talk")[0] == "talk"


def test_settings_load_does_not_echo_invalid_values(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        '{"models":{"bad":{"model":"x","roles":["chat"],"api_key":"sentinel-secret"}}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as raised:
        Settings.load(path)
    assert "sentinel-secret" not in str(raised.value)


def test_invalid_runtime_is_rejected_without_echoing_values() -> None:
    settings = Settings(runtime={"unexpected": "sentinel-secret"})
    with pytest.raises(ValueError) as raised:
        Config.from_settings(settings)
    assert "sentinel-secret" not in str(raised.value)


def test_first_run_creates_provider_free_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings_module, "config_dir", lambda: tmp_path)
    output = Console(file=StringIO(), force_terminal=False)

    assert settings_module.ensure_settings(output) is None
    saved = Settings.load(tmp_path / "settings.json")
    assert not saved.models
    assert not saved.active
    assert "api_key" not in (tmp_path / "settings.json").read_text(encoding="utf-8")
    assert settings_module.ensure_settings(output) == saved


def test_runtime_defaults_match_local_video_limits() -> None:
    config = Config()
    assert config.max_video_duration_seconds == 7200
    assert config.retention_days == 30

