from __future__ import annotations

import json
from importlib import resources
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from kvedio_agent.platform import settings as settings_module
from kvedio_agent.platform.config import Config
from kvedio_agent.platform.settings import ModelEntry, Settings, _strip_jsonc


def test_json_comments_preserve_urls_and_string_comment_markers() -> None:
    text = '{"url":"https://provider.example/v1","note":"a // b"} // trailing\n'
    assert json.loads(_strip_jsonc(text)) == {
        "url": "https://provider.example/v1",
        "note": "a // b",
    }


def test_model_entry_resolves_configured_secret() -> None:
    entry = ModelEntry(model="provider/model", roles=["chat"], api_key="sensitive-value")
    assert entry.resolve_key() == "sensitive-value"
    assert "sensitive-value" in entry.model_dump_json()


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
        '{"models":{"bad":{"model":"x","roles":["invalid"],"api_key":"sentinel-secret"}}}',
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


def test_first_run_creates_complete_model_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings_module, "config_dir", lambda: tmp_path)
    output = Console(file=StringIO(), force_terminal=False)

    assert settings_module.ensure_settings(output) is None
    saved = Settings.load(tmp_path / "settings.json")
    assert set(saved.models) == {"chat-model", "asr-model", "vision-model"}
    assert set(saved.active) == {"chat", "asr", "vision"}
    assert '"api_key":' in (tmp_path / "settings.json").read_text(encoding="utf-8")
    assert settings_module.ensure_settings(output) == saved


def test_first_run_template_lists_all_model_role_and_runtime_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings_module, "config_dir", lambda: tmp_path)
    settings_module.ensure_settings(Console(file=StringIO(), force_terminal=False))
    saved = Settings.load(tmp_path / "settings.json")

    assert set(saved.active) == {"chat", "asr", "vision"}
    for role in ("chat", "asr", "vision"):
        _, entry = saved.resolve(role)
        assert entry.timeout_seconds > 0
        assert entry.max_attempts > 0
        assert entry.requests_per_minute is None
        assert entry.input_cost_per_million is None
        assert entry.output_cost_per_million is None
        assert entry.audio_cost_per_minute is None
    assert set(saved.runtime) == set(Config.model_fields)
    Config.from_settings(saved)


def test_runtime_defaults_match_local_video_limits() -> None:
    config = Config()
    assert config.max_video_duration_seconds == 7200
    assert config.retention_days == 30


def test_provider_reference_is_packaged_resource() -> None:
    reference = resources.files("kvedio_agent.defaults").joinpath("reference.md")
    assert "Model Provider Reference" in reference.read_text(encoding="utf-8")

