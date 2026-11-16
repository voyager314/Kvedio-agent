from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from kvedio_agent1.core.config import (
    AppConfig,
    ConfigError,
    ModelBinding,
    Role,
    is_usable,
    load,
    load_or_create,
    resolved_api_key,
)

BINDING = {
    "model": "some-model",
    "url": "https://example.invalid/v1/chat/completions",
    "api_key": "env:KVDIO_TEST_KEY",
}


def write_config(home: Path, payload: dict) -> Path:
    target = home / "config.yml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return target


def test_first_load_creates_the_packaged_template(home):
    loaded = load_or_create()
    assert loaded.created is True
    assert loaded.path == home / "config.yml"
    assert loaded.config.for_role(Role.ASR).model == "qwen3-asr-flash"
    assert loaded.config.for_role(Role.VIDEO).url.endswith("/chat/completions")
    assert loaded.config.for_role(Role.EMBEDDING).model == "text-embedding-v3"
    assert loaded.config.for_role(Role.EMBEDDING).model == "text-embedding-v3"
    assert loaded.config.runtime.max_global_workers == 6
    assert loaded.config.runtime.asr_chunk_seconds == 300
    assert loaded.config.runtime.asr_concurrency == 2
    assert loaded.config.runtime.coverage_block_seconds == 180
    assert loaded.config.runtime.stage_budget_floor_seconds == 300
    assert loaded.config.runtime.stage_budget_scale == 1.0
    assert loaded.config.runtime.stage_budget_ceiling_seconds == 0


def test_second_load_leaves_the_user_file_alone(home):
    first = load_or_create()
    edited = first.path.read_text(encoding="utf-8").replace(
        "max_global_workers: 6", "max_global_workers: 4"
    )
    first.path.write_text(edited, encoding="utf-8")
    second = load_or_create()
    assert second.created is False
    assert second.config.runtime.max_global_workers == 4


def test_an_unknown_role_is_named_in_the_failure(home):
    path = write_config(home, {"vedio": dict(BINDING)})
    with pytest.raises(ConfigError) as failure:
        load(path)
    message = str(failure.value)
    assert "vedio" in message
    assert str(path) in message


def test_a_retired_field_is_refused_rather_than_ignored(home):
    path = write_config(home, {"supervisor": dict(BINDING, name="deepseek")})
    with pytest.raises(ConfigError, match="name"):
        load(path)


def test_a_literal_key_never_appears_in_a_failure(home):
    secret = "sk-live-a1b2c3d4"
    path = write_config(
        home,
        {"supervisor": dict(BINDING, api_key=secret), "runtime": {"max_global_workers": 0}},
    )
    with pytest.raises(ConfigError) as failure:
        load(path)
    message = str(failure.value)
    assert "runtime.max_global_workers" in message
    assert secret not in message
    assert "a1b2c3d4" not in message


def test_a_zero_budget_scale_is_refused(home):
    path = write_config(home, {"runtime": {"stage_budget_scale": 0}})

    with pytest.raises(ConfigError, match="runtime.stage_budget_scale"):
        load(path)


def test_a_binding_hides_its_key_from_repr_and_serialization():
    binding = ModelBinding(**dict(BINDING, api_key="sk-live-a1b2c3d4"))
    assert "a1b2c3d4" not in repr(binding)
    assert "api_key" not in binding.model_dump()
    assert "a1b2c3d4" not in binding.model_dump_json()


def test_the_url_is_used_exactly_as_written(home):
    address = "https://api.example.com/v1/chat/completions/?api-version=2026-01-01"
    config = load(write_config(home, {"supervisor": dict(BINDING, url=address)}))
    assert config.for_role(Role.SUPERVISOR).url == address


def test_a_url_without_a_scheme_is_refused(home):
    path = write_config(home, {"supervisor": dict(BINDING, url="api.example.com/v1")})
    with pytest.raises(ConfigError) as failure:
        load(path)
    assert "supervisor.url" in str(failure.value)


def test_an_env_reference_resolves_only_at_use(home, monkeypatch):
    monkeypatch.delenv("KVDIO_TEST_KEY", raising=False)
    binding = load(write_config(home, {"supervisor": dict(BINDING)})).for_role(Role.SUPERVISOR)
    assert binding.env_var == "KVDIO_TEST_KEY"
    assert resolved_api_key(binding) is None
    assert is_usable(binding) is False
    monkeypatch.setenv("KVDIO_TEST_KEY", "resolved-at-use")
    assert resolved_api_key(binding) == "resolved-at-use"
    assert is_usable(binding) is True


def test_a_role_left_out_is_unconfigured_not_invalid(home):
    config = load(write_config(home, {"supervisor": dict(BINDING)}))
    assert config.for_role(Role.SUPERVISOR).model == "some-model"
    assert config.for_role(Role.ASR) is None
    assert is_usable(None) is False


def test_config_is_immutable(home):
    config: AppConfig = load(write_config(home, {"supervisor": dict(BINDING)}))
    with pytest.raises(ValidationError):
        config.runtime.max_global_workers = 3
