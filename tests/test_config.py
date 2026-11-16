from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from kvedio_agent1.core.config import (
    AppConfig,
    ConfigError,
    ModelBinding,
    Role,
    load,
    load_or_create,
    read_active,
    resolved_api_key,
    select,
    state_path,
    template_text,
    write_active,
)

BINDING = {
    "name": "primary",
    "model": "some-model",
    "api_base": "https://example.invalid/v1",
    "api_key": "env:KVDIO_TEST_KEY",
}


def write_config(home: Path, payload: dict) -> Path:
    target = home / "config.yml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return target


def minimal(role_lists: dict | None = None) -> dict:
    roles = {
        "supervisor": [dict(BINDING)],
        "asr": [dict(BINDING, name="asr-primary")],
        "video": [dict(BINDING, name="video-primary")],
    }
    if role_lists:
        roles.update(role_lists)
    return {"roles": roles}


def test_first_load_creates_the_packaged_template(home):
    loaded = load_or_create()
    assert loaded.created is True
    assert loaded.path == home / "config.yml"
    assert loaded.path.read_text(encoding="utf-8") == template_text()
    assert loaded.config.roles.for_role(Role.ASR)[0].name == "dashscope"
    assert loaded.config.runtime.max_global_workers == 6


def test_second_load_leaves_the_user_file_alone(home):
    first = load_or_create()
    edited = first.path.read_text(encoding="utf-8").replace(
        "max_global_workers: 6", "max_global_workers: 4"
    )
    first.path.write_text(edited, encoding="utf-8")
    second = load_or_create()
    assert second.created is False
    assert second.config.runtime.max_global_workers == 4


def test_unknown_role_is_named_in_the_failure(home):
    path = write_config(home, {"roles": {"vedio": [dict(BINDING)]}})
    with pytest.raises(ConfigError) as failure:
        load(path)
    message = str(failure.value)
    assert "roles.vedio" in message
    assert str(path) in message


def test_a_repeated_binding_name_is_rejected(home):
    path = write_config(
        home,
        {"roles": {"supervisor": [dict(BINDING), dict(BINDING)], "asr": [], "video": []}},
    )
    with pytest.raises(ConfigError, match="twice"):
        load(path)


def test_a_literal_key_never_appears_in_a_failure(home):
    secret = "sk-live-a1b2c3d4"
    path = write_config(
        home,
        {
            "roles": {
                "supervisor": [dict(BINDING, api_key=secret)],
                "asr": [],
                "video": [],
            },
            "runtime": {"max_global_workers": 0},
        },
    )
    with pytest.raises(ConfigError) as failure:
        load(path)
    message = str(failure.value)
    assert "runtime.max_global_workers" in message
    assert secret not in message
    assert "a1b2c3d4" not in message


def test_a_binding_hides_its_key_from_repr_and_serialization():
    binding = ModelBinding(**dict(BINDING, api_key="sk-live-a1b2c3d4"))
    assert "a1b2c3d4" not in repr(binding)
    assert "api_key" not in binding.model_dump()
    assert "a1b2c3d4" not in binding.model_dump_json()


def test_an_env_reference_resolves_only_at_use(home, monkeypatch):
    monkeypatch.delenv("KVDIO_TEST_KEY", raising=False)
    binding = load(write_config(home, minimal())).roles.for_role(Role.SUPERVISOR)[0]
    assert binding.env_var == "KVDIO_TEST_KEY"
    assert resolved_api_key(binding) is None
    monkeypatch.setenv("KVDIO_TEST_KEY", "resolved-at-use")
    assert resolved_api_key(binding) == "resolved-at-use"


def test_nothing_declared_is_not_an_error(home):
    config = load(write_config(home, {"roles": {"supervisor": [], "asr": [], "video": []}}))
    assert config.roles.for_role(Role.VIDEO) == ()


def test_selection_follows_state_then_config_then_declaration(home):
    path = write_config(
        home,
        {
            "roles": {
                "supervisor": [dict(BINDING), dict(BINDING, name="secondary")],
                "asr": [],
                "video": [],
            },
            "active": {"supervisor": "secondary"},
        },
    )
    config = load(path)
    assert select(config, Role.SUPERVISOR, {}).binding.name == "secondary"
    assert select(config, Role.SUPERVISOR, {Role.SUPERVISOR: "primary"}).binding.name == "primary"
    assert select(config, Role.ASR, {}).binding is None


def test_a_missing_selection_is_reported_not_silent(home):
    config = load(write_config(home, minimal()))
    selection = select(config, Role.SUPERVISOR, {Role.SUPERVISOR: "gone"})
    assert selection.fell_back is True
    assert selection.requested == "gone"
    assert selection.binding.name == "primary"


def test_writing_one_selection_keeps_the_others(home):
    write_active(Role.SUPERVISOR, "primary")
    write_active(Role.ASR, "asr-primary")
    assert read_active() == {Role.SUPERVISOR: "primary", Role.ASR: "asr-primary"}
    loaded = yaml.safe_load(state_path().read_text(encoding="utf-8"))
    assert loaded == {"active": {"supervisor": "primary", "asr": "asr-primary"}}


def test_a_damaged_state_file_reads_as_empty(home):
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("active: [unclosed", encoding="utf-8")
    assert read_active() == {}


def test_config_is_immutable(home):
    config: AppConfig = load(write_config(home, minimal()))
    with pytest.raises(ValidationError):
        config.runtime.max_global_workers = 3
