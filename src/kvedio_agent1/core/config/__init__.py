"""User configuration: what a binding is, where the file lives, which role runs on which model."""

from .loader import (
    ConfigError,
    Loaded,
    Selection,
    config_path,
    database_path,
    load,
    load_or_create,
    read_active,
    resolved_api_key,
    select,
    state_path,
    template_text,
    write_active,
)
from .schema import AppConfig, ModelBinding, Role

__all__ = [
    "AppConfig",
    "ConfigError",
    "Loaded",
    "ModelBinding",
    "Role",
    "Selection",
    "config_path",
    "database_path",
    "load",
    "load_or_create",
    "read_active",
    "resolved_api_key",
    "select",
    "state_path",
    "template_text",
    "write_active",
]
