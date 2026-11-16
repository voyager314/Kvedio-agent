"""User configuration: where the file lives, which model each role uses."""

from .loader import (
    ConfigError,
    Loaded,
    config_path,
    database_path,
    home_dir,
    is_usable,
    load,
    load_or_create,
    resolved_api_key,
    template_text,
)
from .schema import AppConfig, ModelBinding, Role

__all__ = [
    "AppConfig",
    "ConfigError",
    "Loaded",
    "ModelBinding",
    "Role",
    "config_path",
    "database_path",
    "home_dir",
    "is_usable",
    "load",
    "load_or_create",
    "resolved_api_key",
    "template_text",
]
