"""Helpers for notebook-level pipeline configuration."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

try:  # pragma: no cover - import guard only
    import yaml
except Exception:  # pragma: no cover - optional dependency message
    yaml = None  # type: ignore[assignment]


DEFAULT_CONFIG_FILENAME = "pipeline_config.yaml"


def resolve_project_root(start: str | Path | None = None) -> Path:
    """Find the repository root from a notebook or script working directory."""
    root = Path.cwd().resolve() if start is None else Path(start).resolve()
    if (root / "amr").exists():
        return root
    for parent in root.parents:
        if (parent / "amr").exists():
            return parent
    raise RuntimeError(f"Could not find repository root from {root}")


def default_config_path(project_root: str | Path | None = None) -> Path:
    root = resolve_project_root(project_root)
    return root / "outputs" / DEFAULT_CONFIG_FILENAME


def load_pipeline_config(
    path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load the YAML configuration written by notebook 00."""
    if yaml is None:
        raise ImportError("PyYAML is required to load pipeline configuration. Install `pyyaml`.")
    config_path = Path(path) if path is not None else default_config_path(project_root)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing pipeline configuration: {config_path}. "
            "Run notebooks/00_configure_pipeline.ipynb first."
        )
    data = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Pipeline configuration must be a YAML mapping: {config_path}")
    return data


def write_pipeline_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """Write a notebook configuration dictionary as YAML."""
    if yaml is None:
        raise ImportError("PyYAML is required to write pipeline configuration. Install `pyyaml`.")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True))
    return out_path


def cfg(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Return a value from nested config using dot-separated keys."""
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return deepcopy(value)


def cfg_path(
    config: Mapping[str, Any],
    dotted_key: str,
    project_root: str | Path,
    default: str | Path | None = None,
) -> Path:
    """Return a config value as a path, resolving relative paths from project root."""
    raw = cfg(config, dotted_key, default)
    if raw is None:
        raise KeyError(f"Missing path configuration: {dotted_key}")
    path = Path(raw)
    if path.is_absolute():
        return path
    return Path(project_root) / path


__all__ = [
    "DEFAULT_CONFIG_FILENAME",
    "cfg",
    "cfg_path",
    "default_config_path",
    "load_pipeline_config",
    "resolve_project_root",
    "write_pipeline_config",
]
