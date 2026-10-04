"""Load, merge, validate and freeze the configuration; dump it; set the offline environment.

How to run: this module is not a script; the Orchestrator calls `load_config`. Example (PowerShell,
project root, YAML file configs/base.yaml, one value overridden on the command line):
    $env:PYTHONPATH = "src"
    python -c "from stt.config import load_config; c = load_config(__import__('pathlib').Path('configs/base.yaml'), overrides=['runtime.seed=7'], target_environ={}); print(c.runtime)"
Example output:
    online=False seed=7
Precedence (lowest to highest): defaults in code < YAML < environment (STT__SECTION__KEY, also
read from .env) < command line (--set). Real environment variables win over the .env file.
Offline by default: in offline mode HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE and HF_DATASETS_OFFLINE
are set to 1. This module makes no network call and imports no Hugging Face, torch or scipy code.
"""

from __future__ import annotations

import argparse
import difflib
import os
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from stt.config.errors import ConfigError
from stt.config.schema import AppConfig, RuntimeConfig, SectionModel, SectionRegistry
from stt.config.sources import (
    Override,
    apply_overrides,
    deep_merge,
    env_overrides,
    load_yaml_with_extends,
    parse_set,
    read_dotenv,
)

OFFLINE_VARIABLES = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")


def load_config(
    config_path: Path | None = None,
    *,
    overrides: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
    env_file: Path | None = Path(".env"),
    registry: SectionRegistry | None = None,
    apply_environment: bool = True,
    target_environ: MutableMapping[str, str] | None = None,
) -> AppConfig:
    """Build the frozen AppConfig.

    Inputs: optional YAML path (with `extends`), `--set` strings (`a.b=value`, later wins),
    the environment mapping (default: os.environ), the `.env` file (default `.env`; a missing file
    is fine), the registry of component sections (default: only `runtime`), and where the offline
    variables are written (default: os.environ; `apply_environment=False` writes nothing).
    Output: AppConfig. The same inputs always give the same AppConfig.
    Errors: ConfigError with the key, the problem and how to fix it (unknown key with the closest
    valid key, wrong type, missing value with the three ways to set it, bad YAML, bad .env line).
    """
    registry = registry or SectionRegistry()
    tree: dict[str, Any] = load_yaml_with_extends(config_path) if config_path is not None else {}
    process_env = os.environ if environ is None else environ
    merged_env = {**read_dotenv(env_file), **process_env}  # the real environment beats .env
    cli = [parse_set(item) for item in overrides]
    all_overrides: list[Override] = [*env_overrides(merged_env), *cli]
    tree = apply_overrides(tree, all_overrides)
    config = _validate(tree, registry)
    if apply_environment:
        apply_offline_environment(config, os.environ if target_environ is None else target_environ)
    return config


def apply_offline_environment(config: AppConfig, target: MutableMapping[str, str]) -> None:
    """In offline mode set HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE and HF_DATASETS_OFFLINE to 1.

    In online mode (`runtime.online: true`) nothing is set and nothing is removed.
    """
    if not config.runtime.online:
        for name in OFFLINE_VARIABLES:
            target[name] = "1"


def dump_config(config: AppConfig) -> str:
    """Return the fully resolved configuration as YAML text (used later by the RunManager).

    Loading this text again (with the same registry) gives an AppConfig equal to `config`.
    """
    data: dict[str, Any] = {"runtime": config.runtime.model_dump(mode="json")}
    for name in sorted(config.sections):
        data[name] = config.sections[name].model_dump(mode="json")
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add `--config PATH` and repeatable `--set A.B=VALUE` to an argparse parser."""
    parser.add_argument("--config", type=Path, default=None, help="YAML configuration file")
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="A.B=VALUE",
        help="override one key, for example --set runtime.seed=7 (repeatable)",
    )


def load_config_from_args(args: argparse.Namespace, **kwargs: Any) -> AppConfig:
    """Call load_config with the values parsed by `add_config_arguments`."""
    return load_config(args.config, overrides=args.config_overrides, **kwargs)


# --- validation and error messages -------------------------------------------------------------


def _validate(tree: Mapping[str, Any], registry: SectionRegistry) -> AppConfig:
    known: dict[str, type[SectionModel]] = {"runtime": RuntimeConfig, **registry.sections}
    problems: list[str] = []
    for name in sorted(tree):
        if name not in known:
            problems.append(_unknown_message(name, list(known), "section"))
    values: dict[str, SectionModel] = {}
    for name, model in known.items():
        raw = tree.get(name)
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            problems.append(f"Config section '{name}' must contain key: value pairs, but it is the single value {raw!r}.")
            continue
        try:
            values[name] = model.model_validate(dict(raw))
        except ValidationError as exc:
            problems.extend(_format_errors(name, model, exc))
    if problems:
        raise ConfigError("Invalid configuration:\n- " + "\n- ".join(problems))
    runtime = values.pop("runtime")
    assert isinstance(runtime, RuntimeConfig)
    return AppConfig(runtime=runtime, sections=values)


def _unknown_message(key: str, valid: Sequence[str], what: str, prefix: str = "") -> str:
    full = f"{prefix}{key}"
    close = difflib.get_close_matches(key, list(valid), n=1)
    hint = f" Closest valid key: '{prefix}{close[0]}'." if close else ""
    listed = ", ".join(sorted(valid)) or "(none)"
    return f"Unknown {what} '{full}'.{hint} Valid keys here: {listed}."


def _model_at(model: type[BaseModel], path: Sequence[str]) -> type[BaseModel] | None:
    """Follow `path` through nested pydantic models; None when it leaves the model tree."""
    current: type[BaseModel] = model
    for part in path:
        field = current.model_fields.get(part)
        annotation = field.annotation if field is not None else None
        if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
            return None
        current = annotation
    return current


def _format_errors(section: str, model: type[SectionModel], exc: ValidationError) -> list[str]:
    messages: list[str] = []
    for error in exc.errors():
        loc = [str(part) for part in error["loc"]]
        key = ".".join([section, *loc])
        env_name = "STT__" + "__".join(part.upper() for part in [section, *loc])
        where = f"in the YAML file, in the environment as {env_name}=..., or with --set {key}=..."
        kind = error["type"]
        if kind == "extra_forbidden":
            parent = _model_at(model, loc[:-1])
            valid = list(parent.model_fields) if parent is not None else []
            messages.append(_unknown_message(loc[-1], valid, "key", prefix=".".join([section, *loc[:-1]]) + "."))
        elif kind == "missing":
            messages.append(f"Missing required value '{key}'. Set it {where}.")
        else:
            got = error.get("input")
            messages.append(f"Key '{key}': {error['msg']} (got {got!r}). Fix the value {where}.")
    return messages
