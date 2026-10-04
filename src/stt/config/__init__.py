"""stt.config: load, merge, validate and freeze all settings into one read-only AppConfig.

How to run: from Python (project root, PowerShell):
    $env:PYTHONPATH = "src"
    python -c "from pathlib import Path; from stt.config import load_config, dump_config; print(dump_config(load_config(Path('configs/base.yaml'), env_file=None, target_environ={})))"
Example output:
    runtime:
      online: false
      seed: 42
Command line: `add_config_arguments(parser)` adds --config and --set; see docs/COMPONENTS.md (C0).
"""

from stt.config.errors import ConfigError
from stt.config.loader import (
    OFFLINE_VARIABLES,
    add_config_arguments,
    apply_offline_environment,
    dump_config,
    load_config,
    load_config_from_args,
)
from stt.config.schema import AppConfig, RuntimeConfig, SectionModel, SectionRegistry

__all__ = [
    "OFFLINE_VARIABLES",
    "AppConfig",
    "ConfigError",
    "RuntimeConfig",
    "SectionModel",
    "SectionRegistry",
    "add_config_arguments",
    "apply_offline_environment",
    "dump_config",
    "load_config",
    "load_config_from_args",
]
