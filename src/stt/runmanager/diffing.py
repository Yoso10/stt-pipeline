"""Configuration differences (pure functions, no files).

How to run: this module is not a script; the store uses it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.runmanager import diff_configs; print(diff_configs({'train': {'lr': 1, 'steps': 5}}, {'train': {'lr': 2, 'steps': 5}, 'eval': {'x': 1}}))"
Example output:
    {'train': {'lr': 2}, 'eval': {'x': 1}}
The diff holds the NEW value of every key that changed or was added, nested like the config. A key that exists only in the base is
written as None (the config schemas have fixed keys, so this does not happen between two full configs). Lists are compared whole.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import yaml

from stt.config import AppConfig, dump_config


def diff_configs(base: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    """Return what `new` changes compared with `base` (see the module docstring)."""
    result: dict[str, Any] = {}
    for key in new:
        if key not in base:
            result[key] = new[key]
        elif isinstance(new[key], Mapping) and isinstance(base[key], Mapping):
            inner = diff_configs(base[key], new[key])
            if inner:
                result[key] = inner
        elif new[key] != base[key]:
            result[key] = new[key]
    for key in base:
        if key not in new:
            result[key] = None
    return result


def flatten(tree: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Dotted-key view of a nested mapping: {'train.hyperparams.max_steps': 100, ...}."""
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            flat.update(flatten(value, name + "."))
        else:
            flat[name] = value
    return flat


def config_tree(config: AppConfig) -> dict[str, Any]:
    """The resolved config as a plain nested mapping (the same content as `config_full.yaml`)."""
    loaded = yaml.safe_load(dump_config(config))
    return dict(loaded) if loaded else {}


def defaults_tree(config: AppConfig) -> dict[str, Any]:
    """The defaults of the schemas of `config`: the keys that have a default value in code (keys without one, such as `data.columns`, are absent)."""
    tree: dict[str, Any] = {"runtime": type(config.runtime).model_construct().model_dump(mode="json")}
    for name in sorted(config.sections):
        tree[name] = type(config.sections[name]).model_construct().model_dump(mode="json", exclude_unset=False, warnings=False)
    return tree
