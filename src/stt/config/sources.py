"""Raw configuration sources: YAML with `extends`, the `.env` parser, environment and CLI overrides.

How to run: this module is not a script; it is used by stt.config.loader. Example (PowerShell):
    $env:PYTHONPATH = "src"
    python -c "from stt.config.sources import parse_dotenv; print(parse_dotenv('A=1  # c'))"
Example output:
    {'A': '1'}
Everything here works on plain dicts and strings: no validation (that is the loader's job),
no network, no heavy imports.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from stt.config.errors import ConfigError

ENV_PREFIX = "STT__"
_DOTENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

KeyPath = tuple[str, ...]
Override = tuple[KeyPath, str, str]  # (key path, raw value, where it came from)


def deep_merge(base: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    """Return `base` overlaid by `child`: mappings merge key by key, everything else is replaced."""
    merged: dict[str, Any] = dict(base)
    for key, value in child.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_with_extends(path: Path) -> dict[str, Any]:
    """Read a YAML file; if it has `extends: other.yaml` (relative to the file), merge the base first.

    Errors: ConfigError for a missing file, a YAML syntax error (with line), a top level that is
    not a mapping, a non-string `extends`, a missing base file or a cycle (the chain is named).
    """
    return _load_chain(path, [])


def _load_chain(path: Path, chain: list[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    names = " -> ".join(p.name for p in [*chain, resolved])
    if resolved in chain:
        raise ConfigError(f"Config 'extends' cycle: {names}. Remove one of the 'extends' lines.")
    if not resolved.is_file():
        origin = f" (extended by {chain[-1]})" if chain else ""
        raise ConfigError(
            f"Config file not found: {resolved}{origin}. "
            f"Check the path given to --config or to 'extends'. Chain: {names}."
        )
    try:
        data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark is not None else ""
        raise ConfigError(
            f"Cannot parse YAML file {resolved}{where}: {getattr(exc, 'problem', exc)}. "
            "Check the indentation and do not use tab characters."
        ) from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"Config file {resolved} must contain key: value pairs at the top level, not a {type(data).__name__}."
        )
    base_ref = data.pop("extends", None)
    if base_ref is None:
        return data
    if not isinstance(base_ref, str):
        raise ConfigError(f"'extends' in {resolved} must be a single file name (text), not {type(base_ref).__name__}.")
    base = _load_chain(resolved.parent / base_ref, [*chain, resolved])
    return deep_merge(base, data)


def parse_dotenv(text: str, source: str = ".env") -> dict[str, str]:
    """Parse the text of a `.env` file: lines `KEY=VALUE`, `#` comments, blank lines, quotes.

    Rules: an optional leading `export `; values may be wrapped in single or double quotes
    (then `#` inside is kept); in an unquoted value ` #` starts a comment.
    Errors: ConfigError naming `source` and the line number for a malformed line.
    """
    result: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not _DOTENV_KEY.match(key):
            raise ConfigError(
                f"{source} line {number}: expected KEY=VALUE (key with letters, digits, underscore), got '{raw.strip()}'."
            )
        value = value.strip()
        if value[:1] in ("'", '"'):
            end = value.find(value[0], 1)
            if end == -1:
                raise ConfigError(f"{source} line {number}: the quote in '{raw.strip()}' is not closed.")
            value = value[1:end]
        else:
            value = re.split(r"\s#", value, maxsplit=1)[0].strip()
        result[key] = value
    return result


def read_dotenv(path: Path | None) -> dict[str, str]:
    """Read a `.env` file. A missing file (or None) is allowed and gives an empty dict."""
    if path is None or not path.is_file():
        return {}
    return parse_dotenv(path.read_text(encoding="utf-8"), source=str(path))


def env_overrides(values: Mapping[str, str]) -> list[Override]:
    """Turn `STT__SECTION__KEY=value` entries into overrides (key path lower-cased), sorted by name.

    Entries without the `STT__` prefix are ignored. Errors: ConfigError for an empty key part.
    """
    found: list[Override] = []
    for name in sorted(values):
        if not name.upper().startswith(ENV_PREFIX):
            continue
        parts = tuple(part.lower() for part in name[len(ENV_PREFIX) :].split("__"))
        if not all(parts):
            raise ConfigError(
                f"Environment variable '{name}' is malformed. Use STT__SECTION__KEY, for example STT__RUNTIME__SEED=7."
            )
        found.append((parts, values[name], f"environment variable {name}"))
    return found


def parse_set(item: str) -> Override:
    """Parse one `--set a.b.c=value`. Errors: ConfigError when there is no `=` or the key is empty."""
    key, sep, value = item.partition("=")
    parts = tuple(p.strip() for p in key.split("."))
    if not sep or not all(parts):
        raise ConfigError(f"Bad --set argument '{item}'. Use --set section.key=value, for example --set runtime.seed=7.")
    return parts, value, f"--set {item}"


def _convert(raw: str) -> Any:
    """Env/CLI text stays text (pydantic converts it to the schema type); JSON lists/objects are parsed."""
    text = raw.strip()
    if text[:1] in ("[", "{"):
        try:
            return json.loads(text)
        except ValueError:
            return raw
    return raw


def apply_overrides(tree: Mapping[str, Any], overrides: Sequence[Override]) -> dict[str, Any]:
    """Return a copy of `tree` with the overrides applied in order (later ones win).

    Errors: ConfigError when a key path goes through a plain value instead of a section.
    """
    result: dict[str, Any] = deep_merge({}, tree)
    for path, raw, origin in overrides:
        node = result
        for depth, part in enumerate(path[:-1]):
            child = node.get(part)
            if child is None:
                child = node[part] = {}
            elif isinstance(child, Mapping):
                child = node[part] = dict(child)
            else:
                raise ConfigError(
                    f"Cannot set '{'.'.join(path)}' ({origin}): '{'.'.join(path[: depth + 1])}' "
                    "is a single value, not a section."
                )
            node = child
        node[path[-1]] = _convert(raw)
    return result
