"""Group keys and hashing helpers of the Splitter.

How to run: this module is not a script. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.splitter.keys import unit_hash; print(unit_hash(42, 'lesson-1'))"
Hashing uses sha256, never Python's hash(), so results are the same on every machine and run.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from stt.datasource import AudioSample
from stt.splitter.errors import SplitError

_FIELDS = ("group_id", "id", "source_split")


def field_value(sample: AudioSample, column: str) -> str | None:
    """Value of a field of `sample` as text: group_id, id, source_split, or a key of `extra`; None if absent."""
    if column == "group_id":
        return sample.group_id
    if column == "id":
        return sample.id
    if column == "source_split":
        return sample.source_split
    value = sample.extra.get(column)
    return None if value is None else str(value)


def group_keys(samples: Sequence[AudioSample], column: str | None, regex: str | None) -> dict[str, str]:
    """Return {sample id: group key}.

    The key is `group_id` (column None) or the named field, optionally reduced by `regex` (its first
    capture group, or the whole match when the regex has no group).
    Errors: SplitError listing up to 5 examples when the field is missing or the regex does not match.
    """
    pattern = re.compile(regex) if regex else None
    keys: dict[str, str] = {}
    failures: list[str] = []
    for sample in samples:
        raw = sample.group_id if column is None else field_value(sample, column)
        if raw is None or raw == "":
            failures.append(f"{sample.id}: no value in '{column}'")
            continue
        if pattern is not None:
            match = pattern.search(raw)
            if match is None or (match.groups() and not match.group(1)):
                failures.append(f"{sample.id}: '{raw}' does not match the regex")
                continue
            raw = match.group(1) if match.groups() else match.group(0)
        keys[sample.id] = raw
    if failures:
        raise SplitError(
            f"The group key cannot be computed for {len(failures)} sample(s), for example: {'; '.join(failures[:5])}. "
            f"Fix split.group_key.column (valid: {', '.join(_FIELDS)} or a key of the sample's extra fields) "
            "or split.group_key.regex in the YAML configuration."
        )
    return keys


def unit_hash(seed: int, group: str) -> float:
    """Deterministic number in [0, 1) from sha256(seed|group)."""
    digest = hashlib.sha256(f"{seed}|{group}".encode()).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def order_hash(seed: int, group: str) -> str:
    """Deterministic sortable text from sha256(seed|group) (for a seeded, stable group order)."""
    return hashlib.sha256(f"{seed}|{group}".encode()).hexdigest()
