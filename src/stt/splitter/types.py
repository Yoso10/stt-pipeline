"""Typed data objects of the Splitter component and the Splitter Protocol.

How to run: this module is not a script; the other modules of stt.splitter and later components
import it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.splitter.types import SplitStats; print(SplitStats('train', 10, 3, 60.0, 0.8, 0.79))"
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from stt.datasource import AudioSample


@dataclass(frozen=True)
class SplitParams:
    """The parameters that define a split (saved in the split file)."""

    mode: str
    seed: int
    names: tuple[str, ...]
    ratios: tuple[float, ...]
    ratio_unit: str
    group_column: str | None
    group_regex: str | None
    speaker_column: str | None
    version: str


@dataclass(frozen=True)
class SplitStats:
    """One split: samples, groups, total duration, requested and actual share (by the ratio unit)."""

    name: str
    samples: int
    groups: int
    duration_sec: float
    requested_ratio: float
    actual_ratio: float


@dataclass(frozen=True)
class GroupShare:
    """A group and its share of the data (by the ratio unit)."""

    group: str
    samples: int
    duration_sec: float
    share: float


@dataclass(frozen=True)
class SplitReport:
    """What the split looks like. `source_split_table[source_split][split]` counts samples."""

    params: SplitParams
    total_samples: int
    total_groups: int
    stats: tuple[SplitStats, ...]
    largest_groups: tuple[GroupShare, ...]
    largest_group_share: float
    group_overlap_check: str
    source_split_table: Mapping[str, Mapping[str, int]]
    speaker_summary: str
    dropped_guard: int
    dropped_overlap: int
    data_fingerprint: str
    fingerprint: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        """Return the report as short plain text."""
        lines = [
            f"mode {self.params.mode}, seed {self.params.seed}, version {self.params.version}, ratio unit {self.params.ratio_unit}",
            f"samples {self.total_samples}, groups {self.total_groups}, dropped (guard gap) {self.dropped_guard}, dropped (overlap) {self.dropped_overlap}",
        ]
        for s in self.stats:
            lines.append(
                f"  {s.name}: {s.samples} samples, {s.groups} groups, {s.duration_sec / 3600:.2f} h, "
                f"ratio {s.actual_ratio:.3f} (requested {s.requested_ratio:.3f})"
            )
        top = ", ".join(f"{g.group[:8]} {g.share:.2%}" for g in self.largest_groups)
        lines.append(f"largest groups (share of data): {top}")
        lines.append(f"group overlap check: {self.group_overlap_check}")
        lines.append(f"source_split -> split: { {k: dict(v) for k, v in self.source_split_table.items()} }")
        lines.append(self.speaker_summary)
        lines.append(f"data fingerprint {self.data_fingerprint[:16]}, split fingerprint {self.fingerprint[:16]}")
        lines.extend(self.notes)
        return "\n".join(lines)


@dataclass(frozen=True)
class SplitResult:
    """The output of a Splitter.

    `splits` maps each split name to its samples (sorted by id). `dropped_guard` holds the samples
    removed by the time_block guard gap, `dropped_overlap` those removed by `on_overlap: drop_from_train`.
    `fingerprint` identifies this split (assignments plus data); `data_fingerprint` identifies the input.
    """

    splits: Mapping[str, list[AudioSample]]
    dropped_guard: list[AudioSample]
    dropped_overlap: list[AudioSample]
    report: SplitReport
    fingerprint: str
    data_fingerprint: str


class Splitter(Protocol):
    """Anything that divides samples into named splits by group."""

    def split(self, samples: Sequence[AudioSample]) -> SplitResult:
        """Return the split. Raises SplitError when a hard check or a gate fails."""
        ...
