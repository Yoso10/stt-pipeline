"""Typed data objects of the DataSource component: AudioSample, Rejection, LoadReport, LoadResult.

How to run: this module is not a script; the other modules of stt.datasource and later components
import it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.datasource.types import AudioSample; from pathlib import Path; print(AudioSample('a_000001', Path('a.wav'), 'text', 'g1', 3.0, 'train', {}))"
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

ExtraValue = str | int | float | bool


@dataclass(frozen=True)
class AudioSample:
    """One labeled audio clip, uniform for every source.

    `text` is the RAW transcript, unmodified. `duration_sec` comes from the audio header.
    `extra` holds every column that is not mapped to a field above (nested `metadata` fields
    appear with the prefix `meta_`), typed according to `data.extra_types`.
    """

    id: str
    audio_path: Path
    text: str
    group_id: str
    duration_sec: float
    source_split: str | None
    extra: Mapping[str, ExtraValue]


@dataclass(frozen=True)
class Rejection:
    """A sample that was seen but not loaded: its id (or row description) and the reason."""

    id: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class Distribution:
    """Summary of a numeric field: count, min, max, mean and percentiles (5, 25, 50, 75, 95)."""

    count: int
    minimum: float
    maximum: float
    mean: float
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float


@dataclass(frozen=True)
class LoadReport:
    """What was loaded and what was rejected. Informational only: no decision is made here.

    `groups_in_several_splits` counts `group_id` values that occur in more than one `source_split`.
    """

    seen: int
    loaded: int
    rejected: int
    rejection_reasons: Mapping[str, int]
    loaded_per_split: Mapping[str, int]
    rejected_per_split: Mapping[str, int]
    duration: Distribution | None
    extra_distributions: Mapping[str, Distribution]
    missing_extra_values: Mapping[str, int]
    distinct_groups: int
    groups_in_several_splits: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        """Return the report as short plain text (counts, reasons, distributions)."""
        lines = [
            f"seen {self.seen}, loaded {self.loaded}, rejected {self.rejected}",
            f"rejection reasons: {dict(self.rejection_reasons)}",
            f"loaded per source_split: {dict(self.loaded_per_split)}",
            f"rejected per source_split: {dict(self.rejected_per_split)}",
            f"distinct group_id: {self.distinct_groups}; group_id present in more than one source_split: {self.groups_in_several_splits}",
        ]
        if self.duration is not None:
            d = self.duration
            lines.append(
                f"duration s: min {d.minimum:.2f} p50 {d.p50:.2f} mean {d.mean:.2f} p95 {d.p95:.2f} max {d.maximum:.2f}"
            )
        for name, d in self.extra_distributions.items():
            lines.append(f"{name}: n {d.count} min {d.minimum:.4g} p50 {d.p50:.4g} mean {d.mean:.4g} max {d.maximum:.4g}")
        if self.missing_extra_values:
            lines.append(f"empty values left out of extra: {dict(self.missing_extra_values)}")
        lines.extend(self.notes)
        return "\n".join(lines)


@dataclass(frozen=True)
class LoadResult:
    """The output of a DataSource: samples sorted by `id` (deterministic) and the report."""

    samples: list[AudioSample]
    report: LoadReport
    rejections: tuple[Rejection, ...] = field(default_factory=tuple)


class DataSource(Protocol):
    """Anything that can load labeled audio. Implementations: CsvAudioSource, HfParquetSource."""

    def load(self) -> LoadResult:
        """Load all samples and return them with the report. Raises DataSourceError on failure."""
        ...
