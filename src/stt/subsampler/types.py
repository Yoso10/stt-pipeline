"""Typed data objects of the Subsampler component and the SubsampleStrategy Protocol.

How to run: this module is not a script; the other modules and later components import it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.subsampler.types import SubsampleSize; print(SubsampleSize('minutes', 30.0))"
Example output:
    SubsampleSize(kind='minutes', value=30.0)

Additions to the brief (DECISIONS 46): `SubsampleSpec.train_split`, `.val_split` and `.min_train_samples` (the caller names the splits; the brief's spec
did not say which split is the Train).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from stt.textnormalizer import NormalizedSample

SizeKind = Literal["all", "minutes", "samples", "fraction"]


@dataclass(frozen=True)
class SubsampleSize:
    """The requested size: `all`, or minutes of audio, a number of samples, or a fraction of the minutes of audio (`value` is None for `all`)."""

    kind: SizeKind
    value: float | None = None

    def describe(self) -> str:
        """A short text such as `30 minutes`, `20 samples`, `10% of the audio` or `all`."""
        if self.kind == "all":
            return "all"
        if self.kind == "fraction":
            return f"{(self.value or 0.0) * 100:g}% of the audio"
        return f"{self.value:g} {self.kind}"


@dataclass(frozen=True)
class SubsampleSpec:
    """What to select. `train_split` is reduced in every scope; `val_split` is the split that `val: follow` reduces (None = no such split)."""

    size: SubsampleSize
    strategy: str
    scope: str
    val: str
    seed: int
    repeat_index: int
    train_split: str
    val_split: str | None
    min_train_samples: int


class SubsampleStrategy(Protocol):
    """Decides what the unit of selection is: the sample (`by_sample`) or its whole group (`by_group`)."""

    def key(self, sample: NormalizedSample) -> str:
        """The key of the unit this sample belongs to."""
        ...


@dataclass(frozen=True)
class SplitSizes:
    """One split before and after the selection (minutes of audio, samples, groups); `reduced` says whether this split was subsampled."""

    name: str
    reduced: bool
    samples_before: int
    groups_before: int
    minutes_before: float
    samples_after: int
    groups_after: int
    minutes_after: float


@dataclass(frozen=True)
class SubsampleReport:
    """What was selected. `requested` and `achieved` are in the unit of the requested size (`unit`); `overshoot` = achieved minus requested for the
    reduced Train split (at most one unit, rule 9); all three are None for `all` or a disabled subsampler."""

    enabled: bool
    requested: str
    strategy: str
    scope: str
    val: str
    seed: int
    repeat_index: int
    sizes: tuple[SplitSizes, ...]
    unit: str | None
    requested_value: float | None
    achieved_value: float | None
    overshoot: float | None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    ids_location: str | None = None

    def summary(self) -> str:
        """The report as plain text."""
        if not self.enabled:
            return "subsample: disabled, the splits are returned unchanged"
        lines = [f"subsample: requested {self.requested}, strategy {self.strategy}, scope {self.scope}, val {self.val}, seed {self.seed}, repeat {self.repeat_index}"]
        if self.overshoot is not None:
            lines.append(f"  achieved {self.achieved_value:.2f} {self.unit} for {self.requested_value:.2f} requested: overshoot {self.overshoot:+.2f} {self.unit} (at most one unit)")
        for s in self.sizes:
            change = "reduced" if s.reduced else "untouched"
            lines.append(f"  {s.name:8} {change:9} samples {s.samples_before} -> {s.samples_after}, groups {s.groups_before} -> {s.groups_after}, minutes {s.minutes_before:.1f} -> {s.minutes_after:.1f}")
        lines.extend(f"  WARNING: {w}" for w in self.warnings)
        if self.ids_location:
            lines.append(f"  ids stored in {self.ids_location}")
        return "\n".join(lines)


@dataclass(frozen=True)
class SubsampleResult:
    """The output: the splits (input order preserved inside each split), the report and a fingerprint of the spec, the data and the chosen ids."""

    splits: dict[str, list[NormalizedSample]]
    report: SubsampleReport
    fingerprint: str
