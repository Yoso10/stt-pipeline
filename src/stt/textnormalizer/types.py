"""Typed data objects of the TextNormalizer component and the NormalizationStep Protocol.

How to run: this module is not a script; the other modules of stt.textnormalizer and later components
import it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.textnormalizer.types import TimestampSegment; print(TimestampSegment(0.4, 6.5, 'text'))"
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from stt.datasource import AudioSample

FLAGS = (
    "had_timestamps",
    "bad_timestamps",
    "had_brackets",
    "had_digits",
    "had_latin",
    "had_nikud",
    "empty_after_normalization",
)


class NormalizationStep(Protocol):
    """One text step: a `name` and `apply(text) -> text`. Steps are selected by name through a registry."""

    name: str

    def apply(self, text: str) -> str:
        """Return `text` after this step."""
        ...


@dataclass(frozen=True)
class TimestampSegment:
    """One timed piece of text parsed from the timestamp tokens (seconds)."""

    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True)
class NormalizedSample:
    """A sample with its two texts. `sample` is unchanged: the raw text stays in `sample.text`.

    `timestamp_segments` is parsed BEFORE the tokens are stripped (None when there are no tokens or
    they are malformed). `flags` are names from FLAGS.
    """

    sample: AudioSample
    train_text: str
    eval_text: str
    timestamp_segments: tuple[TimestampSegment, ...] | None
    flags: tuple[str, ...]


@dataclass(frozen=True)
class CharStat:
    """One character of the inventory: its code point, Unicode name and counts.

    `raw` counts occurrences in the raw text (timestamp tokens masked), `train` and `eval` in the output
    of each profile, `samples` the number of raw texts that contain it.
    """

    char: str
    codepoint: str
    name: str
    raw: int
    train: int
    eval: int
    samples: int


@dataclass(frozen=True)
class NormalizeReport:
    """What was found in the text and what the profiles changed."""

    samples: int
    flag_counts: Mapping[str, int]
    changed: Mapping[str, int]
    empty_ids: Mapping[str, tuple[str, ...]]
    inventory: tuple[CharStat, ...]
    inventory_total: int
    eval_version: int
    profile_steps: Mapping[str, tuple[str, ...]]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        """Return the report as plain text (counts, flags and the character inventory)."""
        lines = [
            f"samples {self.samples}, eval_version {self.eval_version}",
            f"profiles: { {k: list(v) for k, v in self.profile_steps.items()} }",
            f"texts changed by the profile: {dict(self.changed)}",
            f"flags: {dict(self.flag_counts)}",
            f"became empty: { {k: len(v) for k, v in self.empty_ids.items()} }",
            f"character inventory ({len(self.inventory)} of {self.inventory_total} distinct characters; not Hebrew letters, digits or spaces):",
        ]
        for c in self.inventory:
            lines.append(f"  {c.codepoint} {c.char!r:8} {c.name}: raw {c.raw} in {c.samples} texts, train {c.train}, eval {c.eval}")
        lines.extend(self.notes)
        return "\n".join(lines)


@dataclass(frozen=True)
class NormalizeResult:
    """The output: samples in INPUT order and the report."""

    samples: list[NormalizedSample]
    report: NormalizeReport
