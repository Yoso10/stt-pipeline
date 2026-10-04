"""The common skeleton of every splitter (template method): prepare, assign, assemble, verify.

How to run: this module is not a script. A new splitter subclasses `GroupSplitter`, implements
`assign(groups)` and is registered in a SplitterRegistry (see stt.splitter.factory); nothing else changes.
Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.splitter.base import GroupSplitter, Group"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from stt.datasource import AudioSample
from stt.splitter.checks import check_input, verify_result
from stt.splitter.config import SplitConfig
from stt.splitter.errors import SplitError
from stt.splitter.keys import group_keys
from stt.splitter.report import build_result
from stt.splitter.types import SplitParams, SplitResult


@dataclass(frozen=True)
class Group:
    """All samples of one group (sorted by id) and their weight in the ratio unit."""

    key: str
    samples: tuple[AudioSample, ...]
    weight: float


@dataclass
class Assignment:
    """The decision of a splitter: split name per kept sample id, plus the dropped ids and notes."""

    split_of: dict[str, str] = field(default_factory=dict)
    dropped_guard: set[str] = field(default_factory=set)
    dropped_overlap: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)
    overlap_check: str = "ok: no group appears in more than one split"


class GroupSplitter(ABC):
    """Base class: sorts the input by id, builds the groups, calls `assign`, checks the result.

    The input samples are never modified, no file is read or written.
    """

    groups_may_span_splits = False

    def __init__(self, config: SplitConfig, seed: int) -> None:
        self.config = config
        self.seed = seed

    @property
    def params(self) -> SplitParams:
        """The parameters of this splitter (saved in the split file)."""
        c = self.config
        return SplitParams(
            mode=c.mode,
            seed=self.seed,
            names=c.names,
            ratios=c.ratios,
            ratio_unit=c.ratio_unit,
            group_column=c.group_key.column,
            group_regex=c.group_key.regex,
            speaker_column=c.speaker_column,
            version=c.version,
        )

    def weight(self, sample: AudioSample) -> float:
        """The weight of a sample in the configured ratio unit."""
        return sample.duration_sec if self.config.ratio_unit == "duration" else 1.0

    def split(self, samples: Sequence[AudioSample]) -> SplitResult:
        """Divide `samples` into the configured splits by group. Same input and parameters give the same result,
        whatever the input order. Errors: SplitError (see the brief's edge cases)."""
        ordered = sorted(samples, key=lambda s: s.id)
        check_input(ordered)
        keys = group_keys(ordered, self.config.group_key.column, self.config.group_key.regex)
        members: dict[str, list[AudioSample]] = {}
        for sample in ordered:
            members.setdefault(keys[sample.id], []).append(sample)
        groups = [Group(k, tuple(v), sum(self.weight(s) for s in v)) for k, v in sorted(members.items())]
        assignment = self.assign(groups)
        result = build_result(
            ordered,
            assignment.split_of,
            assignment.dropped_guard,
            assignment.dropped_overlap,
            self.params,
            keys,
            assignment.notes,
            assignment.overlap_check,
        )
        verify_result(result, ordered, keys, self.config, self.groups_may_span_splits)
        return result

    @abstractmethod
    def assign(self, groups: Sequence[Group]) -> Assignment:
        """Decide the split of every group (or sample). `groups` are sorted by key."""

    def check_oversized(self, groups: Sequence[Group], assignment: Assignment) -> None:
        """Fail (or note, per `split.on_oversized_group`) when a group is larger than the smallest split can hold."""
        total = sum(g.weight for g in groups) or 1.0
        limit = min(self.config.ratios) + self.config.ratio_tolerance
        big = [g for g in groups if g.weight / total > limit]
        if not big:
            return
        worst = max(big, key=lambda g: g.weight)
        message = (
            f"Group '{worst.key}' holds {worst.weight / total:.1%} of the data, more than the smallest split can hold "
            f"({limit:.1%} = smallest ratio plus tolerance); {len(big)} group(s) are that large."
        )
        if self.config.on_oversized_group == "fail":
            raise SplitError(
                message + " Choose another group key or the time_block mode; raise split.ratio_tolerance only consciously, "
                "or set split.on_oversized_group to 'warn'."
            )
        assignment.notes.append("WARNING: " + message)


def raise_unknown_mode(name: str, known: Sequence[str]) -> SplitError:
    """Build the error for an unknown `split.mode`."""
    return SplitError(f"Unknown split.mode '{name}'. Registered modes: {', '.join(sorted(known))}.")
