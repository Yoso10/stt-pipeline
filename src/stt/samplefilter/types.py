"""Typed data objects of the SampleFilter component and the SampleRule Protocol.

How to run: this module is not a script; the other modules of stt.samplefilter and later components
import it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.samplefilter.types import RuleVerdict; print(RuleVerdict(True))"
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from stt.textnormalizer import NormalizedSample

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class RuleVerdict:
    """The answer of one rule for one sample: passed, or failed with a reason."""

    passed: bool
    reason: str = ""


class SampleRule(Protocol):
    """One filter rule.

    `name` is the registry name, `kind` is `structural` or `quality`, `required_fields` lists the fields the rule
    reads (`duration_sec`, `eval_text`, `train_text` or `extra:<name>`), and `check` judges one sample.
    """

    name: str
    kind: str
    required_fields: tuple[str, ...]

    def check(self, sample: NormalizedSample) -> RuleVerdict:
        """Return the verdict for `sample`."""
        ...


@dataclass(frozen=True)
class RejectedSample:
    """A sample that failed at least one rule: its id, the failed rule names and the reasons."""

    sample_id: str
    rules: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RuleOutcome:
    """What one rule did in one split: its kind, threshold and how many samples failed it."""

    name: str
    kind: str
    threshold: float | None
    failed: int


@dataclass(frozen=True)
class FilterReport:
    """The report of one split.

    `would_reject` counts the samples that fail at least one rule (in either mode); `kept` is what the filter
    actually returns (everything in report mode). `kept_ratio` is the would-be kept share, the one the gate checks.
    """

    split: str
    mode: str
    seen: int
    kept: int
    would_reject: int
    kept_ratio: float
    rules_run: tuple[RuleOutcome, ...]
    rules_not_applied: tuple[str, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        """Return the report as plain text."""
        lines = [
            f"split {self.split}: mode {self.mode}, seen {self.seen}, kept {self.kept}, would reject {self.would_reject}, would-be kept ratio {self.kept_ratio:.4f}"
        ]
        for rule in self.rules_run:
            lines.append(f"  {rule.name} ({rule.kind}, threshold {rule.threshold}): {rule.failed} fail")
        if self.rules_not_applied:
            lines.append(f"  not applied to this split: {', '.join(self.rules_not_applied)}")
        lines.extend(self.notes)
        return "\n".join(lines)


@dataclass(frozen=True)
class FilterResult:
    """The output of `SampleFilter.filter`.

    `kept` keeps the input order. `rejected` holds the samples actually removed (empty in report mode);
    `would_reject` holds the samples that fail a rule, in both modes.
    """

    kept: list[NormalizedSample]
    rejected: list[RejectedSample]
    would_reject: list[RejectedSample]
    report: FilterReport


@dataclass(frozen=True)
class RuleParams:
    """What a rule factory receives: the effective settings of one rule."""

    threshold: float | None
    field: str | None
    measure_on: str | None
    token_counter: TokenCounter | None


@dataclass(frozen=True)
class RuleSpec:
    """The registry entry of a rule: its defaults and its factory."""

    kind: str
    default_enabled: bool
    default_threshold: float | None
    default_measure_on: str | None
    needs_threshold: bool
    needs_field: bool
    factory: Callable[[RuleParams], SampleRule]


Splits = Mapping[str, list[NormalizedSample]]
