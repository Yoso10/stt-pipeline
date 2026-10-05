"""Typed data objects and Protocols of the Selector component.

How to run: this module is not a script; the other modules and later components import it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.selector.types import SelectionResult; print(list(SelectionResult.__dataclass_fields__)[:4])"
Example output:
    ['verdict', 'winner', 'ranking', 'comparison_table']

No file access and no model imports. Additions to the brief (DECISIONS 51): `SelectionRequest.test_split`, `SelectionResult.test_baseline`, `.test_winner`,
`.test_error`, `.warnings`, `ComparisonRow.rank`, `.eligible`, `.note`, the `ValComparator` Protocol.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from stt.evaluator import ComparisonResult, EvalResult, ReviewList

ParamValue = float | int | str
Verdict = Literal["winner", "baseline_retained"]
TestStatus = Literal["confirmed", "not_confirmed", "not_evaluated"]


class ValComparator(Protocol):
    """What the Selector needs from the Evaluator's `Comparator`: the comparison of a candidate with the baseline (the real Comparator fits)."""

    def compare(self, baseline: EvalResult, candidate: EvalResult) -> ComparisonResult:
        """Compare; positive `delta_wer` means the candidate is better."""
        ...


@dataclass(frozen=True)
class Candidate:
    """A finalist: its search parameters, its model folder, its Val evaluation (by WER), its Val loss and its steps."""

    trial_id: int
    params: Mapping[str, ParamValue]
    model_dir: Path
    val: EvalResult
    val_loss: float
    steps: int


@dataclass(frozen=True)
class SelectionRequest:
    """Everything one selection needs. `evaluate_test(candidate)` is the ONE-TIME Test evaluation supplied by the Orchestrator (`None` = the baseline model);
    it is called at most once per model. `test_split` is the name of the Test split: every Val result must NOT carry it and every Test result MUST."""

    baseline_val: EvalResult
    candidates: Sequence[Candidate]
    comparator: ValComparator
    evaluate_test: Callable[[Candidate | None], EvalResult]
    test_split: str = "test"


@dataclass(frozen=True)
class RankedCandidate:
    """A finalist with its comparison against the baseline on Val and the plain-words reason for its eligibility."""

    candidate: Candidate
    delta_wer_vs_baseline: float | None
    comparison: ComparisonResult
    eligible: bool
    reason: str


@dataclass(frozen=True)
class ComparisonRow:
    """One row of the comparison table (all finalists side by side, the winner first, then the others in a fixed order)."""

    rank: int
    trial_id: int
    params: Mapping[str, ParamValue]
    val_wer: float
    val_cer: float | None
    delta_wer: float | None
    ci_low: float | None
    ci_high: float | None
    verdict: str
    val_loss: float
    steps: int
    eligible: bool
    note: str


@dataclass(frozen=True)
class SelectionResult:
    """The outcome. `reason` is one plain sentence block for a human. `review_list` is the Evaluator's list of the winner (or of the baseline when it stays),
    passed through unchanged. The Test fields are None unless the Test was evaluated (`test_status` says); `test_error` holds the message when the Test
    evaluation failed (the winner is then kept and nothing is hidden)."""

    verdict: Verdict
    winner: Candidate | None
    ranking: list[RankedCandidate]
    comparison_table: list[ComparisonRow]
    test_status: TestStatus
    test_comparison: ComparisonResult | None
    review_list: ReviewList
    reason: str
    test_baseline: EvalResult | None = None
    test_winner: EvalResult | None = None
    test_error: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
