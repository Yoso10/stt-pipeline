"""Selector: decide, from evaluated finalists and the baseline, which model wins, or that the baseline stays (C9).

How to run: from Python (PowerShell, project root; the full working example is in docs/COMPONENTS.md, C9):
    $env:PYTHONPATH = "src"
    python -c "from stt.selector import Selector, SelectConfig; print(Selector(SelectConfig()))"
`Selector(config).select(request) -> SelectionResult`. Val decides (a candidate is eligible when its Val comparison with the baseline reaches `select.min_verdict`;
among the eligible the lowest Val WER wins, a difference below `tie_epsilon` is a tie decided by the lower Val loss). No eligible candidate gives
`baseline_retained`, a legitimate outcome (exit code 0). The Test is evaluated ONCE, only for the winner and for the baseline, through the injected callable
(never when the baseline stays); a winner that the Test does not confirm is flagged, not dropped. The Selector reads no file, trains nothing, evaluates
nothing itself, uses no randomness, and imports no model library.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from stt.evaluator import ComparisonResult, EvaluatorError
from stt.selector.config import SelectConfig
from stt.selector.errors import SelectorError
from stt.selector.guard import TestOnceGuard
from stt.selector.types import Candidate, ComparisonRow, RankedCandidate, SelectionRequest, SelectionResult

logger = logging.getLogger("stt.selector")
LEVELS = {"worse": 0, "no_clear_difference": 1, "improved": 2}


@dataclass(frozen=True)
class Judgement:
    """A comparison read as a level: improved, no_clear_difference, worse, or unusable (not comparable, or rejected for too few groups), with a note."""

    level: str
    note: str


def _params_text(candidate: Candidate) -> str:
    return ", ".join(f"{k} {v:g}" if isinstance(v, int | float) else f"{k} {v}" for k, v in candidate.params.items()) or "no searched parameters"


class Selector:
    """Chooses the winner. One implementation (no interface)."""

    def __init__(self, config: SelectConfig) -> None:
        self._config = config

    # ---- judging ---------------------------------------------------------------------------------------------------
    def judge(self, comparison: ComparisonResult) -> Judgement:
        """Read a comparison: its verdict, or for `insufficient_groups` the configured policy (plain WER difference, flagged, or reject)."""
        verdict = comparison.verdict
        if verdict == "not_comparable":
            return Judgement("unusable", "the results are not comparable (different evaluation profiles)")
        if verdict == "insufficient_groups":
            if self._config.on_insufficient_groups == "reject":
                return Judgement("unusable", f"only {comparison.n_groups} group(s), too few for a confidence statement (select.on_insufficient_groups is reject)")
            delta = comparison.delta_wer or 0.0
            level = "improved" if delta > 0 else "worse" if delta < 0 else "no_clear_difference"
            return Judgement(level, f"no confidence statement: only {comparison.n_groups} group(s), so it is judged by the plain WER difference of {delta:+.4f}")
        return Judgement(verdict, comparison.explanation.rstrip("."))

    def _eligible(self, judgement: Judgement) -> bool:
        return judgement.level != "unusable" and LEVELS[judgement.level] >= LEVELS[self._config.min_verdict]

    # ---- the selection ---------------------------------------------------------------------------------------------------
    def select(self, request: SelectionRequest) -> SelectionResult:
        """Make the selection.

        Errors: SelectorError (exit code 4) for no candidates, duplicate trial ids, a candidate without a Val result, a Val result that carries the Test split name,
        results of another evaluation profile, results on different file sets, or a Test result that is not the Test split; TestAlreadyEvaluatedError when the Test is
        requested a second time for one model. "The baseline stays" is a result, not an error.
        """
        self._validate(request)
        baseline = request.baseline_val
        judged: list[tuple[Candidate, ComparisonResult, Judgement, bool]] = []
        for candidate in request.candidates:
            try:
                comparison = request.comparator.compare(baseline, candidate.val)
            except EvaluatorError as exc:
                raise SelectorError(f"Candidate trial {candidate.trial_id} cannot be compared with the baseline: {exc}") from exc
            judgement = self.judge(comparison)
            judged.append((candidate, comparison, judgement, self._eligible(judgement)))
        winner = self._choose([(c, e) for c, _, _, e in judged])
        ranking = self._ranking(judged, winner)
        warnings = [f"trial {r.candidate.trial_id}: {self.judge(r.comparison).note}" for r in ranking if r.comparison.verdict == "insufficient_groups"]
        table = [self._row(i, r, winner) for i, r in enumerate(ranking, 1)]
        if winner is None:
            return self._baseline_stays(request, ranking, table, tuple(warnings))
        return self._with_test(request, winner, ranking, table, tuple(warnings))

    def _validate(self, request: SelectionRequest) -> None:
        candidates = list(request.candidates)
        if not candidates:
            raise SelectorError("There are no candidates to select from: the search returned no finalist. Read the search report and its failure reasons.")
        ids = [c.trial_id for c in candidates]
        if len(set(ids)) != len(ids):
            raise SelectorError(f"Duplicate trial ids among the candidates: {sorted({i for i in ids if ids.count(i) > 1})}. Every finalist must have its own trial id.")
        for c in candidates:
            if c.val is None:
                raise SelectorError(f"Candidate trial {c.trial_id} has no Val evaluation. Evaluate every finalist on Val before the selection.")
        everything = [("the baseline", request.baseline_val), *((f"candidate trial {c.trial_id}", c.val) for c in candidates)]
        for name, result in everything:
            if result.split_name == request.test_split:
                raise SelectorError(f"The Val result of {name} carries the Test split name '{request.test_split}'. The choice is made on Val only; the Test is evaluated once, afterwards, for the winner.")
            if result.eval_version != request.baseline_val.eval_version:
                raise SelectorError(
                    f"The Val result of {name} uses eval_version {result.eval_version} but the baseline uses {request.baseline_val.eval_version}: the comparison would be invalid. "
                    "Re-evaluate the baseline with the current text profile."
                )

    @staticmethod
    def _key(c: Candidate) -> tuple[float, float, int]:
        return (c.val.wer, c.val_loss if math.isfinite(c.val_loss) else math.inf, c.trial_id)

    def _choose(self, items: list[tuple[Candidate, bool]]) -> Candidate | None:
        eligible = [c for c, e in items if e]
        if not eligible:
            return None
        best_wer = min(c.val.wer for c in eligible)
        contenders = [c for c in eligible if c.val.wer - best_wer < self._config.tie_epsilon]  # a difference below tie_epsilon is a tie
        return min(contenders, key=lambda c: (self._key(c)[1], c.val.wer, c.trial_id))  # the lower Val loss decides a tie

    def _ranking(self, judged: list[tuple[Candidate, ComparisonResult, Judgement, bool]], winner: Candidate | None) -> list[RankedCandidate]:
        ranked = []
        for candidate, comparison, judgement, eligible in judged:
            if candidate is winner:
                reason = f"winner: {judgement.note}"
                tied = [o for o, _, _, e in judged if e and o is not candidate and abs(o.val.wer - candidate.val.wer) < self._config.tie_epsilon]
                if tied:
                    reason += f" It was within the tie margin {self._config.tie_epsilon:g} of trial(s) {', '.join(str(o.trial_id) for o in tied)}, and the lower Val loss ({candidate.val_loss:.4f}) decided."
            elif eligible:
                reason = f"eligible but not the best: {judgement.note}"
            else:
                reason = f"not eligible ({self._config.min_verdict} needed): {judgement.note}"
            ranked.append(RankedCandidate(candidate, comparison.delta_wer, comparison, eligible, reason))
        winner_first = sorted(ranked, key=lambda r: (r.candidate is not winner, not r.eligible, *self._key(r.candidate)))
        return winner_first

    @staticmethod
    def _row(rank: int, r: RankedCandidate, winner: Candidate | None) -> ComparisonRow:
        c = r.candidate
        note = "WINNER" if c is winner else ("eligible" if r.eligible else "not eligible")
        if r.comparison.verdict == "insufficient_groups":
            note += ", no confidence statement"
        return ComparisonRow(rank, c.trial_id, dict(c.params), c.val.wer, c.val.cer, r.delta_wer_vs_baseline, r.comparison.ci_low, r.comparison.ci_high, r.comparison.verdict, c.val_loss, c.steps, r.eligible, note)

    # ---- outcomes -----------------------------------------------------------------------------------------------------------
    def _baseline_stays(self, request: SelectionRequest, ranking: list[RankedCandidate], table: list[ComparisonRow], warnings: tuple[str, ...]) -> SelectionResult:
        best = ranking[0]
        reason = (
            f"The baseline stays: none of the {len(ranking)} finalist(s) beat it on Val at the required level (select.min_verdict {self._config.min_verdict}). "
            f"The best candidate by Val WER was trial {best.candidate.trial_id} ({_params_text(best.candidate)}) with Val WER {best.candidate.val.wer:.4f} against {request.baseline_val.wer:.4f} for the baseline: "
            f"{self.judge(best.comparison).note}. This is a legitimate outcome, not a failure; the Test was not evaluated."
        )
        logger.info(reason)
        return SelectionResult("baseline_retained", None, ranking, table, "not_evaluated", None, request.baseline_val.review, reason, warnings=warnings)

    def _with_test(self, request: SelectionRequest, winner: Candidate, ranking: list[RankedCandidate], table: list[ComparisonRow], warnings: tuple[str, ...]) -> SelectionResult:
        top = ranking[0]
        base_val = request.baseline_val
        sentence = (
            f"Trial {winner.trial_id} ({_params_text(winner)}) is the winner: its Val WER is {winner.val.wer:.4f} against {base_val.wer:.4f} for the baseline "
            f"(improvement {top.delta_wer_vs_baseline or 0.0:+.4f}; {self.judge(top.comparison).note})."
        )
        if not self._config.test_confirmation:
            reason = sentence + " The Test confirmation is switched off (select.test_confirmation false)."
            return SelectionResult("winner", winner, ranking, table, "not_evaluated", None, winner.val.review, reason, warnings=warnings)
        guard = request.evaluate_test if isinstance(request.evaluate_test, TestOnceGuard) else TestOnceGuard(request.evaluate_test)
        try:
            test_winner = guard(winner)
            test_baseline = guard(None)
        except SelectorError:
            raise  # a second look at the Test is never swallowed
        except Exception as exc:  # the winner is kept; the failure is reported, not hidden
            error = f"{type(exc).__name__}: {exc}"
            logger.error("The Test evaluation failed: %s", error)
            reason = sentence + f" The Test evaluation FAILED ({error}), so the winner is not confirmed or contradicted (test_status not_evaluated)."
            return SelectionResult("winner", winner, ranking, table, "not_evaluated", None, winner.val.review, reason, test_error=error, warnings=warnings)
        for name, result in (("winner", test_winner), ("baseline", test_baseline)):
            if result.split_name != request.test_split or result.eval_version != base_val.eval_version:
                raise SelectorError(
                    f"The Test result of the {name} is on split '{result.split_name}' with eval_version {result.eval_version}, but the Test split is '{request.test_split}' "
                    f"and the evaluation profile is version {base_val.eval_version}. Fix the Test evaluation callable."
                )
        try:
            comparison = request.comparator.compare(test_baseline, test_winner)
        except EvaluatorError as exc:
            raise SelectorError(f"The Test results of the winner and the baseline cannot be compared: {exc}") from exc
        judgement = self.judge(comparison)
        confirmed = judgement.level == "improved"
        test_text = (
            f"Test WER {test_winner.wer:.4f} for the winner against {test_baseline.wer:.4f} for the baseline (improvement {comparison.delta_wer or 0.0:+.4f}; {judgement.note})."
        )
        if confirmed:
            reason = sentence + " The Test confirms it: " + test_text
        else:
            reason = (
                sentence + " The Test does NOT confirm it: " + test_text + " The winner stays the winner (owner rule) but is FLAGGED: read the Test numbers beside the Val numbers "
                "and listen to the review list."
            )
        return SelectionResult(
            "winner", winner, ranking, table, "confirmed" if confirmed else "not_confirmed", comparison, winner.val.review, reason, test_baseline=test_baseline, test_winner=test_winner, warnings=warnings
        )

