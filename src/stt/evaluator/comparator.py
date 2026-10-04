"""Comparator: compare two scored results with a significance test over groups (C7).

How to run: from Python (PowerShell, project root; the full working example is in docs/COMPONENTS.md, C7):
    $env:PYTHONPATH = "src"
    python -c "from stt.evaluator import Comparator, EvalConfig; print(Comparator(EvalConfig(), 42))"
`Comparator(config, seed).compare(baseline, candidate) -> ComparisonResult`. The verdict is worded as a level of evidence, not as proof.
Rules: results with a different `eval_version` are `not_comparable` (no numbers); different file sets FAIL; fewer groups than
`eval.significance.min_groups` give `insufficient_groups` with the plain WER difference and no interval.
"""

from __future__ import annotations

from collections import defaultdict

from stt.evaluator.config import EvalConfig
from stt.evaluator.errors import EvaluatorError
from stt.evaluator.registry import SignificanceRegistry, default_significance_registry
from stt.evaluator.types import ComparisonResult, EvalResult, GroupErrors


class Comparator:
    """Compares a candidate with a baseline. The significance test is selected by `eval.significance.method`."""

    def __init__(self, config: EvalConfig, seed: int, registry: SignificanceRegistry | None = None) -> None:
        self._config = config.significance
        self._seed = seed
        self._test = (registry or default_significance_registry()).create(config.significance)  # an unknown method fails here

    def compare(self, baseline: EvalResult, candidate: EvalResult) -> ComparisonResult:
        """Compare the two results. delta_wer is the baseline minus the candidate (positive = improvement).

        Errors: EvaluatorError when the two results were not scored on the same files.
        """
        c = self._config
        if baseline.eval_version != candidate.eval_version:
            return ComparisonResult(
                None, None, None, c.confidence, c.one_sided, 0, False, "not_comparable",
                f"The results use different eval profiles (version {baseline.eval_version} and {candidate.eval_version}); "
                "re-evaluate the baseline with the current profile.",
            )
        base_files = {f.sample_id: f for f in baseline.files}
        cand_files = {f.sample_id: f for f in candidate.files}
        if base_files.keys() != cand_files.keys():
            only_base, only_cand = sorted(base_files.keys() - cand_files.keys()), sorted(cand_files.keys() - base_files.keys())
            raise EvaluatorError(
                f"The two results were not scored on the same files: {len(only_base)} only in the baseline (for example {only_base[:3]}), "
                f"{len(only_cand)} only in the candidate (for example {only_cand[:3]}). Evaluate both on the same split."
            )
        per_group: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        for sample_id, b in base_files.items():
            row = per_group[b.group_id]
            row[0] += b.word_errors
            row[1] += cand_files[sample_id].word_errors
            row[2] += b.ref_words
        groups = [GroupErrors(*row) for _, row in sorted(per_group.items())]
        words = sum(g.words for g in groups)
        delta = (sum(g.baseline_errors for g in groups) - sum(g.candidate_errors for g in groups)) / max(1, words)
        if len(groups) < c.min_groups:
            return ComparisonResult(
                delta, None, None, c.confidence, c.one_sided, len(groups), delta > c.min_effect_abs, "insufficient_groups",
                f"Only {len(groups)} groups (the minimum is {c.min_groups}), so no statement about confidence is made; "
                f"the plain WER difference is {delta:+.4f} (positive = the candidate is better).",
            )
        low, high = self._test.interval(groups, self._seed)
        level = f"{c.confidence:.0%}" + (" one-sided" if c.one_sided else "")
        if low > c.min_effect_abs:
            verdict, text = "improved", f"The candidate is better than the baseline at the {level} level (WER difference {delta:+.4f}, lower bound {low:+.4f}, {len(groups)} groups)."
        elif high < -c.min_effect_abs:
            verdict, text = "worse", f"The candidate is worse than the baseline at the {level} level (WER difference {delta:+.4f}, upper bound {high:+.4f}, {len(groups)} groups)."
        else:
            verdict, text = "no_clear_difference", f"No clear difference at the {level} level (WER difference {delta:+.4f}, bounds {low:+.4f} to {high:+.4f}, {len(groups)} groups)."
        return ComparisonResult(delta, low, high, c.confidence, c.one_sided, len(groups), low > c.min_effect_abs, verdict, text)  # type: ignore[arg-type]
