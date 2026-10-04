"""Paired bootstrap over GROUPS: the significance test of the Comparator (numpy only).

How to run: this module is not a script; the Comparator uses it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.evaluator.bootstrap import PairedGroupBootstrap; from stt.evaluator.types import GroupErrors; g = [GroupErrors(10, 5, 50)] * 6; print(PairedGroupBootstrap(0.9, True, 200).interval(g, 42))"
Example output:
    (0.1, 0.1)
Files of one group (for example one lesson) are not independent, so whole groups are resampled with replacement, the same groups for
both models (paired). The statistic of each resample is delta_wer = (baseline errors - candidate errors) / reference words.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from stt.evaluator.types import GroupErrors


def resample_deltas(groups: Sequence[GroupErrors], n_resamples: int, seed: int) -> np.ndarray:
    """The delta_wer of `n_resamples` resamples of the groups (deterministic for a fixed seed)."""
    baseline = np.array([g.baseline_errors for g in groups], dtype=np.int64)
    candidate = np.array([g.candidate_errors for g in groups], dtype=np.int64)
    words = np.array([g.words for g in groups], dtype=np.int64)
    picks = np.random.default_rng(seed).integers(0, len(groups), size=(n_resamples, len(groups)))
    total_words = words[picks].sum(axis=1)
    delta = (baseline[picks].sum(axis=1) - candidate[picks].sum(axis=1)) / np.maximum(total_words, 1)
    return np.asarray(delta, dtype=np.float64)


class PairedGroupBootstrap:
    """`paired_group_bootstrap`: the (low, high) bounds of delta_wer at `confidence` (one-sided bounds, or a two-sided interval)."""

    def __init__(self, confidence: float, one_sided: bool, n_resamples: int) -> None:
        self._confidence = confidence
        self._one_sided = one_sided
        self._n = n_resamples

    def interval(self, groups: Sequence[GroupErrors], seed: int) -> tuple[float, float]:
        """Return (low, high). One-sided: the (1-confidence) and the confidence percentiles. Two-sided: the central interval."""
        deltas = resample_deltas(groups, self._n, seed)
        tail = (1.0 - self._confidence) if self._one_sided else (1.0 - self._confidence) / 2.0
        low, high = np.percentile(deltas, [100.0 * tail, 100.0 * (1.0 - tail)])
        return float(low), float(high)
