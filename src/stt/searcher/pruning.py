"""The pruning rule (pure Python, the same for every strategy).

How to run: this module is not a script; the searcher uses it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.searcher.pruning import MedianRule; r = MedianRule(min_evals=2); print(r.should_prune([(50, 3.0), (100, 2.9)], [[(50, 2.0), (100, 1.9)], [(50, 2.2), (100, 2.0)]]))"
Example output:
    True
MEDIAN RULE (the pruner of the owner-approved design, DECISIONS 49): a trial is stopped when, at its n-th Val evaluation (n at least `min_evals`), its Val loss is
WORSE than the median of the n-th Val losses of the other finished trials that had at least n evaluations, and at least `n_startup` such trials exist (the first
trials are never pruned). It needs no library, so `grid` and `optuna_tpe` prune the same way. With few evaluations per trial it rarely fires: it is meant for
trials that evaluate four times or more.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

N_STARTUP = 2


class MedianRule:
    """Prune a trial whose n-th Val loss is above the median of the others' n-th Val losses."""

    def __init__(self, min_evals: int, n_startup: int = N_STARTUP) -> None:
        self._min_evals = min_evals
        self._n_startup = n_startup

    def should_prune(self, history: Sequence[tuple[int, float]], others: Sequence[Sequence[tuple[int, float]]]) -> bool:
        """`history` = (step, Val loss) of the running trial so far; `others` = the histories of the finished trials."""
        n = len(history)
        if n < self._min_evals:
            return False
        peers = [h[n - 1][1] for h in others if len(h) >= n]
        if len(peers) < self._n_startup:
            return False
        return history[-1][1] > statistics.median(peers)
