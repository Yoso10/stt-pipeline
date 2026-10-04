"""EvalMonitor: tracks the best Val loss and decides early stopping (pure Python, no torch).

How to run: this module is not a script; the Hugging Face adapter calls it after every evaluation. Example
(PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.trainercore.monitor import EvalMonitor; from stt.trainercore.types import RunState; m = EvalMonitor(2, 0.0, RunState()); print([m.record(s, v).stop for s, v in [(50, 1.0), (100, 1.1), (150, 1.2)]])"
Example output:
    [False, False, True]
An evaluation IMPROVES when its loss is lower than the best so far by MORE than `min_delta`; `patience` evaluations in a row
without improvement stop the run. The first evaluation always improves (there is no best yet).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from stt.trainercore.types import RunState


@dataclass(frozen=True)
class EvalVerdict:
    """The result of one evaluation: did it improve, must the run stop, and is it the "never improved after the first" case."""

    improved: bool
    stop: bool
    never_improved_after_first: bool


class EvalMonitor:
    """Best-loss and patience bookkeeping on a `RunState` (which is saved in every checkpoint)."""

    def __init__(self, patience: int, min_delta: float, state: RunState) -> None:
        self._patience = patience
        self._min_delta = min_delta
        self.state = state

    def record(self, step: int, val_loss: float) -> EvalVerdict:
        """Register the Val loss of `step` (must be finite) and return the verdict."""
        if not math.isfinite(val_loss):
            raise ValueError(f"the Val loss at step {step} is not finite")
        state = self.state
        state.evals_done += 1
        if state.first_eval_step is None:
            state.first_eval_step = step
        improved = state.best_val_loss is None or val_loss < state.best_val_loss - self._min_delta
        if improved:
            state.best_val_loss = val_loss
            state.best_step = step
            state.bad_evals = 0
        else:
            state.bad_evals += 1
        stop = state.bad_evals >= self._patience
        return EvalVerdict(improved, stop, stop and state.best_step == state.first_eval_step)
