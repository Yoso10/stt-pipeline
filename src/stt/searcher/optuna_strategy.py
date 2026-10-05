"""The `optuna_tpe` strategy: the ONLY module of the component that imports Optuna (an adapter).

How to run: this module is not a script; the registry imports it only when `search.strategy` is `optuna_tpe`. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.searcher.registry import default_registry; print(default_registry().names())"
Example output:
    ['grid', 'optuna_tpe']
Optuna is used ONLY for its TPE sampler through the ask-and-tell interface, with an in-memory study and a fixed seed. The study that is stored on disk for a
resume is the searcher's own `study.json` (so no Optuna storage, no database and no file locking is involved): on a resume the earlier trials are
replayed into the in-memory study with `add_trial`. The pruning decision is the searcher's own median rule (`pruning.py`), so Optuna's pruners are not used.
The first `N_STARTUP_TRIALS` trials are random (TPE needs some data); a resumed search continues with the same sampler seed but not with the exact sequence
of an uninterrupted run (the report says so).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import optuna
from optuna.distributions import BaseDistribution, CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.trial import TrialState, create_trial

from stt.searcher.types import ParamRange, ParamValue, SearchSpec, Suggestion, TrialOutcome, TrialRecord

N_STARTUP_TRIALS = 3


def _distribution(r: ParamRange) -> BaseDistribution:
    if r.kind == "categorical":
        return CategoricalDistribution(list(r.choices))
    if r.kind == "int":
        return IntDistribution(int(r.low or 0), int(r.high or 0))
    return FloatDistribution(float(r.low or 0.0), float(r.high or 0.0), log=r.kind == "log_float")


class OptunaTpeStrategy:
    """TPE sampler by ask and tell."""

    name = "optuna_tpe"

    def __init__(self, spec: SearchSpec) -> None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self._distributions = {name: _distribution(r) for name, r in spec.parameters.items()}
        sampler = optuna.samplers.TPESampler(seed=spec.seed, n_startup_trials=N_STARTUP_TRIALS)
        self._study = optuna.create_study(direction="minimize", sampler=sampler, pruner=optuna.pruners.NopPruner())
        self._open: dict[int, Any] = {}

    def ask(self, trial_id: int) -> Suggestion:
        """The next parameter set proposed by the sampler."""
        trial = self._study.ask(self._distributions)
        self._open[trial_id] = trial
        return Suggestion({name: _plain(value) for name, value in trial.params.items()})

    def record_eval(self, trial_id: int, step: int, val_loss: float) -> None:
        """Pass an intermediate Val loss to Optuna (the sampler may use it for a pruned trial)."""
        trial = self._open.get(trial_id)
        if trial is not None:
            try:
                trial.report(val_loss, step)
            except (ValueError, RuntimeError):
                pass  # a repeated step is not an error for the search

    def tell(self, trial_id: int, params: Mapping[str, ParamValue], outcome: TrialOutcome) -> None:
        """Tell Optuna how the trial ended."""
        trial = self._open.pop(trial_id, None)
        if trial is None:
            return
        if outcome.status == "complete" and outcome.val_loss is not None:
            self._study.tell(trial, outcome.val_loss)
        elif outcome.status == "pruned":
            self._study.tell(trial, state=TrialState.PRUNED)
        else:
            self._study.tell(trial, state=TrialState.FAIL)

    def replay(self, records: list[TrialRecord]) -> None:
        """Teach the in-memory study the trials of an earlier part of the stored study."""
        for r in records:
            state = {"complete": TrialState.COMPLETE, "pruned": TrialState.PRUNED}.get(r.outcome.status, TrialState.FAIL)
            value = r.outcome.val_loss if state == TrialState.COMPLETE else None
            if state == TrialState.PRUNED and r.outcome.reports:
                value = r.outcome.reports[-1][1]
            self._study.add_trial(create_trial(state=state, value=value, params=dict(r.trial.params), distributions=self._distributions))


def _plain(value: Any) -> ParamValue:
    return value.item() if hasattr(value, "item") else value  # type: ignore[no-any-return]
