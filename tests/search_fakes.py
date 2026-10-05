"""Fake trial runners for the tests of the HyperparamSearcher (module-level classes so that they can be sent to worker processes).

How to run: this module is not a script; tests/test_searcher.py imports it. The objective is synthetic: a smooth bowl in (log learning rate, log2 accumulation).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from pathlib import Path

from stt.searcher import Trial, TrialOutcome
from stt.trainercore import StepMetrics


def bowl(params: dict[str, float | int | str], best_lr: float = 2.2e-5, best_acc: int = 4) -> float:
    """The synthetic Val loss: 0.5 at the best point, growing with the distance in log learning rate and in log2 accumulation."""
    lr = float(params.get("learning_rate", best_lr))
    acc = float(params.get("grad_accum_steps", best_acc))
    return 0.5 + (math.log10(lr) - math.log10(best_lr)) ** 2 + 0.05 * (math.log2(acc) - math.log2(best_acc)) ** 2


def metrics(step: int, loss: float | None) -> StepMetrics:
    return StepMetrics(step, 0.0, loss, loss, 1e-5, None, 0.1, 0.0, None)


class BowlRunner:
    """Reports four decreasing Val losses and returns the bowl value; stops early (status pruned) when `report` says so."""

    def __init__(self, best_lr: float = 2.2e-5, offset: dict[int, float] | None = None) -> None:
        self.best_lr = best_lr
        self.offset = offset or {}

    def __call__(self, trial: Trial, report: Callable[[StepMetrics], bool]) -> TrialOutcome:
        final = bowl(dict(trial.params), self.best_lr) + self.offset.get(trial.trial_id, 0.0)
        seen = []
        for i, step in enumerate((50, 100, 150, 200), 1):
            loss = final * (1.0 + 1.0 / i)  # the loss falls towards `final`
            seen.append(step)
            if report(metrics(step, loss)):
                return TrialOutcome("pruned", loss, step, 0.1)
        return TrialOutcome("complete", final, 200, 0.1, Path(f"model_{trial.trial_id}"))


class ScriptedRunner:
    """Behaviour by trial id: 'ok', 'raise', 'fail' (returns failed), 'nan' (a NaN failure message)."""

    def __init__(self, script: dict[int, str]) -> None:
        self.script = script

    def __call__(self, trial: Trial, report: Callable[[StepMetrics], bool]) -> TrialOutcome:
        action = self.script.get(trial.trial_id, "ok")
        if action == "raise":
            raise RuntimeError(f"boom in trial {trial.trial_id}")
        if action == "fail":
            return TrialOutcome("failed", None, 3, 0.1, None, "the loss became NaN at step 3")
        report(metrics(50, 1.0))
        return TrialOutcome("complete", bowl(dict(trial.params)), 50, 0.1)


class RecordingRunner:
    """Appends one JSON line per trial (trial id, process id, thread limit) to a file, then behaves like the bowl; used to verify parallel mode."""

    def __init__(self, path: str, sleep: float = 0.0) -> None:
        self.path = path
        self.sleep = sleep

    def __call__(self, trial: Trial, report: Callable[[StepMetrics], bool]) -> TrialOutcome:
        import time

        time.sleep(self.sleep)
        stop = report(metrics(50, bowl(dict(trial.params))))
        with open(self.path, "a", encoding="utf-8") as handle:  # one short append per trial: safe enough for the test
            handle.write(json.dumps({"trial": trial.trial_id, "pid": os.getpid(), "threads": trial.threads, "stop": stop}) + "\n")
        return TrialOutcome("complete", bowl(dict(trial.params)), 50, 0.1)


class DyingRunner:
    """Kills its own process without sending a result (a crashed worker)."""

    def __call__(self, trial: Trial, report: Callable[[StepMetrics], bool]) -> TrialOutcome:
        os._exit(3)


class ClockRunner:
    """Advances a shared fake clock (a one-element list) by `step` seconds at every report; used for the time-budget tests."""

    def __init__(self, clock: list[float], step: float, reports: int = 4) -> None:
        self.clock = clock
        self.step = step
        self.reports = reports

    def __call__(self, trial: Trial, report: Callable[[StepMetrics], bool]) -> TrialOutcome:
        for i in range(1, self.reports + 1):
            self.clock[0] += self.step
            if report(metrics(i * 10, 1.0 / i)):
                return TrialOutcome("complete", 0.25, i * 10, self.step * i)  # a runner that stops when told, reporting what it has
        return TrialOutcome("complete", bowl(dict(trial.params)), self.reports * 10, self.step * self.reports)


class InterruptingRunner:
    """Raises SearchInterrupted at trial `at` (the user pressed Ctrl+C); the other trials behave like the bowl."""

    def __init__(self, at: int) -> None:
        self.at = at

    def __call__(self, trial: Trial, report: Callable[[StepMetrics], bool]) -> TrialOutcome:
        from stt.searcher import SearchInterrupted

        if trial.trial_id == self.at:
            raise SearchInterrupted("stop requested")
        report(metrics(50, bowl(dict(trial.params))))
        return TrialOutcome("complete", bowl(dict(trial.params)), 50, 0.1)
