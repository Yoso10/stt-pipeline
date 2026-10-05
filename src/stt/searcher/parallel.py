"""Worker processes for parallel trials (standard library only).

How to run: this module is not a script; the searcher uses it when `search.parallel_trials` is above 1. Design (DECISIONS 49): the PARENT process owns the
strategy, the pruning rule and the study file; every trial runs in its own worker process (`spawn`, the only start method on Windows) and talks to the parent
through a pipe: each `report(metrics)` call of the runner is sent to the parent, which answers True (stop) or False. Therefore no file is shared between
processes, there is one writer of the study file, and `grid` and `optuna_tpe` behave the same. Each worker is told its own thread limit
(`Trial.threads`). The runner must be picklable. A worker that dies without a result is recorded as a failed trial.
"""

from __future__ import annotations

from collections.abc import Callable
from multiprocessing.connection import Connection

from stt.searcher.errors import SearchInterrupted
from stt.searcher.types import Trial, TrialOutcome, TrialRunner
from stt.trainercore import StepMetrics


def worker_main(conn: Connection, runner: TrialRunner, trial: Trial) -> None:
    """Run one trial in a worker process and send the result to the parent. Runs at the top level of the process (picklable by name)."""

    def report(metrics: StepMetrics) -> bool:
        conn.send(("report", metrics))
        return bool(conn.recv())

    try:
        outcome: TrialOutcome = runner(trial, report)
        conn.send(("done", outcome))
    except BaseException as exc:  # the parent records the message as a failed trial (or as an interrupt); the worker ends
        try:
            conn.send(("interrupted" if isinstance(exc, SearchInterrupted) else "error", f"{type(exc).__name__}: {exc}"))
        except (OSError, ValueError):
            pass
    finally:
        conn.close()


ReportFn = Callable[[StepMetrics], bool]
