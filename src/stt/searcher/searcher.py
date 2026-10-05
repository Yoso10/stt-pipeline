"""HyperparamSearcher: propose parameter sets, hand each to a trial runner, record every outcome, return the ranked candidates (C8).

How to run: from Python (PowerShell, project root; the full working example is in docs/COMPONENTS.md, C8):
    $env:PYTHONPATH = "src"
    python -c "from stt.searcher import HyperparamSearcher; print(HyperparamSearcher)"
`HyperparamSearcher(registry).search(spec, run_trial) -> SearchResult`. The searcher never trains or evaluates: the trial runner (supplied by the Orchestrator) does.
The objective is the Val loss. Trials run one after another by default; `parallel_trials` above 1 runs worker processes (see parallel.py). The study is stored after
every finished trial in `spec.study_dir` (resume). The searcher never declares a winner (the Selector does).
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable
from multiprocessing import get_context
from multiprocessing.connection import wait

from stt.searcher.errors import SearchConfigError, SearchError, SearchInterrupted
from stt.searcher.parallel import worker_main
from stt.searcher.pruning import MedianRule
from stt.searcher.registry import StrategyRegistry, default_registry
from stt.searcher.study import StudyStore
from stt.searcher.types import (
    SearchReport,
    SearchResult,
    SearchSpec,
    SearchStrategy,
    StoppedReason,
    Trial,
    TrialOutcome,
    TrialRecord,
    TrialRunner,
)
from stt.trainercore import StepMetrics

logger = logging.getLogger("stt.searcher")
TIME_BUDGET = "time_budget"


class _Run:
    """The bookkeeping of one search (shared by the sequential and the parallel executor)."""

    def __init__(self, spec: SearchSpec, strategy: SearchStrategy, existing: list[TrialRecord], store: StudyStore, clock: Callable[[], float]) -> None:
        self.spec = spec
        self.strategy = strategy
        self.store = store
        self.clock = clock
        self.started_at = clock()
        self.records: list[TrialRecord] = list(existing)
        self.n_resumed = len(existing)
        self.next_id = max((r.trial.trial_id for r in existing), default=-1) + 1
        self.reason: StoppedReason | None = None
        self.threads = max(1, spec.total_threads // spec.parallel_trials)
        self.rule = MedianRule(spec.pruning.min_evals)
        self.running: dict[int, dict[str, object]] = {}
        self.started = len(existing)

    # ---- starting trials ---------------------------------------------------------------------------------------------
    def _budget_over(self) -> bool:
        minutes = self.spec.time_budget_minutes
        return minutes is not None and (self.clock() - self.started_at) >= minutes * 60.0

    def _too_many_failures(self) -> bool:
        finished = [r for r in self.records if r.outcome.error != TIME_BUDGET]
        failed = sum(r.outcome.status == "failed" for r in finished)
        return len(finished) >= self.spec.min_trials_before_abort and failed / len(finished) > self.spec.max_failed_ratio

    def next_trial(self) -> Trial | None:
        """The next trial to start, or None (and `reason` set) when the search must not start another one."""
        if self.reason is not None:
            return None
        if self.started >= self.spec.n_trials:
            self.reason = "n_trials"
        elif self._budget_over():
            self.reason = "time_budget"
        elif self._too_many_failures():
            self.reason = "too_many_failures"
        if self.reason is not None:
            return None
        suggestion = self.strategy.ask(self.next_id)
        if suggestion is None:
            self.reason = "space_exhausted"
            return None
        trial = Trial(self.next_id, dict(suggestion.params), self.threads)
        self.running[trial.trial_id] = {"history": [], "budget_stopped": False, "pruned": False}
        self.next_id += 1
        self.started += 1
        return trial

    # ---- during and after a trial -------------------------------------------------------------------------------------
    def on_report(self, trial_id: int, metrics: StepMetrics) -> bool:
        """The `report` callback of a trial: True when the trial must stop (time budget over, or pruned)."""
        state = self.running[trial_id]
        if self._budget_over():
            state["budget_stopped"] = True
            return True
        loss = metrics.val_loss
        if loss is None or not math.isfinite(loss):
            return False
        history: list[tuple[int, float]] = state["history"]  # type: ignore[assignment]
        history.append((metrics.step, float(loss)))
        self.strategy.record_eval(trial_id, metrics.step, float(loss))
        if self.spec.pruning.enabled:
            others = [r.outcome.reports for r in self.records if r.outcome.status in ("complete", "pruned") and r.outcome.reports]
            if self.rule.should_prune(history, others):
                state["pruned"] = True
                return True
        return False

    def on_finish(self, trial: Trial, outcome: TrialOutcome) -> None:
        """Record the outcome of a trial (classified by the searcher), tell the strategy, and store the study."""
        state = self.running.pop(trial.trial_id)
        history: list[tuple[int, float]] = state["history"]  # type: ignore[assignment]
        reports = tuple(history)
        if state["budget_stopped"]:
            final = TrialOutcome("failed", None, outcome.steps, outcome.seconds, None, TIME_BUDGET, reports)  # never counted as a result
        elif outcome.status == "failed":
            final = TrialOutcome("failed", None, outcome.steps, outcome.seconds, None, outcome.error or "the trial failed", reports)
        elif state["pruned"]:
            final = TrialOutcome("pruned", history[-1][1] if history else outcome.val_loss, outcome.steps, outcome.seconds, outcome.model_dir, None, reports)
        elif outcome.status == "complete" and outcome.val_loss is None:
            final = TrialOutcome("failed", None, outcome.steps, outcome.seconds, None, "the trial completed without a Val loss", reports)
        else:
            final = TrialOutcome(outcome.status, outcome.val_loss, outcome.steps, outcome.seconds, outcome.model_dir, outcome.error, reports)
        self.records.append(TrialRecord(trial, final))
        self.strategy.tell(trial.trial_id, trial.params, final)
        if self.spec.study_dir is not None:
            self.store.save(self.spec.study_dir, self.spec, self.records)
        logger.info("Trial %d %s %s (Val loss %s).", trial.trial_id, final.status, dict(trial.params), "-" if final.val_loss is None else f"{final.val_loss:.4f}")


class HyperparamSearcher:
    """Runs a search. The strategies come from a registry; `clock` and `cores` are injectable for tests."""

    def __init__(self, registry: StrategyRegistry | None = None, clock: Callable[[], float] = time.monotonic, cores: Callable[[], int | None] = os.cpu_count) -> None:
        self._registry = registry or default_registry()
        self._clock = clock
        self._cores = cores

    def search(self, spec: SearchSpec, run_trial: TrialRunner) -> SearchResult:
        """Run the search and return the result.

        Errors: SearchConfigError (exit code 2) for an unknown strategy, a missing Optuna package, `parallel_trials` above the number of cores, or a stored study
        with another space or seed; SearchError (exit code 4) when no trial completed (the failure reasons are summarized, or the advice to relax pruning) or the
        study file cannot be written. The SearchError carries the result made so far.
        """
        cores = self._cores() or 1
        if spec.parallel_trials > cores:
            raise SearchConfigError(f"search.parallel_trials is {spec.parallel_trials} but this machine has {cores} cores. Lower search.parallel_trials to at most {cores}.")
        strategy = self._registry.create(spec)  # an unknown name or a missing package fails here
        store = StudyStore()
        existing = store.load(spec.study_dir, spec) if spec.study_dir is not None else []
        strategy.replay(existing)
        run = _Run(spec, strategy, existing, store, self._clock)
        if spec.parallel_trials == 1:
            self._sequential(run, run_trial)
        else:
            self._parallel(run, run_trial)
        result = self._result(run)
        if result.report.n_complete == 0:
            raise SearchError(self._no_result_message(run, result), result)
        return result

    # ---- executors -----------------------------------------------------------------------------------------------------
    @staticmethod
    def _sequential(run: _Run, run_trial: TrialRunner) -> None:
        while (trial := run.next_trial()) is not None:
            started = time.monotonic()
            try:
                outcome = run_trial(trial, lambda m, t=trial: run.on_report(t.trial_id, m))
            except SearchInterrupted:
                raise  # not a result and not a failure: the study keeps the finished trials, a rerun continues
            except Exception as exc:  # a failing trial is recorded and the search goes on
                outcome = TrialOutcome("failed", None, 0, time.monotonic() - started, None, f"{type(exc).__name__}: {exc}")
            run.on_finish(trial, outcome)

    @staticmethod
    def _parallel(run: _Run, run_trial: TrialRunner) -> None:
        context = get_context("spawn")  # the only start method on Windows; the runner must be picklable
        running: dict[object, tuple[Trial, object]] = {}
        interrupted = False
        try:
            while True:
                while not interrupted and len(running) < run.spec.parallel_trials and (trial := run.next_trial()) is not None:
                    parent, child = context.Pipe()
                    process = context.Process(target=worker_main, args=(child, run_trial, trial), daemon=True)
                    process.start()
                    child.close()
                    running[parent] = (trial, process)
                if not running:
                    break
                for conn in wait(list(running), timeout=0.2):  # type: ignore[arg-type]
                    trial, process = running[conn]
                    try:
                        kind, payload = conn.recv()  # type: ignore[attr-defined]
                    except (EOFError, OSError):
                        kind, payload = "error", "the worker process ended without a result"
                    if kind == "report":
                        conn.send(run.on_report(trial.trial_id, payload))  # type: ignore[attr-defined]
                        continue
                    if kind == "interrupted":
                        interrupted = True  # no new trials; the running ones finish; the interrupted one is not recorded
                        run.running.pop(trial.trial_id, None)
                    else:
                        outcome = payload if kind == "done" else TrialOutcome("failed", None, 0, 0.0, None, str(payload))
                        run.on_finish(trial, outcome)
                    conn.close()  # type: ignore[attr-defined]
                    process.join(timeout=10)  # type: ignore[attr-defined]
                    del running[conn]
        finally:
            for _, process in running.values():
                process.terminate()  # type: ignore[attr-defined]
        if interrupted:
            raise SearchInterrupted("a trial was interrupted on request")

    # ---- result ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _result(run: _Run) -> SearchResult:
        spec = run.spec
        complete = sorted((r for r in run.records if r.outcome.status == "complete" and r.outcome.val_loss is not None), key=lambda r: (r.outcome.val_loss, r.trial.trial_id))  # type: ignore[arg-type,return-value]
        best = complete[0] if complete else None
        notes = []
        if spec.parallel_trials > 1 or spec.pruning.enabled:
            notes.append("with pruning or parallel trials the exact sequence of trials can vary between runs with the same seed")
        if run.n_resumed:
            notes.append(f"{run.n_resumed} trial(s) came from an earlier run of this study; a resumed search does not repeat the exact sequence of an uninterrupted one")
        if best is not None:
            for name, r in spec.parameters.items():
                value = best.trial.params.get(name)
                if r.kind != "categorical" and r.low is not None and r.high is not None and isinstance(value, int | float) and (math.isclose(value, r.low, rel_tol=1e-6) or math.isclose(value, r.high, rel_tol=1e-6)):
                    notes.append(f"the best trial is at the edge of the range of '{name}' ({value:g}): consider a wider range in the next experiment")
        report = SearchReport(
            strategy=spec.strategy,
            seed=spec.seed,
            n_started=len(run.records),
            n_complete=len(complete),
            n_pruned=sum(r.outcome.status == "pruned" for r in run.records),
            n_failed=sum(r.outcome.status == "failed" for r in run.records),
            n_resumed=run.n_resumed,
            best_params=dict(best.trial.params) if best else None,
            best_val_loss=best.outcome.val_loss if best else None,
            seconds=run.clock() - run.started_at,
            parallel_trials=spec.parallel_trials,
            threads_per_trial=run.threads,
            notes=tuple(notes),
        )
        return SearchResult(list(run.records), complete[: spec.finalists_k], run.reason or "n_trials", report)

    @staticmethod
    def _no_result_message(run: _Run, result: SearchResult) -> str:
        pruned = result.report.n_pruned
        failures = [f"trial {r.trial.trial_id}: {r.outcome.error}" for r in run.records if r.outcome.status == "failed"]
        if pruned and not failures:
            return f"All {pruned} trial(s) were pruned and none completed. Relax the pruning: raise search.pruning.min_evals or set search.pruning.enabled to false."
        if not run.records:
            return "No trial ran (the budget or the space allowed none). Raise search.n_trials or search.time_budget_minutes."
        return (
            f"No trial completed ({result.report.n_failed} failed, {pruned} pruned; stopped because of {result.stopped_reason}). Reasons: " + "; ".join(failures[:5]) + ". "
            "A common cause is a learning rate range that is too high: lower the upper bound of search.parameters.learning_rate."
        )
