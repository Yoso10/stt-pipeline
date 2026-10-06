"""The one-time Test look of an experiment, kept on disk (standard library only; DECISIONS 57, O68).

How to run: this module is not a script; the select_and_register stage uses it. Example (PowerShell, project root; a throw-away experiment folder):
    $env:PYTHONPATH = "src"
    python -c "from stt.orchestrator.testlook import look_key; print(look_key(None))"
Example output:
    baseline
Rule (owner, O68): the Test split is evaluated ONCE per model, as the final unbiased confirmation. The in-process guard (`TestOnceGuard`) cannot know about an earlier process, so
the look is also written to the experiment: `reports/test_once.json` lists the models whose Test evaluation was STARTED (written BEFORE the evaluation, so a crash also counts as a
look) and `reports/test_result_<model>.json` holds the finished result (the lossless codec). `once_per_experiment(store, evaluate)` wraps the real Test evaluation:
a stored result is RETURNED (looking at a stored result is not a new look at the Test), a started-but-unfinished look FAILS with `TestAlreadyEvaluatedError` (the Test was already
looked at; start a new experiment), otherwise the look is marked, evaluated, stored and returned. Nothing here deletes or resets a look.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from stt.evaluator import EvalResult, eval_result_from_json, eval_result_to_json
from stt.runmanager import ExperimentHandle
from stt.selector import Candidate, TestAlreadyEvaluatedError

LOOKED_FILE = "test_once.json"


def look_key(candidate: Candidate | None) -> str:
    """`baseline` or `trial_<id>`: the same keys as the `TestOnceGuard`."""
    return "baseline" if candidate is None else f"trial_{candidate.trial_id}"


def result_file(key: str) -> str:
    """The name (inside `reports/`) of the stored Test result of a model."""
    return f"test_result_{key}.json"


class TestLookStore:
    """Reads and writes the look records of one experiment."""

    __test__ = False  # not a unittest class, in spite of the name

    def __init__(self, experiment: ExperimentHandle) -> None:
        self._experiment = experiment

    def looked(self) -> list[str]:
        """The models whose Test evaluation was started in this experiment."""
        path = self._experiment.paths.reports / LOOKED_FILE
        try:
            return list(json.loads(path.read_text(encoding="utf-8"))["looked"])
        except (OSError, ValueError, KeyError, TypeError):
            return []

    def mark(self, key: str) -> None:
        """Record that the Test evaluation of `key` starts now (before it runs)."""
        self._experiment.write_report(LOOKED_FILE, json.dumps({"looked": [*self.looked(), key]}, indent=1))

    def result(self, key: str) -> EvalResult | None:
        """The stored Test result of `key`, or None. Errors: EvaluatorError when the stored text is damaged."""
        path = self._experiment.paths.reports / result_file(key)
        return eval_result_from_json(path.read_text(encoding="utf-8")) if path.is_file() else None

    def save(self, key: str, result: EvalResult) -> None:
        """Store the finished Test result of `key`."""
        self._experiment.write_report(result_file(key), eval_result_to_json(result))


def once_per_experiment(store: TestLookStore, evaluate: Callable[[Candidate | None], EvalResult]) -> Callable[[Candidate | None], EvalResult]:
    """The Test callable for the Selector: see the module docstring. Errors: TestAlreadyEvaluatedError for a started look without a stored result."""

    def look(candidate: Candidate | None) -> EvalResult:
        key = look_key(candidate)
        stored = store.result(key)
        if stored is not None:
            return stored
        if key in store.looked():
            raise TestAlreadyEvaluatedError(
                f"The Test split was already looked at for '{key}' in this experiment (reports/{LOOKED_FILE}) but no result was stored, so the evaluation did not finish. "
                "It is evaluated ONCE per model; a second look would let a choice be tuned on the Test. Start a new experiment, and do not delete the record."
            )
        store.mark(key)
        result = evaluate(candidate)
        store.save(key, result)
        return result

    return look
