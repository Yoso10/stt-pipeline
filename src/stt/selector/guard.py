"""TestOnceGuard: the Test evaluation can be asked for at most once per model (standard library only).

How to run: this module is not a script; the Orchestrator wraps its Test evaluation in it ONCE per experiment and the Selector uses it. Example
(PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.selector.guard import TestOnceGuard; g = TestOnceGuard(lambda c: 'result'); print(g(None)); print(g.calls)"
Example output:
    result
    ['baseline']
A second call for the same model (`None` is the baseline, a Candidate is its trial) raises TestAlreadyEvaluatedError, also when the first call failed: the
Test set was already looked at. This guards against tuning on the Test (owner rule, O68). Nothing resets the guard; make a new one for a new experiment.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from stt.selector.errors import TestAlreadyEvaluatedError
from stt.selector.types import Candidate

__test__ = False  # not a unittest module, in spite of the name


class TestOnceGuard:
    """Wraps the one-time Test evaluation callable."""

    __test__ = False  # pytest-style collectors must not treat it as a test class

    def __init__(self, evaluate_test: Callable[[Candidate | None], Any]) -> None:
        self._evaluate = evaluate_test
        self._used: list[str] = []

    @property
    def calls(self) -> list[str]:
        """The models for which the Test was requested, in order (`baseline` or `trial_<id>`)."""
        return list(self._used)

    def __call__(self, candidate: Candidate | None) -> Any:
        key = "baseline" if candidate is None else f"trial_{candidate.trial_id}"
        if key in self._used:
            raise TestAlreadyEvaluatedError(
                f"The Test split was already evaluated for '{key}'. It is evaluated ONCE per model, as the final unbiased confirmation; "
                "a second evaluation would let a choice be tuned on the Test. Fix the caller; never bypass this guard."
            )
        self._used.append(key)  # counted before the call: a failed Test evaluation also counts as a look at the Test
        return self._evaluate(candidate)
