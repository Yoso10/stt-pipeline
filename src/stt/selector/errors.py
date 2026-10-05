"""Errors of the Selector component.

How to run: this module is not a script; it is imported by the other modules of stt.selector.
"""

from __future__ import annotations


class SelectorError(Exception):
    """The selection cannot be made (no candidates, results that cannot be compared, a split that is not the Val split, a misused Test callable).

    The Orchestrator maps this error to exit code 4 (failure in a selection stage). "The baseline stays" is NOT an error: it is a result with exit code 0.
    """


class TestAlreadyEvaluatedError(SelectorError):
    """The Test evaluation was requested a second time for the same model. It always fails (it guards against tuning on the Test); never bypass it."""

    __test__ = False  # not a unittest test case, in spite of the name
