"""Errors of the Evaluator component.

How to run: this module is not a script; it is imported by the other modules of stt.evaluator.
"""

from __future__ import annotations


class EvaluatorError(Exception):
    """An evaluation or a comparison cannot be done. The message names the cause and how to fix it.

    The Orchestrator maps this error to exit code 4 (failure in an evaluation stage).
    """


class EvaluatorConfigError(EvaluatorError):
    """The evaluation settings are wrong (unknown metric or method name, impossible value).

    The Orchestrator maps this error to exit code 2 (config error).
    """
