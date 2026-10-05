"""Errors of the HyperparamSearcher component.

How to run: this module is not a script; it is imported by the other modules of stt.searcher.
"""

from __future__ import annotations


class SearchError(Exception):
    """A search cannot run or ended without a usable result. The message names the numbers, the study file and how to fix it.

    The Orchestrator maps this error to exit code 4 (failure in a training stage). `result` holds the SearchResult made so far (all trials, the study file is
    kept), when there is one, so that the report can still be written after the failure.
    """

    def __init__(self, message: str, result: object | None = None) -> None:
        super().__init__(message)
        self.result = result


class SearchInterrupted(Exception):
    """Raised BY A TRIAL RUNNER when the user asked to stop (the Orchestrator's stop flag). The searcher does not record that trial, keeps the stored study
    (the finished trials), lets running trials finish in parallel mode, and raises this exception again so that the caller can set the status `stopped`.
    A rerun with the same study continues."""


class SearchConfigError(SearchError):
    """The search settings are wrong (unknown strategy or parameter, a space that differs from the stored study, too many parallel trials, a missing package).

    The Orchestrator maps this error to exit code 2 (config error).
    """
