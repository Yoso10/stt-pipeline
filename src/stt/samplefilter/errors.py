"""Errors of the SampleFilter component.

How to run: this module is not a script; it is imported by the other modules of stt.samplefilter.
"""

from __future__ import annotations

from typing import Any


class FilterError(Exception):
    """The filter cannot run or a safety gate failed. The message names the rule, field or split and how to fix it.

    The Orchestrator maps this error to exit code 3 (data error).
    """


class FilterGateError(FilterError):
    """A safety gate failed. `result` holds the full FilterResult (counts per rule, would-be rejections),
    so that the report can still be read after the failure."""

    def __init__(self, message: str, result: Any = None) -> None:
        super().__init__(message)
        self.result = result
