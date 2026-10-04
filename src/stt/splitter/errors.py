"""Errors of the Splitter component.

How to run: this module is not a script; it is imported by the other modules of stt.splitter.
"""

from __future__ import annotations


class SplitError(Exception):
    """The split cannot be made or is not valid. The message says what is wrong and how to fix it.

    The Orchestrator maps this error to exit code 3 (data error).
    """
