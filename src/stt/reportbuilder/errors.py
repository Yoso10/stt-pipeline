"""Errors of the ReportBuilder component.

How to run: this module is not a script; it is imported by the other modules of stt.reportbuilder.
"""

from __future__ import annotations


class ReportError(Exception):
    """A report cannot be built. The message names the cause and how to fix it. The Orchestrator maps this error to exit code 4."""


class ReportConfigError(ReportError):
    """The report settings are wrong (an unknown diagnosis rule: the registered names are listed). The Orchestrator maps this error to exit code 2."""
