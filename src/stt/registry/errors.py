"""Errors of the ModelRegistry component.

How to run: this module is not a script; it is imported by the other modules of stt.registry.
"""

from __future__ import annotations


class RegistryError(Exception):
    """A registration, a lookup or an index operation failed. The message names the path or the entry and how to fix it.

    The Orchestrator maps this error to exit code 4 (failure in a stage). The registry is left unchanged by a failed registration.
    """


class RegistryVerifyError(RegistryError):
    """`verify` found a missing or changed file. `report` holds the VerifyReport (the files are listed in the message).

    The Orchestrator maps this error to exit code 5 (verification failure).
    """

    def __init__(self, message: str, report: object | None = None) -> None:
        super().__init__(message)
        self.report = report
