"""Errors of the Orchestrator component.

How to run: this module is not a script; it is imported by the other modules of stt.orchestrator.
"""

from __future__ import annotations


class OrchestratorError(Exception):
    """The run cannot start or a stage refuses to run. The message names the stage, the key or the rule and how to fix it.

    `exit_code` is the process exit code of this error (2 = config error by default).
    """

    def __init__(self, message: str, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class StageInterrupted(Exception):
    """A stage stopped on request (the stop flag): its marker is not written, the status becomes `stopped`, the exit code is 130."""
