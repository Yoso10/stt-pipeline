"""Errors of the RunManager component.

How to run: this module is not a script; it is imported by the other modules of stt.runmanager.
"""

from __future__ import annotations


class RunManagerError(Exception):
    """An experiment folder cannot be created, opened, locked or changed. The message names the path or the experiment and the fix.

    The Orchestrator maps this error to exit code 2 (config error) when it is raised before a run starts, and to the code of the
    failing stage otherwise.
    """
