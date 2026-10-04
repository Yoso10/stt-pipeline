"""Errors of the TrainerCore component.

How to run: this module is not a script; it is imported by the other modules of stt.trainercore.
"""

from __future__ import annotations


class TrainerCoreError(Exception):
    """A training run failed or cannot start. The message names the step, folder or value and how to fix it.

    The Orchestrator maps this error to exit code 4 (failure in a training stage).
    """


class TrainerConfigError(TrainerCoreError):
    """The training settings are wrong (unknown policy or sink, impossible value, resume without a checkpoint).

    The Orchestrator maps this error to exit code 2 (config error).
    """
