"""Errors of the Subsampler component.

How to run: this module is not a script; it is imported by the other modules of stt.subsampler.
"""

from __future__ import annotations


class SubsampleError(Exception):
    """A subset cannot be made or a stored subset is stale. The message names the numbers, the split or the file and how to fix it.

    The Orchestrator maps this error to exit code 3 (data error).
    """


class SubsampleConfigError(SubsampleError):
    """The subsample settings are wrong (unknown strategy or preset, a size that does not fit the scope).

    The Orchestrator maps this error to exit code 2 (config error).
    """
