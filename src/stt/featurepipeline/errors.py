"""Errors of the FeaturePipeline component.

How to run: this module is not a script; it is imported by the other modules of stt.featurepipeline.
"""

from __future__ import annotations


class FeaturePipelineError(Exception):
    """The features cannot be built. The message names the count, the sample or the file, and how to fix it.

    The Orchestrator maps this error to exit code 3 (data error).
    """


class AudioReadError(Exception):
    """An audio file cannot be read (raised by the audio adapter; the pipeline turns it into a FeaturePipelineError)."""
