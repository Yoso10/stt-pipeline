"""Errors of the TextNormalizer component.

How to run: this module is not a script; it is imported by the other modules of stt.textnormalizer.
"""

from __future__ import annotations


class TextNormalizerError(Exception):
    """The text configuration cannot be used. The message names the step or profile and how to fix it.

    The Orchestrator maps this error to exit code 2 (config error).
    """
