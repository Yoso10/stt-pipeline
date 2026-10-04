"""Errors of the ModelLoader component.

How to run: this module is not a script; it is imported by the other modules of stt.modelloader.
"""

from __future__ import annotations


class ModelLoaderError(Exception):
    """The model cannot be loaded. The message names the folder, the missing files or the problem, and how to fix it.

    The Orchestrator maps this error to exit code 2 (config error).
    """
