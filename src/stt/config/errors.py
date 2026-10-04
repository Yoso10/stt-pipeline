"""Errors of the Config component.

How to run: this module is not a script; it is imported by the other modules of stt.config.
"""

from __future__ import annotations


class ConfigError(Exception):
    """The configuration is wrong. The message says what is wrong, which key or path, and how to fix it.

    The Orchestrator maps this error to exit code 2 (config error).
    """
