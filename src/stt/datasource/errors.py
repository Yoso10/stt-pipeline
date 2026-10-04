"""Errors of the DataSource component.

How to run: this module is not a script; it is imported by the other modules of stt.datasource.
"""

from __future__ import annotations


class DataSourceError(Exception):
    """The data cannot be loaded. The message says what is wrong, which column or path, and how to fix it.

    The Orchestrator maps this error to exit code 3 (data error).
    """
