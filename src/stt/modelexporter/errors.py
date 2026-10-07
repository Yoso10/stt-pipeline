"""Errors of the ModelExporter component.

How to run: this module is not a script; it is imported by the other modules of stt.modelexporter.
"""

from __future__ import annotations


class ExportError(Exception):
    """An export or its verification failed. The message names the entry, the exporter and how to fix it.

    The Orchestrator maps this error to exit code 5 (export or verification failure, C13 brief rule 4).
    """


class ExportConfigError(ExportError):
    """The export settings are wrong (an unknown exporter name: the registered names are listed). The Orchestrator maps this error to exit code 2 (config error)."""


class ExportDependencyError(ExportError):
    """An exporter needs a package that is not installed. The message names the package and says that the machine is offline, so the package must be brought in
    advance with the other wheels (never enable online mode as a fix). Exit code 5."""
