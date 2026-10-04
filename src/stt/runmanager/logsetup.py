"""LogSetup: one log file per experiment with every detail, and only the important lines on the console (standard library only).

How to run: this module is not a script; the Orchestrator calls it once per run. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.runmanager import LogSetup, RunConfig; print(LogSetup(RunConfig()))"
`configure(handle)` attaches two handlers to the logger `stt` (all components log below it): the file `logs/run.log` at `run.log.file_level`
(DEBUG: everything, with time and logger name) and the console (stderr) at `run.log.console_level` (INFO: short lines). `close()` removes them
again and restores the logger. The final JSON summary of a run is printed on stdout and is not a log line.
"""

from __future__ import annotations

import logging
import sys

from stt.runmanager.config import RunConfig
from stt.runmanager.handle import ExperimentHandle

ROOT_LOGGER = "stt"
FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
CONSOLE_FORMAT = "%(levelname)s %(message)s"


class LogSetup:
    """Configures logging for one experiment."""

    def __init__(self, config: RunConfig) -> None:
        self._config = config
        self._handlers: list[logging.Handler] = []
        self._saved: tuple[int, bool] | None = None

    def configure(self, handle: ExperimentHandle) -> None:
        """Attach the file and console handlers. Errors: RunManagerError when the log file cannot be opened. Calling it twice replaces the handlers."""
        self.close()
        logger = logging.getLogger(ROOT_LOGGER)
        self._saved = (logger.level, logger.propagate)
        handle.paths.logs.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(handle.paths.run_log, encoding="utf-8")
        file_handler.setLevel(self._config.log.file_level)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(self._config.log.console_level)
        console.setFormatter(logging.Formatter(CONSOLE_FORMAT))
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        for h in (file_handler, console):
            logger.addHandler(h)
            self._handlers.append(h)

    def close(self) -> None:
        """Remove the handlers and restore the logger."""
        logger = logging.getLogger(ROOT_LOGGER)
        for h in self._handlers:
            logger.removeHandler(h)
            h.close()
        self._handlers = []
        if self._saved is not None:
            logger.setLevel(self._saved[0])
            logger.propagate = self._saved[1]
            self._saved = None
