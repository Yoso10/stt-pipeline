"""InterruptFlag: the stop flag that Ctrl+C (or a closing console window) sets (standard library only).

How to run: this module is not a script; the command line installs it around a run. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.orchestrator.stopflag import InterruptFlag; f = InterruptFlag(); print(f()); f.set(); print(f())"
Example output:
    False
    True
First Ctrl+C: the flag is set, the TrainerCore saves a checkpoint and stops after the current step, the status becomes `stopped` and the exit
code is 130; a rerun with the same experiment resumes. Second Ctrl+C: KeyboardInterrupt at once (for a long evaluation that does not look at the flag).
On Windows a closing console window (CTRL_BREAK / close) also sets the flag; the window may be killed before the checkpoint is written, which is why
the TrainerCore also saves checkpoints at a fixed rate.
"""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType
from typing import Any

logger = logging.getLogger("stt.orchestrator")


class InterruptFlag:
    """A thread-safe flag that is callable (True once a stop was requested)."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._presses = 0

    def __call__(self) -> bool:
        return self._event.is_set()

    def set(self) -> None:
        """Request a stop (also what the unit tests use to simulate an interrupt)."""
        self._event.set()

    def _handler(self, signum: int, frame: FrameType | None) -> None:
        self._presses += 1
        if self._presses == 1:
            self._event.set()
            logger.warning("Stop requested: finishing safely (a training run saves a checkpoint first). Press Ctrl+C again to stop at once.")
            return
        raise KeyboardInterrupt

    @contextmanager
    def installed(self) -> Iterator[InterruptFlag]:
        """Install the signal handlers for the duration of a `with` block (only in the main thread; elsewhere nothing is installed)."""
        names = [n for n in ("SIGINT", "SIGBREAK") if hasattr(signal, n)]
        previous: dict[int, Any] = {}
        if threading.current_thread() is threading.main_thread():
            for name in names:
                number = getattr(signal, name)
                previous[number] = signal.signal(number, self._handler)
        try:
            yield self
        finally:
            for number, old in previous.items():
                signal.signal(number, old)
