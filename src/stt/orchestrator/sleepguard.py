"""Sleep guards: keep Windows awake during a long run (standard library only, no admin rights).

How to run: this module is not a script; the command line wraps `run`, `train` and `smoke` in a guard. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.orchestrator.sleepguard import NoopSleepGuard; print(NoopSleepGuard().__enter__())"
`WindowsSleepGuard` calls the Windows function SetThreadExecutionState with ES_CONTINUOUS | ES_SYSTEM_REQUIRED on entry (the computer does not
go to sleep while the run is active; the screen may still turn off) and with ES_CONTINUOUS alone on exit, which restores the normal power policy.
It needs no admin rights. It is switched off with `run.prevent_sleep: false`. On other systems the no-op guard is used.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from types import TracebackType
from typing import Protocol

logger = logging.getLogger("stt.orchestrator")
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


class SleepGuard(Protocol):
    """A context manager that prevents sleep while it is active."""

    def __enter__(self) -> object:
        """Start preventing sleep."""
        ...

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        """Restore the normal power policy."""
        ...


class NoopSleepGuard:
    """Does nothing (other systems, or `run.prevent_sleep: false`)."""

    def __enter__(self) -> NoopSleepGuard:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        return None


def _windows_set_state(flags: int) -> int:
    import ctypes

    return int(ctypes.windll.kernel32.SetThreadExecutionState(flags))  # type: ignore[attr-defined]


class WindowsSleepGuard:
    """Calls SetThreadExecutionState on entry and exit. `set_state` is injectable so the tests need no Windows call."""

    def __init__(self, set_state: Callable[[int], int] | None = None) -> None:
        self._set_state = set_state or _windows_set_state

    def __enter__(self) -> WindowsSleepGuard:
        if self._set_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) == 0:
            logger.warning("Windows refused to prevent sleep (a power policy may block it); the run goes on and may be put to sleep.")
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        self._set_state(ES_CONTINUOUS)


def default_sleep_guard(enabled: bool) -> SleepGuard:
    """The guard for this system: Windows when enabled, else the no-op guard (with a log line when the system is not Windows)."""
    if not enabled:
        return NoopSleepGuard()
    if sys.platform == "win32":
        return WindowsSleepGuard()
    logger.info("Sleep prevention is only available on Windows; using the no-op guard.")
    return NoopSleepGuard()
