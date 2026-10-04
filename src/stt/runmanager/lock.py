"""RunLock: the heartbeat lock file of an experiment (standard library only).

How to run: this module is not a script; `ExperimentHandle.lock()` uses it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from pathlib import Path; from stt.runmanager.lock import RunLock; import tempfile; p = Path(tempfile.mkdtemp()) / 'run.lock'; l = RunLock(p, 30, 10); l.acquire(); print(p.exists()); l.release(); print(p.exists())"
Example output:
    True
    False
While a run is active the lock file `run.lock` exists and its modification time is refreshed every `heartbeat_seconds`. A second opener FAILS.
A lock whose heartbeat is older than `stale_minutes` is reported as STALE and also FAILS, with the instruction to delete the file by hand
after checking that no run is active. The lock is never removed automatically. The process id is NOT probed (on Windows signalling a
process id can terminate that process); the id inside the file is only information for the human.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

from stt.runmanager.errors import RunManagerError


class RunLock:
    """Exclusive lock on one experiment folder, with a heartbeat thread."""

    def __init__(self, path: Path, heartbeat_seconds: float, stale_minutes: float, clock: Callable[[], float] = time.time) -> None:
        self._path = path
        self._heartbeat = heartbeat_seconds
        self._stale_seconds = stale_minutes * 60.0
        self._clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def acquire(self) -> None:
        """Take the lock. Errors: RunManagerError when it is held (naming the heartbeat age) or stale (with the removal instruction)."""
        try:
            fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            self._explain_existing()
            return
        except OSError as exc:
            raise RunManagerError(f"Cannot create the lock file {self._path} ({exc}). Check that the folder is writable and the disk is not full.") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid()}, handle)
        self._stop.clear()
        self._thread = threading.Thread(target=self._beat, name="run-lock-heartbeat", daemon=True)
        self._thread.start()

    def _explain_existing(self) -> None:
        try:
            age = self._clock() - self._path.stat().st_mtime
        except OSError:
            age = 0.0
        if age >= self._stale_seconds:
            raise RunManagerError(
                f"The lock {self._path} is STALE: its heartbeat is {age / 60:.1f} minutes old (the limit is {self._stale_seconds / 60:g} minutes). "
                f"A run probably died without releasing it. Check that no run is active in this experiment, then delete the file {self._path} by hand "
                "and start again. It is never removed automatically."
            )
        raise RunManagerError(
            f"The experiment is locked by another run: {self._path} has a heartbeat of {age:.0f} s ago. "
            "Wait for that run to end or stop it, then try again."
        )

    def _beat(self) -> None:
        while not self._stop.wait(self._heartbeat):
            try:
                os.utime(self._path, None)
            except OSError:
                pass  # a missing or locked file is noticed by the next opener; the run itself goes on

    def release(self) -> None:
        """Stop the heartbeat and remove the lock file (only a lock that this object acquired)."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(1.0, 2 * self._heartbeat))
        self._thread = None
        try:
            self._path.unlink()
        except OSError:
            pass
