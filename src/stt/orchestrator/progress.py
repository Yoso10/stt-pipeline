"""Live progress lines for the long phases of a run (Wave 3 task 7; DECISIONS 78).

How to run: this module is not a script; the stages and the composition root use it. The lines are written through the logger `stt.orchestrator` at INFO, so they look the same on a console and in
a file the output is redirected to, and they are also saved in `logs/run.log`. No colour, no cursor movement, no console detection. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.orchestrator.progress import progress_text; print(progress_text('WER scoring of the baseline model', 'clip', 120, 875, 231.0))"
Example output:
    PROGRESS WER scoring of the baseline model: clip 120 of 875 (13.7%), elapsed 0:03:51, remaining about 0:24:13
What is here:
  - `Progress`: for a phase whose items arrive often (the clips of a WER scoring): at most one line every `progress.interval_seconds` (the first update and the last one are always written);
    `update(done, total)` has the signature of `EvalRequest.on_progress`.
  - `ItemProgress`: for a phase of a few long items (finalists, learning-curve points): one line when each item starts, with the share of the items already finished.
  - `ThrottledSink`: wraps the console metrics sink of a training so that its line (`step 40 of 400 (10.0%) ... elapsed ... remaining ...`) is written at most once per interval; an evaluation record and the
    last step are always written.
Every line has the phase, the position and the total, the percent, the elapsed time and the estimated time remaining (`H:MM:SS`; the estimate is the elapsed time divided by the items done, times
the items left, so it is rough at the beginning). A console that fails to take a line never stops a run: the error is swallowed.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from stt.config import AppConfig
from stt.trainercore import MetricsSink, StepMetrics
from stt.trainercore.sinks import clock_text

logger = logging.getLogger("stt.orchestrator")
DEFAULT_INTERVAL_SECONDS = 30.0
Emit = Callable[[str], None]


def progress_interval(config: AppConfig) -> float:
    """`progress.interval_seconds` of the config (DEFAULT 30 when the section is not loaded)."""
    return float(getattr(config.sections.get("progress"), "interval_seconds", DEFAULT_INTERVAL_SECONDS))


def log_line(text: str) -> None:
    """Write one line through the logger; an error while writing (a broken console) is swallowed."""
    try:
        logger.info("%s", text)
    except Exception:  # noqa: BLE001  a progress line must never stop a run
        pass


def _remaining(elapsed: float, done: int, total: int) -> str:
    if done >= total:
        return "done"
    if done <= 0:
        return "remaining unknown"
    return f"remaining about {clock_text(elapsed * (total - done) / done)}"


def progress_text(phase: str, unit: str, done: int, total: int, elapsed: float, extra: str = "") -> str:
    """`PROGRESS <phase>: <unit> <done> of <total> (<percent>%), elapsed H:MM:SS, remaining about H:MM:SS` (plus `extra`)."""
    percent = 100.0 * min(done, total) / total if total > 0 else 0.0
    return f"PROGRESS {phase}: {unit} {done} of {total} ({percent:.1f}%), elapsed {clock_text(elapsed)}, {_remaining(elapsed, done, total)}{extra}"


class Progress:
    """Progress of one phase whose items arrive often. `update(done, total)` writes a line at most once per `interval_seconds` (the first and the last update are always written)."""

    def __init__(self, phase: str, unit: str, interval_seconds: float, clock: Callable[[], float] = time.monotonic, emit: Emit = log_line) -> None:
        self._phase, self._unit, self._interval = phase, unit, interval_seconds
        self._clock, self._emit = clock, emit
        self._started = clock()
        self._last: float | None = None
        self._final_written = False

    def update(self, done: int, total: int, extra: str = "") -> None:
        """Count `done` of `total` items; write a line when the interval has passed. Never raises."""
        try:
            if total <= 0:
                return
            now = self._clock()
            final = done >= total and not self._final_written
            if self._last is not None and not final and now - self._last < self._interval:
                return
            self._last = now
            self._final_written = self._final_written or done >= total
            self._emit(progress_text(self._phase, self._unit, done, total, now - self._started, extra))
        except Exception:  # noqa: BLE001  a progress line must never stop a run
            pass


class ItemProgress:
    """Progress of a phase of a few long items (finalists, learning-curve points): `starting(n)` writes one line when item `n` (1-based) begins."""

    def __init__(self, phase: str, unit: str, total: int, clock: Callable[[], float] = time.monotonic, emit: Emit = log_line) -> None:
        self._phase, self._unit, self._total = phase, unit, total
        self._clock, self._emit = clock, emit
        self._started = clock()

    def starting(self, number: int, extra: str = "") -> None:
        """Write `PROGRESS <phase>: starting <unit> n of N (x% done), elapsed ..., remaining about ...`. Never raises."""
        try:
            completed = max(0, number - 1)
            elapsed = self._clock() - self._started
            percent = 100.0 * completed / self._total if self._total > 0 else 0.0
            if completed > 0:
                remaining = f"remaining about {clock_text(elapsed * (self._total - completed) / completed)} (from the {completed} finished)"
            else:
                remaining = f"remaining unknown until the first {self._unit} is finished"
            self._emit(f"PROGRESS {self._phase}: starting {self._unit} {number} of {self._total} ({percent:.1f}% done), elapsed {clock_text(elapsed)}, {remaining}{extra}")
        except Exception:  # noqa: BLE001
            pass


class ThrottledSink:
    """Forwards a training's metrics records to `inner` (the console sink) at most once per `interval_seconds`.

    Always forwarded: the first record, a record with a Val loss (an evaluation) and the last step (`max_steps`). The other sinks (the CSV file) are not affected: they get every record."""

    def __init__(self, inner: MetricsSink, interval_seconds: float, max_steps: int | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self._inner, self._interval, self._max_steps, self._clock = inner, interval_seconds, max_steps, clock
        self._last: float | None = None

    def write(self, record: StepMetrics) -> None:
        """Forward the record when the interval has passed or the record is one of the always-written kinds. A failing inner sink is ignored."""
        try:
            now = self._clock()
            last_step = self._max_steps is not None and record.step >= self._max_steps
            if self._last is not None and record.val_loss is None and not last_step and now - self._last < self._interval:
                return
            self._last = now
            self._inner.write(record)
        except Exception:  # noqa: BLE001  a failing console must never stop a training
            pass
