"""Metrics sinks: where the records of a run go while it is in progress (`train.sinks`).

How to run: this module is not a script; the registry creates the sinks. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.trainercore import default_sink_registry; print(default_sink_registry().names())"
Example output:
    ['console', 'csv']
Follow a run live: open `<output folder>/metrics.csv` in Excel (it is appended to and closed after every record, so it can be
read at any time; refresh by reopening it) or run `Get-Content <output folder>/metrics.csv -Wait` in PowerShell. A sink that
cannot write (for example the file is locked by another program) logs a warning and the training goes on.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import fields
from pathlib import Path

from stt.trainercore.types import SinkContext, StepMetrics

logger = logging.getLogger("stt.trainercore")
CSV_NAME = "metrics.csv"
COLUMNS = tuple(f.name for f in fields(StepMetrics))


def _cell(value: object) -> str:
    return "" if value is None else repr(value) if isinstance(value, float) else str(value)


class CsvMetricsSink:
    """The live file: one row per record in `<output_dir>/metrics.csv`, opened, appended and closed every time.

    On a resume the rows after the resumed step are removed first (they will be written again), so the file never holds a step twice.
    """

    def __init__(self, context: SinkContext) -> None:
        self.path = context.output_dir / CSV_NAME
        self._warned = False
        if context.resume_step is not None and self.path.exists():
            self._keep_up_to(context.resume_step)

    def _keep_up_to(self, step: int) -> None:
        try:
            with self.path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            kept = [rows[0]] + [r for r in rows[1:] if r and int(r[0]) <= step] if rows else []
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(kept)
        except (OSError, ValueError) as exc:
            self._warn(exc)

    def _warn(self, exc: Exception) -> None:
        if not self._warned:  # once, so a locked file does not flood the log
            logger.warning("The metrics file %s cannot be written (%s); training continues, the records stay in the log.", self.path, exc)
            self._warned = True

    def write(self, record: StepMetrics) -> None:
        """Append the record (a header first when the file is new). Never raises for a file problem."""
        try:
            new = not self.path.exists() or self.path.stat().st_size == 0
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if new:
                    writer.writerow(COLUMNS)
                writer.writerow([_cell(getattr(record, name)) for name in COLUMNS])
        except OSError as exc:
            self._warn(exc)


class ConsoleMetricsSink:
    """One log line per record (the log goes to the console and, later, to the run's log file)."""

    def __init__(self, context: SinkContext) -> None:
        del context  # nothing to prepare

    def write(self, record: StepMetrics) -> None:
        """Log the record."""
        def number(value: float | None, spec: str) -> str:
            return "-" if value is None else format(value, spec)

        logger.info(
            "step %d (epoch %.2f) train_loss %s val_loss %s lr %s grad_norm %s %.2f s/step elapsed %.0f s eta %s",
            record.step,
            record.epoch_fraction,
            number(record.train_loss, ".4f"),
            number(record.val_loss, ".4f"),
            number(record.learning_rate, ".3g"),
            number(record.grad_norm, ".3f"),
            record.seconds_per_step,
            record.elapsed_sec,
            "-" if record.eta_sec is None else f"{record.eta_sec:.0f} s",
        )


def metrics_path(output_dir: Path) -> Path:
    """The live metrics file of a run folder."""
    return output_dir / CSV_NAME
