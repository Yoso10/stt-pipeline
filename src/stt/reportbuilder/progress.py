"""HtmlProgressSink: the live progress page of a training run (a TrainerCore `MetricsSink`).

How to run: this module is not a script; the Orchestrator passes the sink to the TrainerCore (the task that wires it also names the file). Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from pathlib import Path; from stt.reportbuilder import HtmlProgressSink, ReportConfig; s = HtmlProgressSink(Path('data/progress_demo.html'), ReportConfig(), 'demo'); print(s.path)"
Example output:
    data\\progress_demo.html
The sink collects the metrics records it receives and rewrites the page after every `report.progress.every_n_records` records (a temporary file and a rename, so a browser never reads a half-written
page). The page asks the browser to reload itself every `report.progress.refresh_seconds` seconds (`<meta http-equiv="refresh">`), shows the latest step, the training and Val loss, the learning rate, the step
time and the estimated time left (the estimate is the TrainerCore's own, so the page never reads the clock), and the curves of the losses, the learning rate and the step time. A failure to write
the file is logged (the first time and then every 50th time) and the run goes on: the live page must never stop a training. The page is self-contained (inline SVG, no network). It keeps refreshing after
the run ends and shows the last record.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from html import escape
from pathlib import Path

from stt.reportbuilder import svg
from stt.reportbuilder.config import ReportConfig
from stt.reportbuilder.correlation import thin
from stt.trainercore import StepMetrics

logger = logging.getLogger("stt.reportbuilder")


def _span(seconds: float | None) -> str:
    if seconds is None or seconds != seconds:
        return "unknown"
    total = int(round(seconds))
    if total < 60:
        return f"{total} s"
    if total < 3600:
        return f"{total // 60} min {total % 60} s"
    return f"{total // 3600} h {(total % 3600) // 60} min"


def render_progress(records: Sequence[StepMetrics], settings: ReportConfig, title: str) -> str:
    """The progress page for the records so far (pure: the same records give the same page)."""
    cap = settings.max_points_per_chart
    refresh = settings.progress.refresh_seconds
    head = (
        f'<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><meta http-equiv="refresh" content="{refresh}"><title>{escape(title)} - progress</title>'
        "<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:20px auto;padding:0 14px}.panel{background:#f3f7fb;border:1px solid #b9cde0;border-radius:8px;padding:12px 16px}"
        "svg{max-width:100%;height:auto}.small{font-size:12.5px;color:#444}@media print{body{max-width:none}}</style></head><body>"
        f"<h1>{escape(title)}: training progress</h1>"
    )
    if not records:
        return head + f'<p>No metrics record yet. This page reloads itself every {refresh} seconds.</p></body></html>\n'
    last = records[-1]
    val = next((m for m in reversed(records) if m.val_loss is not None), None)
    panel = (
        f"<p>Step <b>{last.step}</b> (epoch {last.epoch_fraction:.2f}) &middot; training loss <b>{'-' if last.train_loss is None else f'{last.train_loss:.4f}'}</b> &middot; "
        f"latest Val loss <b>{'-' if val is None or val.val_loss is None else f'{val.val_loss:.4f} (step {val.step})'}</b> &middot; learning rate {'-' if last.learning_rate is None else f'{last.learning_rate:.3g}'}<br>"
        f"{last.seconds_per_step:.2f} s per step &middot; elapsed {_span(last.elapsed_sec)} &middot; <b>estimated time left {_span(last.eta_sec)}</b></p>"
    )
    train, _ = thin([(float(m.step), float(m.train_loss)) for m in records if m.train_loss is not None], cap)
    vals, _ = thin([(float(m.step), float(m.val_loss)) for m in records if m.val_loss is not None], cap)
    charts = svg.line_chart("Loss against step", [("training loss", train, svg.PALETTE[1], False), ("Val loss", vals, svg.PALETTE[0], False)], "step", "loss")
    lr, _ = thin([(float(m.step), float(m.learning_rate)) for m in records if m.learning_rate is not None], cap)
    speed, _ = thin([(float(m.step), float(m.seconds_per_step)) for m in records], cap)
    if lr:
        charts += svg.line_chart("Learning rate against step", [("learning rate", lr, svg.PALETTE[2], False)], "step", "learning rate")
    charts += svg.line_chart("Seconds per step", [("seconds per step", speed, svg.PALETTE[3], False)], "step", "seconds")
    note = f'<p class="small">{len(records)} records. The page reloads itself every {refresh} seconds; the estimate of the time left is the TrainerCore\'s and changes as the steps get faster or slower.</p>'
    return head + f'<div class="panel">{panel}</div>{charts}{note}</body></html>\n'


class HtmlProgressSink:
    """A `MetricsSink` that keeps the live progress page up to date."""

    def __init__(self, path: Path, settings: ReportConfig, title: str) -> None:
        self.path = path
        self._settings = settings
        self._title = title
        self._records: list[StepMetrics] = []
        self._failures = 0

    def write(self, record: StepMetrics) -> None:
        """Take one record and, every `every_n_records` records, rewrite the page. Never raises: a failure is logged and the caller goes on."""
        self._records.append(record)
        if len(self._records) != 1 and len(self._records) % self._settings.progress.every_n_records:
            return
        try:
            html = render_progress(self._records, self._settings, self._title)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + ".tmp")
            temporary.write_text(html, encoding="utf-8", newline="\n")
            os.replace(temporary, self.path)
        except Exception as exc:  # the page must never stop a training run
            self._failures += 1
            if self._failures == 1 or self._failures % 50 == 0:
                logger.error("The progress page %s cannot be written (%s: %s); the training goes on (failure %d).", self.path, type(exc).__name__, exc, self._failures)
