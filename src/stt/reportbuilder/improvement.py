"""The improvement line of the report: the relative reduction of the Val WER against the baseline, against an optional target (Wave 3 task 7; DECISIONS 76, 78).

How to run: this module is not a script; the ReportBuilder uses it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.reportbuilder.improvement import improvement_line; print(improvement_line(0.60, 0.54, 0.10, True))"
Example output:
    Relative reduction of the Val WER against the baseline: 10.0% (target: 10%): reached
The numbers are the STORED Val WER of the baseline and of the best model (nothing is recomputed): reduction = (baseline - best) / baseline. Without a target (`report.target_relative_wer_reduction` is
none) the line says `no target set`; a baseline WER of 0 gives `not defined`; a reached target whose verdict is not `Proven improvement` ends with `Point estimate only`.
The Orchestrator prints the same line under `VERDICT:` from its own copy of this rule (it cannot import this package); a test keeps the two texts equal.
"""

from __future__ import annotations

import math

PREFIX = "Relative reduction of the Val WER against the baseline: "
POINT_ESTIMATE = " Point estimate only: the verdict is not 'Proven improvement' (see the verdict above)."


def relative_reduction(baseline: float | None, best: float | None) -> float | None:
    """(baseline - best) / baseline, or None when a number is missing or the baseline WER is 0 (or not finite)."""
    if baseline is None or best is None or not math.isfinite(baseline) or not math.isfinite(best) or baseline <= 0:
        return None
    return (baseline - best) / baseline


def improvement_line(baseline: float | None, best: float | None, target: float | None, proven: bool | None) -> str:
    """The one line (see the module docstring). `proven` is True for a `Proven improvement`, False for any other verdict, None when it is not known."""
    if baseline is None or best is None:
        return PREFIX + "not available (the Val WER of the baseline or of the best model is not stored)."
    if not math.isfinite(baseline) or not math.isfinite(best):
        return PREFIX + "not available (a stored Val WER is not a number)."
    if baseline <= 0:
        return PREFIX + "not defined (the baseline Val WER is 0)."
    reduction = relative_reduction(baseline, best)
    assert reduction is not None
    shown = f"{100.0 * reduction:.1f}%"
    if target is None:
        return f"{PREFIX}{shown} (no target set)."
    reached = reduction >= target - 1e-12
    line = f"{PREFIX}{shown} (target: {100.0 * target:g}%): {'reached' if reached else 'not reached'}"
    if reached and proven is False:
        return line + "." + POINT_ESTIMATE
    return line
