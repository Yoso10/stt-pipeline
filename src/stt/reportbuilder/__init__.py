"""stt.reportbuilder: the results of a run as one self-contained HTML report, and a live progress page (C14 ReportBuilder).

How to run: from Python (PowerShell, project root; the working example is in docs/COMPONENTS.md, C14):
    $env:PYTHONPATH = "src"
    python -c "from stt.reportbuilder import default_rule_registry; print(default_rule_registry().names())"
Example output:
    ['edge_of_range', 'failed_or_pruned', 'gain_concentrated', 'length_cap_or_empty', 'no_learning', 'overfitting', 'short_clip_weakness', 'unstable_training']
Importing this package imports no torch, no transformers, no plotting library and no network code.
"""

from stt.reportbuilder.builder import ReportBuilder
from stt.reportbuilder.convergence import ABSOLUTE_WER_TARGET_BAND, ConvergenceVerdict, verdict_from_points, verdict_of
from stt.reportbuilder.config import HintsConfig, ProgressConfig, ReportConfig
from stt.reportbuilder.correlation import correlation_row, ranks, spearman, thin
from stt.reportbuilder.errors import ReportConfigError, ReportError
from stt.reportbuilder.hints import RuleRegistry, default_rule_registry, group_errors
from stt.reportbuilder.progress import HtmlProgressSink, render_progress
from stt.reportbuilder.types import (
    BenchmarkView,
    ClipRow,
    CorrelationRow,
    CurvePointView,
    DiagnosisHint,
    DiagnosisRule,
    FinalTestView,
    HistoryPoint,
    LearningCurveView,
    ModelNumbers,
    ReportData,
    ReportOutput,
    RunInfoView,
    SearchView,
    SelectionView,
    TrainedRunView,
    TrialView,
)

__all__ = [
    "ABSOLUTE_WER_TARGET_BAND",
    "ClipRow",
    "ConvergenceVerdict",
    "FinalTestView",
    "ModelNumbers",
    "TrainedRunView",
    "verdict_from_points",
    "verdict_of",
    "BenchmarkView",
    "CorrelationRow",
    "CurvePointView",
    "DiagnosisHint",
    "DiagnosisRule",
    "HintsConfig",
    "HistoryPoint",
    "HtmlProgressSink",
    "LearningCurveView",
    "ProgressConfig",
    "ReportBuilder",
    "ReportConfig",
    "ReportConfigError",
    "ReportData",
    "ReportError",
    "ReportOutput",
    "RuleRegistry",
    "RunInfoView",
    "SearchView",
    "SelectionView",
    "TrialView",
    "correlation_row",
    "default_rule_registry",
    "group_errors",
    "ranks",
    "render_progress",
    "spearman",
    "thin",
]
