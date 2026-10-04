"""stt.evaluator: transcribe an evaluation set, score it (WER, CER), compare two results, and list files to review (C7 Evaluator).

How to run: from Python (PowerShell, project root; the working example is in docs/COMPONENTS.md, C7):
    $env:PYTHONPATH = "src"
    python -c "from stt.evaluator import default_metric_registry; print(default_metric_registry().names())"
Example output:
    ['cer', 'wer']
Importing this package does not import torch or transformers (the adapter `hf_transcriber` is imported by the composition root).
"""

from stt.evaluator.bootstrap import PairedGroupBootstrap, resample_deltas
from stt.evaluator.comparator import Comparator
from stt.evaluator.config import EvalConfig
from stt.evaluator.errors import EvaluatorConfigError, EvaluatorError
from stt.evaluator.evaluator import Evaluator, make_decode_settings
from stt.evaluator.metrics import CharErrorRate, WordErrorRate, edit_distance
from stt.evaluator.registry import MetricRegistry, SignificanceRegistry, default_metric_registry, default_significance_registry
from stt.evaluator.review import build_review_list, word_diff
from stt.evaluator.types import (
    DURATION_ATTRIBUTE,
    ComparisonResult,
    DecodeSettings,
    EvalItem,
    EvalRequest,
    EvalResult,
    FileScore,
    GroupErrors,
    Metric,
    MetricScore,
    RawHypothesis,
    ReviewEntry,
    ReviewList,
    SignificanceTest,
    Transcriber,
)

__all__ = [
    "DURATION_ATTRIBUTE",
    "CharErrorRate",
    "Comparator",
    "ComparisonResult",
    "DecodeSettings",
    "EvalConfig",
    "EvalItem",
    "EvalRequest",
    "EvalResult",
    "Evaluator",
    "EvaluatorConfigError",
    "EvaluatorError",
    "FileScore",
    "GroupErrors",
    "Metric",
    "MetricRegistry",
    "MetricScore",
    "PairedGroupBootstrap",
    "RawHypothesis",
    "ReviewEntry",
    "ReviewList",
    "SignificanceRegistry",
    "SignificanceTest",
    "Transcriber",
    "WordErrorRate",
    "build_review_list",
    "default_metric_registry",
    "default_significance_registry",
    "edit_distance",
    "make_decode_settings",
    "resample_deltas",
    "word_diff",
]
