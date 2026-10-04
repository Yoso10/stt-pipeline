"""stt.samplefilter: decide per split which samples are kept, and report exactly what was removed and why (C1b SampleFilter).

How to run: from Python (PowerShell, project root; the working example is in docs/COMPONENTS.md, C1b):
    $env:PYTHONPATH = "src"
    python -c "from stt.samplefilter import default_registry; print(default_registry().names())"
Example output:
    ['max_bad_segments', 'max_duration', 'max_label_tokens', 'min_duration', 'min_quality_score', 'min_text_chars', 'require_golden']
"""

from stt.samplefilter.config import FilterConfig, RuleConfig
from stt.samplefilter.errors import FilterError, FilterGateError
from stt.samplefilter.filter import SampleFilter
from stt.samplefilter.registry import RuleRegistry, default_registry
from stt.samplefilter.rules import (
    MaxBadSegments,
    MaxDuration,
    MaxLabelTokens,
    MinDuration,
    MinQualityScore,
    MinTextChars,
    RequireGolden,
)
from stt.samplefilter.types import (
    FilterReport,
    FilterResult,
    RejectedSample,
    RuleOutcome,
    RuleParams,
    RuleSpec,
    RuleVerdict,
    SampleRule,
    TokenCounter,
)

__all__ = [
    "FilterConfig",
    "FilterError",
    "FilterGateError",
    "FilterReport",
    "FilterResult",
    "MaxBadSegments",
    "MaxDuration",
    "MaxLabelTokens",
    "MinDuration",
    "MinQualityScore",
    "MinTextChars",
    "RejectedSample",
    "RequireGolden",
    "RuleConfig",
    "RuleOutcome",
    "RuleParams",
    "RuleRegistry",
    "RuleSpec",
    "RuleVerdict",
    "SampleFilter",
    "SampleRule",
    "TokenCounter",
    "default_registry",
]
