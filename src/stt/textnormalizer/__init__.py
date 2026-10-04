"""stt.textnormalizer: two texts per sample, `train_text` and `eval_text`, plus a character inventory (C2 TextNormalizer).

How to run: from Python (PowerShell, project root; the working example is in docs/COMPONENTS.md, C2):
    $env:PYTHONPATH = "src"
    python -c "from stt.textnormalizer import default_registry; print(default_registry().names())"
Example output:
    ['collapse_whitespace', 'lowercase_latin', 'remove_bracket_markup', 'remove_nikud', 'remove_punctuation', 'strip_timestamps', 'unicode_clean', 'unify_quotes']
"""

from stt.textnormalizer.config import TextConfig
from stt.textnormalizer.errors import TextNormalizerError
from stt.textnormalizer.normalizer import PROFILE_NAMES, TextNormalizer, build_profile
from stt.textnormalizer.registry import StepRegistry, default_registry
from stt.textnormalizer.timestamps import parse_segments
from stt.textnormalizer.types import (
    FLAGS,
    CharStat,
    NormalizationStep,
    NormalizedSample,
    NormalizeReport,
    NormalizeResult,
    TimestampSegment,
)

__all__ = [
    "FLAGS",
    "PROFILE_NAMES",
    "CharStat",
    "NormalizationStep",
    "NormalizeReport",
    "NormalizeResult",
    "NormalizedSample",
    "StepRegistry",
    "TextConfig",
    "TextNormalizer",
    "TextNormalizerError",
    "TimestampSegment",
    "build_profile",
    "default_registry",
    "parse_segments",
]
