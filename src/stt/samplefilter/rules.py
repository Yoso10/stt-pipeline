"""The filter rules. Each rule is a small class: it reads one thing about a sample and says pass or fail.

How to run: this module is not a script; the registry builds the rules from the `filter` config. Example
(PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.samplefilter.rules import MinDuration; print(MinDuration(1.0).name)"
Example output:
    min_duration
A new rule is a new class here (or anywhere) plus one entry in the registry; the filter never changes.
Rules are generic: thresholds and field names come from Config, never from the dataset.
"""

from __future__ import annotations

from stt.samplefilter.types import RuleVerdict, TokenCounter
from stt.textnormalizer import NormalizedSample

EPSILON = 1e-6  # floating-point tolerance for duration limits (a 30 s clip read as 30.0000001 passes a 30 s limit)
PASS = RuleVerdict(True)


def _extra_number(sample: NormalizedSample, name: str) -> float | None:
    value = sample.sample.extra.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class MinDuration:
    """STRUCTURAL: the clip must last at least `threshold` seconds."""

    name = "min_duration"
    kind = "structural"
    required_fields = ("duration_sec",)

    def __init__(self, threshold: float) -> None:
        self._limit = threshold

    def check(self, sample: NormalizedSample) -> RuleVerdict:
        """Fail when the duration is below the limit."""
        d = sample.sample.duration_sec
        return PASS if d >= self._limit - EPSILON else RuleVerdict(False, f"duration {d:.2f} s is below {self._limit} s")


class MaxDuration:
    """STRUCTURAL: the clip must last at most `threshold` seconds (the Whisper window)."""

    name = "max_duration"
    kind = "structural"
    required_fields = ("duration_sec",)

    def __init__(self, threshold: float) -> None:
        self._limit = threshold

    def check(self, sample: NormalizedSample) -> RuleVerdict:
        """Fail when the duration is above the limit."""
        d = sample.sample.duration_sec
        return PASS if d <= self._limit + EPSILON else RuleVerdict(False, f"duration {d:.2f} s is above {self._limit} s")


class MinTextChars:
    """STRUCTURAL: the measured text (`eval_text` by default) must have at least `threshold` characters."""

    name = "min_text_chars"
    kind = "structural"

    def __init__(self, threshold: float, measure_on: str = "eval_text") -> None:
        self._limit = threshold
        self._on = measure_on
        self.required_fields = (measure_on,)

    def check(self, sample: NormalizedSample) -> RuleVerdict:
        """Fail when the text is shorter than the limit."""
        text = getattr(sample, self._on).strip()
        return PASS if len(text) >= self._limit else RuleVerdict(False, f"{self._on} has {len(text)} characters (minimum {self._limit:g})")


class MinQualityScore:
    """QUALITY: the sample's quality score field must be at least `threshold`."""

    name = "min_quality_score"
    kind = "quality"

    def __init__(self, threshold: float, field: str) -> None:
        self._limit = threshold
        self._field = field
        self.required_fields = (f"extra:{field}",)

    def check(self, sample: NormalizedSample) -> RuleVerdict:
        """Fail when the score is below the limit (or is not a number)."""
        value = _extra_number(sample, self._field)
        if value is None:
            return RuleVerdict(False, f"{self._field} is not a number")
        return PASS if value >= self._limit else RuleVerdict(False, f"{self._field} {value:.4g} is below {self._limit:g}")


class MaxBadSegments:
    """QUALITY: the sample's bad-segments count field must be at most `threshold`."""

    name = "max_bad_segments"
    kind = "quality"

    def __init__(self, threshold: float, field: str) -> None:
        self._limit = threshold
        self._field = field
        self.required_fields = (f"extra:{field}",)

    def check(self, sample: NormalizedSample) -> RuleVerdict:
        """Fail when the count is above the limit (or is not a number)."""
        value = _extra_number(sample, self._field)
        if value is None:
            return RuleVerdict(False, f"{self._field} is not a number")
        return PASS if value <= self._limit else RuleVerdict(False, f"{self._field} {value:g} is above {self._limit:g}")


class RequireGolden:
    """QUALITY: the sample's golden field must be true."""

    name = "require_golden"
    kind = "quality"

    def __init__(self, field: str) -> None:
        self._field = field
        self.required_fields = (f"extra:{field}",)

    def check(self, sample: NormalizedSample) -> RuleVerdict:
        """Fail unless the field is the boolean true."""
        return PASS if sample.sample.extra.get(self._field) is True else RuleVerdict(False, f"{self._field} is not true")


class MaxLabelTokens:
    """STRUCTURAL (optional, off by default): the label (`train_text` by default) must have at most `threshold` tokens.

    The token counter is a plain callable `Callable[[str], int]` given to the filter by the composition root
    (the real one comes from the ModelLoader later); this module imports no tokenizer. This rule is the
    alternative of open issue O34; the brief's default keeps label-length removal in the FeaturePipeline.
    """

    name = "max_label_tokens"
    kind = "structural"

    def __init__(self, threshold: float, counter: TokenCounter, measure_on: str = "train_text") -> None:
        self._limit = threshold
        self._counter = counter
        self._on = measure_on
        self.required_fields = (measure_on,)

    def check(self, sample: NormalizedSample) -> RuleVerdict:
        """Fail when the label has more tokens than the limit."""
        count = self._counter(getattr(sample, self._on))
        return PASS if count <= self._limit else RuleVerdict(False, f"{self._on} has {count} tokens (maximum {self._limit:g})")
