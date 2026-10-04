"""The rule registry: rule name -> RuleSpec (defaults and factory), so rules are selected by name from Config.

How to run: this module is not a script. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.samplefilter.registry import default_registry; print(default_registry().names())"
Example output:
    ['max_bad_segments', 'max_duration', 'max_label_tokens', 'min_duration', 'min_quality_score', 'min_text_chars', 'require_golden']
Defaults (permissive, DEFAULT status): min_duration 1.0 s, max_duration 30.0 s, min_text_chars 2 on eval_text;
the quality rules, require_golden and max_label_tokens are off until configured.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from stt.samplefilter.errors import FilterError
from stt.samplefilter.rules import MaxBadSegments, MaxDuration, MaxLabelTokens, MinDuration, MinQualityScore, MinTextChars, RequireGolden
from stt.samplefilter.types import RuleParams, RuleSpec, SampleRule


def _need(value: float | None, rule: str) -> float:
    if value is None:
        raise FilterError(f"Rule '{rule}' is enabled but has no threshold. Set filter.rules.{rule}.threshold.")
    return value


def _field(params: RuleParams, rule: str) -> str:
    if not params.field:
        raise FilterError(
            f"Rule '{rule}' is enabled but filter.rules.{rule}.field is not set. Name the sample field it reads "
            "(a key of the DataSource `extra` fields, for example meta_quality_score)."
        )
    return params.field


def _counter_rule(params: RuleParams) -> SampleRule:
    if params.token_counter is None:
        raise FilterError(
            "Rule 'max_label_tokens' is enabled but no token counter was given to the SampleFilter. The composition root "
            "must pass one (from the ModelLoader), or the rule must be disabled."
        )
    return MaxLabelTokens(_need(params.threshold, "max_label_tokens"), params.token_counter, params.measure_on or "train_text")


class RuleRegistry:
    """A read-only name -> RuleSpec mapping."""

    def __init__(self, specs: Mapping[str, RuleSpec]) -> None:
        self._specs: Mapping[str, RuleSpec] = MappingProxyType(dict(specs))

    def names(self) -> list[str]:
        """The registered rule names, sorted."""
        return sorted(self._specs)

    def spec(self, name: str) -> RuleSpec:
        """The entry of rule `name`. Errors: FilterError listing the registered names."""
        found = self._specs.get(name)
        if found is None:
            raise FilterError(
                f"Unknown filter rule '{name}' in filter.rules. Registered rules: {', '.join(self.names())}. "
                "Fix the name in the YAML configuration."
            )
        return found


def default_registry() -> RuleRegistry:
    """The registry with the six rules of the brief and the optional token-length rule."""
    return RuleRegistry(
        {
            "min_duration": RuleSpec("structural", True, 1.0, None, True, False, lambda p: MinDuration(_need(p.threshold, "min_duration"))),
            "max_duration": RuleSpec("structural", True, 30.0, None, True, False, lambda p: MaxDuration(_need(p.threshold, "max_duration"))),
            "min_text_chars": RuleSpec(
                "structural", True, 2.0, "eval_text", True, False, lambda p: MinTextChars(_need(p.threshold, "min_text_chars"), p.measure_on or "eval_text")
            ),
            "min_quality_score": RuleSpec(
                "quality", False, None, None, True, True, lambda p: MinQualityScore(_need(p.threshold, "min_quality_score"), _field(p, "min_quality_score"))
            ),
            "max_bad_segments": RuleSpec(
                "quality", False, None, None, True, True, lambda p: MaxBadSegments(_need(p.threshold, "max_bad_segments"), _field(p, "max_bad_segments"))
            ),
            "require_golden": RuleSpec("quality", False, None, None, False, True, lambda p: RequireGolden(_field(p, "require_golden"))),
            "max_label_tokens": RuleSpec("structural", False, None, "train_text", True, False, _counter_rule),
        }
    )
