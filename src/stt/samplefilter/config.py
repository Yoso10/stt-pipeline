"""The `filter` configuration section, owned by the SampleFilter component.

How to run: this module is not a script. The composition root registers the section:
    from stt.config import SectionRegistry, load_config
    from stt.samplefilter.config import FilterConfig
    config = load_config(path, registry=SectionRegistry({"filter": FilterConfig}))
    filter_config = config.section("filter", FilterConfig)
Example YAML with every key: configs/filter_default.yaml. Only the rules you mention are overridden; every
other rule keeps the default of its registry entry (see stt.samplefilter.registry), so a partial YAML never
loses the other rules.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from stt.config import SectionModel


class RuleConfig(SectionModel):
    """Overrides of one rule. A key left unset keeps the default of the rule."""

    enabled: bool | None = Field(
        default=None, description="DEFAULT unset = the rule's default; setting a threshold on a disabled rule enables it."
    )
    applies_to: tuple[str, ...] | None = Field(
        default=None, description='DEFAULT unset = all splits for a structural rule, only "train" for a quality rule. "*" means every split.'
    )
    kind: Literal["structural", "quality"] | None = Field(default=None, description="DEFAULT unset = the rule's own kind.")
    threshold: float | None = Field(default=None, ge=0.0, description="The rule's threshold (seconds, characters, score, count or tokens).")
    field: str | None = Field(
        default=None, description="Name of the sample's `extra` field a metadata rule reads (dataset specific, so it has no default in code)."
    )
    measure_on: Literal["eval_text", "train_text"] | None = Field(default=None, description="Which text a text rule measures.")

    @field_validator("threshold")
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        if value is not None and value != value:
            raise ValueError("threshold must be a number.")
        return value


class FilterConfig(SectionModel):
    """The `filter` section."""

    mode: Literal["report", "apply"] = Field(
        default="report", description="DEFAULT report: count what WOULD be removed and keep everything. `apply` removes."
    )
    rules: dict[str, RuleConfig] = Field(
        default_factory=dict, description="Per-rule overrides by rule name. An unknown name fails and lists the registered rules."
    )
    min_kept_ratio: float = Field(
        default=0.4, ge=0.0, le=1.0, description="DEFAULT 0.4. A split that would keep less than this fraction fails (safety gate)."
    )
