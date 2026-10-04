"""The `eval` configuration section, owned by the Evaluator component.

How to run: this module is not a script. The composition root registers the section:
    from stt.config import SectionRegistry, load_config
    from stt.evaluator.config import EvalConfig
    config = load_config(path, registry=SectionRegistry({"eval": EvalConfig}))
    ev = config.section("eval", EvalConfig)
Example YAML with every key and its DEFAULT: configs/eval_default.yaml. The seed comes from `runtime.seed`.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from stt.config import SectionModel


class GenerationConfig(SectionModel):
    """`eval.generation`: how the model decodes."""

    num_beams: int = Field(default=1, ge=1, description="DEFAULT 1 (greedy).")
    max_new_tokens: int | None = Field(default=None, ge=1, description="DEFAULT none = the model's label limit.")
    max_tokens_per_audio_sec: float | None = Field(
        default=None, gt=0, description="DEFAULT none = off (to be set from the first runs). Cap on the output length per second of audio."
    )


class ThresholdsConfig(SectionModel):
    """`eval.thresholds`."""

    wer_below: list[float] = Field(default_factory=lambda: [0.10, 0.20], description="DEFAULT 0.10, 0.20. Share of files with WER under each value.")

    @field_validator("wer_below")
    @classmethod
    def _in_range(cls, values: list[float]) -> list[float]:
        if any(not 0 < v <= 1 for v in values):
            raise ValueError("eval.thresholds.wer_below values must be in (0, 1], for example 0.10")
        return values


class SignificanceConfig(SectionModel):
    """`eval.significance`: the comparison of two results (lenient on purpose, owner)."""

    method: str = Field(default="paired_group_bootstrap", description="DEFAULT paired_group_bootstrap.")
    confidence: float = Field(default=0.90, gt=0, lt=1, description="DEFAULT 0.90.")
    one_sided: bool = Field(default=True, description="DEFAULT true (the candidate is better than the baseline).")
    n_resamples: int = Field(default=2000, ge=10, description="DEFAULT 2000.")
    min_effect_abs: float = Field(default=0.0, ge=0, description="DEFAULT 0.0. Absolute WER the improvement must exceed.")
    min_groups: int = Field(default=5, ge=2, description="DEFAULT 5. Fewer groups give `insufficient_groups`.")


class ReviewConfig(SectionModel):
    """`eval.review`: the human review list."""

    n_random: int = Field(default=20, ge=0, description="DEFAULT 20 random files.")
    n_worst: int = Field(default=10, ge=0, description="DEFAULT 10 worst files.")


class SanityConfig(SectionModel):
    """`eval.sanity`."""

    max_baseline_wer: float | None = Field(default=None, gt=0, description="DEFAULT none = off (phase 1: a small model has a very high Hebrew WER).")


class EvalConfig(SectionModel):
    """The `eval` section."""

    metrics: list[str] = Field(default_factory=lambda: ["wer", "cer"], min_length=1, description="DEFAULT wer, cer. Registered: cer, wer.")
    generation: GenerationConfig = GenerationConfig()
    batch_size: int = Field(default=4, ge=1, description="DEFAULT 4. Files per generation batch.")
    thresholds: ThresholdsConfig = ThresholdsConfig()
    significance: SignificanceConfig = SignificanceConfig()
    review: ReviewConfig = ReviewConfig()
    sanity: SanityConfig = SanityConfig()
