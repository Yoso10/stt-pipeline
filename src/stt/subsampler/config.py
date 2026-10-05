"""The `subsample` configuration section, owned by the Subsampler component.

How to run: this module is not a script. The composition root registers the section:
    from stt.config import SectionRegistry, load_config
    from stt.subsampler.config import SubsampleConfig
    config = load_config(path, registry=SectionRegistry({"subsample": SubsampleConfig}))
    sub = config.section("subsample", SubsampleConfig)
Example YAML with every key and its DEFAULT: configs/subsample_default.yaml. The seed comes from `runtime.seed`. A size is `all` or exactly one of
`{minutes: 30}`, `{samples: 20}`, `{fraction: 0.1}` (on the command line: `--set 'subsample.size={"minutes": 30}'`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from stt.config import SectionModel
from stt.subsampler.errors import SubsampleConfigError
from stt.subsampler.types import SubsampleSize, SubsampleSpec


class SizeConfig(SectionModel):
    """A size with exactly one of the three forms."""

    minutes: float | None = Field(default=None, gt=0, description="Minutes of audio (the owner's unit).")
    samples: int | None = Field(default=None, ge=1, description="A number of samples.")
    fraction: float | None = Field(default=None, gt=0, le=1, description="A fraction (0 to 1] of the minutes of audio, for example 0.1 for one tenth.")

    @model_validator(mode="after")
    def _exactly_one(self) -> SizeConfig:
        given = [name for name in ("minutes", "samples", "fraction") if getattr(self, name) is not None]
        if len(given) != 1:
            raise ValueError(f"a size needs exactly one of minutes, samples or fraction (got {given or 'none'}); or use the word all")
        return self

    def to_size(self) -> SubsampleSize:
        """The typed size."""
        for name in ("minutes", "samples", "fraction"):
            value = getattr(self, name)
            if value is not None:
                return SubsampleSize(name, float(value))  # type: ignore[arg-type]
        raise AssertionError("unreachable")


class PresetConfig(SectionModel):
    """A named size, optionally with a strategy and a scope. Only the presets you list are overridden; the built-in ones stay."""

    size: SizeConfig | Literal["all"] = Field(default="all", description="The size of the preset.")
    strategy: str | None = Field(default=None, description="Optional strategy of the preset (by_sample or by_group); empty = the config's.")
    scope: Literal["train_only", "all_splits"] | None = Field(default=None, description="Optional scope of the preset; empty = the config's.")


class SubsampleConfig(SectionModel):
    """The `subsample` section."""

    enabled: bool = Field(default=False, description="DEFAULT false. Off: the splits are returned unchanged (the main runs use all the data).")
    size: SizeConfig | Literal["all"] = Field(default="all", description="DEFAULT all. `all` or one of minutes, samples, fraction.")
    strategy: str = Field(default="by_sample", description="DEFAULT by_sample. The unit of selection: by_sample or by_group (registry).")
    scope: Literal["train_only", "all_splits"] = Field(default="train_only", description="DEFAULT train_only. all_splits reduces Train, Val and Test by the same fraction (rehearsal only).")
    val: Literal["fixed", "follow"] = Field(default="fixed", description="DEFAULT fixed. follow reduces Val by the same share as Train (train_only scope).")
    val_samples: int | None = Field(default=None, ge=1, description="NEW, DEFAULT none. A FIXED number of Val samples (a Val slice, for example 100), whatever the Train size. Train_only scope; not with val: follow.")
    repeats: int = Field(default=1, ge=1, description="DEFAULT 1. How many draws of the same size the caller makes (repeat_index 0 to repeats-1).")
    repeat_index: int = Field(default=0, ge=0, description="NEW, DEFAULT 0. Which draw this run uses; must be below `repeats`.")
    min_train_samples: int = Field(default=10, ge=1, description="DEFAULT 10. Fewer selected Train samples FAIL the run.")
    preset: str | None = Field(default=None, description="NEW, DEFAULT none. Name of a preset (smoke, tiny, work_like, half, full, or one of `presets`): it replaces size and, where defined, strategy and scope.")
    presets: dict[str, PresetConfig] = Field(default_factory=dict, description="DEFAULT empty. Your own presets; the built-in ones are in the code (smoke, tiny, work_like, half, full).")
    dir: Path | None = Field(default=None, description="NEW, DEFAULT none = nothing is saved. Folder for the stored subsets (JSON, one file per spec).")

    @model_validator(mode="after")
    def _repeat_index_in_range(self) -> SubsampleConfig:
        if self.repeat_index >= self.repeats:
            raise ValueError(f"subsample.repeat_index ({self.repeat_index}) must be below subsample.repeats ({self.repeats}).")
        if self.val_samples is not None and self.val == "follow":
            raise ValueError("subsample.val_samples (a fixed Val slice) cannot be combined with subsample.val: follow (a Val reduced in the Train share): choose one.")
        return self


BUILT_IN_PRESETS: dict[str, PresetConfig] = {
    "smoke": PresetConfig(size=SizeConfig(samples=20)),
    "tiny": PresetConfig(size=SizeConfig(minutes=30)),
    "work_like": PresetConfig(size=SizeConfig(fraction=0.1), strategy="by_group", scope="all_splits"),
    "half": PresetConfig(size=SizeConfig(fraction=0.5)),
    "full": PresetConfig(size="all"),
}


def preset_names(config: SubsampleConfig) -> list[str]:
    """The built-in and the configured preset names, sorted."""
    return sorted({*BUILT_IN_PRESETS, *config.presets})


def build_spec(config: SubsampleConfig, runtime_seed: int, train_split: str, val_split: str | None) -> SubsampleSpec:
    """The typed spec of the config (a preset replaces the size and, where it defines them, the strategy and the scope).

    Errors: SubsampleConfigError for an unknown preset (the names are listed).
    """
    size, strategy, scope = config.size, config.strategy, config.scope
    if config.preset is not None:
        preset = config.presets.get(config.preset) or BUILT_IN_PRESETS.get(config.preset)
        if preset is None:
            raise SubsampleConfigError(f"Unknown subsample.preset '{config.preset}'. Known presets: {', '.join(preset_names(config))}.")
        size, strategy, scope = preset.size, preset.strategy or strategy, preset.scope or scope
    typed = SubsampleSize("all") if size == "all" else size.to_size()  # type: ignore[union-attr]
    return SubsampleSpec(typed, strategy, scope, config.val, runtime_seed, config.repeat_index, train_split, val_split, config.min_train_samples, config.val_samples)
