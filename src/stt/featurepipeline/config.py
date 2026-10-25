"""The `features` configuration section, owned by the FeaturePipeline component.

How to run: this module is not a script. The composition root registers the section:
    from stt.config import SectionRegistry, load_config
    from stt.featurepipeline.config import FeaturesConfig
    config = load_config(path, registry=SectionRegistry({"features": FeaturesConfig}))
    features = config.section("features", FeaturesConfig)
Example YAML with every key and its DEFAULT: configs/features_default.yaml. The sampling rate, the audio window,
the number of mel bins and the label limit are NOT here: they are read from the loaded model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from stt.config import SectionModel


class CacheConfig(SectionModel):
    """The optional feature cache."""

    enabled: bool = Field(default=False, description="DEFAULT false. When true, features are computed once and stored as files.")
    dir: Path | None = Field(default=None, description="Folder of the cache (required when enabled; outside src/ and tests/).")


class EncoderCacheConfig(SectionModel):
    """The optional encoder-output cache (DECISIONS 89): needs a FROZEN encoder (`train.freeze_policy: freeze_encoder`)."""

    enabled: bool = Field(
        default=False,
        description="DEFAULT false (nothing changes). When true, the output of the frozen audio encoder is computed once per clip and reused for training, the Val loss and the decoding (identical results, about 8 times faster training, 3 times faster decoding on the CPU).",
    )
    dir: Path | None = Field(default=None, description="Folder of the encoder cache (required when enabled; about 7.7 MB per clip for the turbo model in float32, half in float16).")
    dtype: Literal["float16", "float32"] = Field(
        default="float16", description="DEFAULT float16 (half the disk; the self-check and the 8-clip test showed identical decoded text). float32 is exact."
    )


class FeaturesConfig(SectionModel):
    """The `features` section."""

    num_workers: int = Field(default=0, ge=0, description="DEFAULT 0 (O2). Threads used to fill the cache; 0 = one at a time.")
    cache: CacheConfig = CacheConfig()
    encoder_cache: EncoderCacheConfig = EncoderCacheConfig()
    max_dropped_ratio: float = Field(
        default=0.05, ge=0.0, le=1.0, description="DEFAULT 0.05. The run fails when more than this fraction of labels is removed for length."
    )
    downmix: Literal["mean"] = Field(default="mean", description="DEFAULT mean. Stereo is averaged to mono.")
    shuffle: bool = Field(default=False, description="DEFAULT false. When true the examples are put in a seeded random order.")
    seed: int | None = Field(default=None, description="DEFAULT none = use runtime.seed. Seed of the shuffle.")

    @model_validator(mode="after")
    def _cache_needs_a_folder(self) -> FeaturesConfig:
        if self.cache.enabled and self.cache.dir is None:
            raise ValueError("features.cache.dir is required when features.cache.enabled is true: set a folder for the cache.")
        return self

    @model_validator(mode="after")
    def _encoder_cache_needs_a_folder(self) -> FeaturesConfig:
        if self.encoder_cache.enabled and self.encoder_cache.dir is None:
            raise ValueError("features.encoder_cache.dir is required when features.encoder_cache.enabled is true: set a folder for the encoder cache.")
        return self


def resolve_seed(features: FeaturesConfig, runtime_seed: int) -> int:
    """The seed of the shuffle: `features.seed` when set, else `runtime.seed`."""
    return features.seed if features.seed is not None else runtime_seed
