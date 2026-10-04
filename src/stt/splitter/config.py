"""The `split` configuration section, owned by the Splitter component.

How to run: this module is not a script. The composition root registers the section:
    from stt.config import SectionRegistry, load_config
    from stt.splitter.config import SplitConfig
    config = load_config(path, registry=SectionRegistry({"split": SplitConfig}))
    split = config.section("split", SplitConfig)
Example YAML: configs/split_default.yaml. Every key has a DEFAULT; nothing is hard-coded in the code.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from stt.config import SectionModel
from stt.config.validators import check_ratios_sum_to_one


class GroupKeyConfig(SectionModel):
    """How the group of a sample is computed (whole groups are never split)."""

    column: str | None = Field(
        default=None,
        description="DEFAULT none = the DataSource group (`group_id`). Else a field name: group_id, id, source_split, or a key of `extra` (for example meta_source).",
    )
    regex: str | None = Field(
        default=None, description="DEFAULT none. A regular expression applied to the column value; its first capture group is the group."
    )

    @field_validator("regex")
    @classmethod
    def _compiles(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                re.compile(value)
            except re.error as exc:
                raise ValueError(f"not a valid regular expression: {exc}") from exc
        return value


class TimeBlockConfig(SectionModel):
    """Settings of the `time_block` mode."""

    order_column: str = Field(default="meta_seek", description="DEFAULT meta_seek. Key of `extra` with the start time (seconds) of the clip in its recording.")
    guard_gap_sec: float = Field(default=30.0, ge=0.0, description="DEFAULT 30. Clips closer than this to the previous block are dropped.")


class PredefinedConfig(SectionModel):
    """Settings of the `predefined` mode."""

    source_map: dict[str, str] = Field(
        default_factory=dict,
        description="DEFAULT empty = the source split name is the split name. Maps a source split name to a split name (for example eval: test).",
    )


class SplitConfig(SectionModel):
    """The `split` section."""

    mode: str = Field(default="grouped_ratio", description="DEFAULT grouped_ratio. Splitter in the registry: predefined, grouped_ratio, hash_group, time_block.")
    group_key: GroupKeyConfig = GroupKeyConfig()
    names: tuple[str, ...] = Field(default=("train", "val", "test"), description="DEFAULT train, val, test. The split names, in order.")
    ratios: tuple[float, ...] = Field(default=(0.8, 0.1, 0.1), description="DEFAULT 0.8, 0.1, 0.1. Must sum to 1, one per name.")
    ratio_unit: Literal["duration", "samples"] = Field(default="duration", description="DEFAULT duration. What the ratios measure.")
    ratio_tolerance: float = Field(default=0.05, ge=0.0, le=1.0, description="DEFAULT 0.05 absolute. Allowed deviation from a requested ratio.")
    seed: int | None = Field(default=None, description="DEFAULT none = use runtime.seed.")
    min_samples_per_split: int = Field(default=30, ge=0, description="DEFAULT 30. Review when the data volume is known (O22).")
    min_groups_per_split: int = Field(default=2, ge=0, description="DEFAULT 2. Review when the data volume is known (O22).")
    on_oversized_group: Literal["fail", "warn"] = Field(default="fail", description="DEFAULT fail. A group larger than a split can hold.")
    on_overlap: Literal["fail", "drop_from_train"] = Field(default="fail", description="DEFAULT fail. For `predefined`: a group present in several source splits.")
    time_block: TimeBlockConfig = TimeBlockConfig()
    predefined: PredefinedConfig = PredefinedConfig()
    speaker_column: str | None = Field(
        default=None,
        description="DEFAULT none. Key of `extra` with the speaker or creator; the report then counts speakers shared between splits.",
    )
    dir: Path | None = Field(default=None, description="DEFAULT none = nothing is saved. Folder for saved splits (`<dir>/<version>/split.json`).")
    version: str = Field(default="v1", description="DEFAULT v1. A new version is the only way to create a new split.")
    reuse: bool = Field(default=True, description="DEFAULT true. Reuse a saved split of this version.")
    export_csv: bool = Field(default=False, description="DEFAULT false. Also write `split.csv` (id, split) next to split.json.")

    @model_validator(mode="after")
    def _names_and_ratios(self) -> SplitConfig:
        if len(self.names) != len(self.ratios):
            raise ValueError(f"split.names has {len(self.names)} entries but split.ratios has {len(self.ratios)}; they must match.")
        if len(set(self.names)) != len(self.names) or not all(self.names):
            raise ValueError("split.names must be unique, non-empty names.")
        if any(r <= 0 for r in self.ratios):
            raise ValueError("every value of split.ratios must be greater than 0.")
        check_ratios_sum_to_one(dict(zip(self.names, self.ratios)))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.version):
            raise ValueError("split.version may only contain letters, digits, '_', '.' and '-'.")
        return self


def resolve_seed(split: SplitConfig, runtime_seed: int) -> int:
    """The seed of the split: `split.seed` when set, else `runtime.seed`."""
    return split.seed if split.seed is not None else runtime_seed
