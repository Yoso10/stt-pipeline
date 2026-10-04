"""The `data` configuration section, owned by the DataSource component.

How to run: this module is not a script. The composition root registers the section:
    from stt.config import SectionRegistry, load_config
    from stt.datasource.config import DataConfig
    config = load_config(path, registry=SectionRegistry({"data": DataConfig}))
    data = config.section("data", DataConfig)
Example YAML: configs/data_gmara.yaml (an example for the Daf Yomi dataset; no dataset names in code).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from stt.config import SectionModel
from stt.config.validators import ExistingPath


class SourceConfig(SectionModel):
    """Where the data is and which kind of source reads it."""

    type: str = Field(description="Source name in the DataSource registry: `csv` or `hf_parquet`.")
    path: ExistingPath = Field(description="Folder with the data (a folder with manifest.csv, or a folder of parquet files).")


class ColumnsConfig(SectionModel):
    """Which columns of the manifest (or parquet table) hold what. Nested `metadata` fields are `meta_<name>`."""

    id: str | None = Field(default=None, description="Optional column with a unique sample id. Without it the audio path is the id.")
    audio: str = Field(description="Column with the audio file path (parquet: the audio struct).")
    text: str = Field(description="Column with the transcript.")
    group: str | None = Field(default=None, description="Column with the group (for example the lesson) that must not be split.")
    split: str | None = Field(default=None, description="Optional column with the source split name.")


class ExtractConfig(SectionModel):
    """Where parquet audio is extracted to (needed for `hf_parquet`)."""

    dir: Path | None = Field(default=None, description="Folder for the extracted audio and manifest.csv (outside src/ and tests/).")
    overwrite: bool = Field(default=False, description="DEFAULT false: reuse a complete earlier extraction.")


class DataConfig(SectionModel):
    """The `data` section."""

    source: SourceConfig
    columns: ColumnsConfig
    group_fallback: Literal["parent_dir", "file_stem", "fail"] = Field(
        default="fail", description="DEFAULT fail. What to do without a group column; a wrong group causes train/test leakage."
    )
    extra_types: dict[str, Literal["str", "int", "float", "bool"]] = Field(
        default_factory=dict, description="Column -> type of the values kept in `extra`; unlisted columns stay text."
    )
    extract: ExtractConfig = ExtractConfig()
    on_invalid: Literal["skip", "fail"] = Field(default="skip", description="DEFAULT skip. `fail` stops at the first rejected sample.")
    max_rejected_ratio: float = Field(
        default=0.05, ge=0.0, le=1.0, description="DEFAULT 0.05. The run fails when more than this fraction is rejected."
    )

    @model_validator(mode="after")
    def _parquet_needs_extract_dir(self) -> DataConfig:
        if self.source.type == "hf_parquet" and self.extract.dir is None:
            raise ValueError(
                "data.extract.dir is required when data.source.type is hf_parquet: "
                "set a folder for the extracted audio, for example data/extracted/<name>."
            )
        return self
