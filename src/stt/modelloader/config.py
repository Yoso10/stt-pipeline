"""The `model` configuration section, owned by the ModelLoader component.

How to run: this module is not a script. The composition root registers the section:
    from stt.config import SectionRegistry, load_config
    from stt.modelloader.config import ModelConfig
    config = load_config(path, registry=SectionRegistry({"model": ModelConfig}))
    model = config.section("model", ModelConfig)
Example YAML: configs/model_default.yaml (phase 1: a local copy of openai/whisper-tiny; a commented later
example for an ivrit-ai model). No model name is hard-coded: the model is chosen only by `model.path`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from stt.config import SectionModel


class ModelConfig(SectionModel):
    """The `model` section."""

    path: Path = Field(description="REQUIRED. Local folder of the model (a saved Hugging Face Whisper folder). Never a Hub name.")
    loader: str = Field(default="hf_whisper", description="DEFAULT hf_whisper. Loader in the registry.")
    language: str = Field(default="he", description="DEFAULT he. Language code written into the generation settings (Hebrew only in this project).")
    task: Literal["transcribe", "translate"] = Field(default="transcribe", description="DEFAULT transcribe.")
    dtype: Literal["float32", "float16", "bfloat16"] = Field(default="float32", description="DEFAULT float32 (CPU).")
