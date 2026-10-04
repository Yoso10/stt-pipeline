"""Typed data objects of the ModelLoader component and the small Protocols of the loaded objects.

How to run: this module is not a script; the other modules and later components import it. Example
(PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.modelloader.types import ModelInfo; print(ModelInfo.__dataclass_fields__.keys())"

Design of the Protocols (decision 37 in docs/DECISIONS.md): they hold only what the FeaturePipeline, the
TrainerCore, the Evaluator and the SampleFilter wiring need, so that no Hugging Face type leaks into
high-level code. The model itself is opaque (`object`): only the adapter modules of the trainer and the
evaluator look inside it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class FeatureExtractor(Protocol):
    """Turns one audio clip into model input features. All numbers are READ from the model, never hard-coded.

    `sampling_rate` (Hz) is what `extract` expects; `window_seconds` is the audio window the model takes (the
    clips are padded up to it); `num_mel_bins` is the number of mel bins of the features.
    """

    sampling_rate: int
    window_seconds: float
    num_mel_bins: int

    def extract(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Return float32 features of shape (num_mel_bins, frames) for a mono float32 clip.

        Errors: ValueError when `sampling_rate` differs from `self.sampling_rate` (the caller converts) or when
        the clip is longer than the window (no silent truncation).
        """
        ...


class LabelTokenizer(Protocol):
    """Turns text into label ids and back. `max_label_tokens` is the decoder limit of the model."""

    max_label_tokens: int

    def encode_label(self, text: str) -> list[int]:
        """Return the full training label for `text`: the prefix tokens (start, language, task, no-timestamps),
        the text tokens and the end token, exactly as the model is trained."""
        ...

    def count_tokens(self, text: str) -> int:
        """Return the number of text tokens of `text` WITHOUT any special token."""
        ...

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Return the text of `ids`."""
        ...


@dataclass(frozen=True)
class ModelInfo:
    """What was loaded. Identification is by name and path only (no weight hash).

    `source_config_ref` links to the config or config-diff file of the experiment that produced the model,
    read from `pipeline_model_info.json` in the folder when that file exists; None for a downloaded base model.
    """

    name: str
    path: Path
    num_parameters: int
    dtype: str
    num_mel_bins: int
    language: str
    task: str
    source_config_ref: str | None


@dataclass(frozen=True)
class LoadedModel:
    """The result of `ModelLoader.load()`: the model (opaque), its feature extractor, its tokenizer and a description."""

    model: object
    feature_extractor: FeatureExtractor
    tokenizer: LabelTokenizer
    info: ModelInfo


class ModelLoader(Protocol):
    """Anything that can load a model. Implementation: HfWhisperLoader."""

    def load(self) -> LoadedModel:
        """Load the model. Every call returns independent objects. Raises ModelLoaderError."""
        ...
