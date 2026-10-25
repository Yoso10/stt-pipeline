"""Typed data objects of the FeaturePipeline component and its Protocols.

How to run: this module is not a script; the other modules and later components import it. Example
(PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.featurepipeline.types import FeatureExample; import numpy as np; print(FeatureExample('a', np.zeros((80, 3000), dtype='float32'), [1, 2]).input_features.shape)"

Choices (decision 38 in docs/DECISIONS.md): `input_features` is a plain float32 numpy array (mel bins x frames), not
a torch tensor, so the data type carries no framework; the padded batch is made by the collator, which lives in
the Hugging Face adapter module (the only module that imports torch).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class FeatureExample:
    """One model-ready example: `input_features` float32 (mel bins, frames) and `labels` (the full label ids).

    With the encoder cache (DECISIONS 89) `encoder_output` holds the float32 (frames, width) output of the frozen encoder and `input_features` is an empty placeholder;
    otherwise `encoder_output` is None.
    """

    sample_id: str
    input_features: np.ndarray
    labels: list[int]
    encoder_output: np.ndarray | None = None


class FeatureDataset(Protocol):
    """What the trainer receives: a plain dataset with a length and indexing (no `datasets` library)."""

    def __len__(self) -> int:
        """Number of examples."""
        ...

    def __getitem__(self, index: int) -> FeatureExample:
        """The example at `index` (computed on demand unless cached)."""
        ...


@dataclass(frozen=True)
class AudioHeader:
    """What a header says about an audio file."""

    sample_rate: int
    channels: int
    frames: int

    @property
    def duration_sec(self) -> float:
        """Duration in seconds."""
        return self.frames / self.sample_rate if self.sample_rate else 0.0


@dataclass(frozen=True)
class LoadedAudio:
    """Mono float32 audio at the requested rate and what was done to get it."""

    samples: np.ndarray
    resampled: bool
    downmixed: bool


class AudioReader(Protocol):
    """Reads audio headers and audio. The default implementation (soundfile and scipy) is an adapter module."""

    def header(self, path: Path) -> AudioHeader:
        """Read the header only. Raises AudioReadError."""
        ...

    def load(self, path: Path, target_rate: int) -> LoadedAudio:
        """Decode the file, average stereo to mono, convert to `target_rate`. Raises AudioReadError."""
        ...


@dataclass(frozen=True)
class LengthStats:
    """Label lengths in ids (prefix tokens, text tokens and the end token): min, mean, 95th percentile, max."""

    count: int
    minimum: int
    mean: float
    p95: float
    maximum: int


@dataclass(frozen=True)
class FeatureReport:
    """What was built. `removed_label_ids` lists the samples dropped because the label exceeded the limit.

    `limit_audio_sec` and `limit_label_tokens` are the limits that were applied, both read from the loaded model.
    """

    split: str
    seen: int
    built: int
    removed_for_label_length: int
    removed_label_ids: tuple[str, ...]
    resampled: int
    downmixed: int
    label_lengths: LengthStats
    limit_audio_sec: float
    limit_label_tokens: int
    target_rate: int
    cache_enabled: bool
    cache_reused: int
    cache_computed: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        """Return the report as plain text."""
        s = self.label_lengths
        lines = [
            f"split {self.split}: seen {self.seen}, built {self.built}, removed for label length {self.removed_for_label_length}",
            f"resampled to {self.target_rate} Hz: {self.resampled}, downmixed to mono: {self.downmixed}",
            f"label ids per sample: min {s.minimum}, mean {s.mean:.1f}, p95 {s.p95:.0f}, max {s.maximum} "
            f"(limit {self.limit_label_tokens}); audio limit {self.limit_audio_sec:g} s",
        ]
        if self.cache_enabled:
            lines.append(f"cache: {self.cache_reused} reused, {self.cache_computed} computed")
        lines.extend(self.notes)
        return "\n".join(lines)


@dataclass(frozen=True)
class FeatureResult:
    """The output of `FeaturePipeline.build`: the dataset and the report."""

    dataset: FeatureDataset
    report: FeatureReport


Batch = Mapping[str, object]
