"""LazyFeatureDataset: the examples are computed on demand (or read from the cache).

How to run: this module is not a script; `FeaturePipeline.build` creates the dataset. Example (PowerShell, project root;
`result` is a FeatureResult):
    $env:PYTHONPATH = "src"
    python -c "from stt.featurepipeline.dataset import LazyFeatureDataset"
Nothing is decoded until an item is read. `dataset[i]` decodes the audio, averages it to mono, converts the rate, runs
the feature extractor and returns a FeatureExample. With a cache, `warm(workers)` fills it once with a simple loop
(`workers` threads) and later reads use the files.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stt.featurepipeline.cache import FeatureCache
from stt.featurepipeline.encoder_cache import EncoderCache
from stt.featurepipeline.errors import AudioReadError, FeaturePipelineError
from stt.featurepipeline.types import AudioReader, FeatureExample
from stt.modelloader import FeatureExtractor


@dataclass(frozen=True)
class Item:
    """What the dataset needs about one example: its id, audio file and label ids."""

    sample_id: str
    audio_path: Path
    labels: list[int]


class LazyFeatureDataset:
    """A plain dataset (`__len__`, `__getitem__`) over the items, in the given order."""

    def __init__(
        self,
        items: Sequence[Item],
        extractor: FeatureExtractor,
        reader: AudioReader,
        cache: FeatureCache | None = None,
        encoder_cache: EncoderCache | None = None,
    ) -> None:
        self._items = list(items)
        self._extractor = extractor
        self._reader = reader
        self._cache = cache
        self._encoder_cache = encoder_cache

    @property
    def uses_encoder_cache(self) -> bool:
        """True when the examples carry the cached encoder output instead of the mel features (the TrainerCore refuses this unless the encoder is frozen)."""
        return self._encoder_cache is not None

    @property
    def items(self) -> list[Item]:
        """The items in dataset order."""
        return list(self._items)

    def compute_features(self, item: Item) -> np.ndarray:
        """The mel features of one item, computed from the audio (ignoring every cache). Errors: FeaturePipelineError."""
        return self._compute(item)

    def __len__(self) -> int:
        """Number of examples."""
        return len(self._items)

    def __getitem__(self, index: int) -> FeatureExample:
        """The example at `index`. Errors: IndexError, or FeaturePipelineError naming the sample when its audio cannot be used."""
        if not -len(self._items) <= index < len(self._items):
            raise IndexError(index)
        item = self._items[index]
        if self._encoder_cache is not None:
            output = self._encoder_cache.get(self._encoder_cache.key(item.sample_id, item.audio_path))
            if output is None:
                raise FeaturePipelineError(f"The encoder cache has no usable file for sample {item.sample_id} in {self._encoder_cache.folder}. Build the dataset again (the build fills the cache).")
            return FeatureExample(item.sample_id, np.zeros((0, 0), dtype=np.float32), list(item.labels), output)
        return FeatureExample(item.sample_id, self._features(item), list(item.labels))

    @property
    def sample_ids(self) -> list[str]:
        """The sample ids in dataset order."""
        return [i.sample_id for i in self._items]

    def _features(self, item: Item) -> np.ndarray:
        key = self._cache.key(item.sample_id, item.audio_path) if self._cache is not None else None
        if self._cache is not None and key is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        features = self._compute(item)
        if self._cache is not None and key is not None:
            self._cache.put(key, features)
        return features

    def _compute(self, item: Item) -> np.ndarray:
        try:
            audio = self._reader.load(item.audio_path, self._extractor.sampling_rate)
            return self._extractor.extract(audio.samples, self._extractor.sampling_rate)
        except (AudioReadError, ValueError) as exc:
            raise FeaturePipelineError(f"Cannot build the features of sample {item.sample_id} ({item.audio_path}): {exc}") from exc

    def warm(self, workers: int = 0) -> tuple[int, int]:
        """Fill the cache with every missing example. Returns (reused, computed). Needs a cache."""
        if self._cache is None:
            raise FeaturePipelineError("warm() needs the cache: set features.cache.enabled and features.cache.dir.")
        cache = self._cache

        def one(item: Item) -> bool:
            key = cache.key(item.sample_id, item.audio_path)
            if cache.has(key) and cache.get(key) is not None:
                return False
            cache.put(key, self._compute(item))
            return True

        if workers > 0:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                flags = list(pool.map(one, self._items))
        else:
            flags = [one(item) for item in self._items]
        computed = sum(flags)
        return len(flags) - computed, computed
