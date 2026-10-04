"""FeaturePipeline: turns the kept samples of one split into a lazy, model-ready dataset and reports what it did.

How to run: this module is not a script; the composition root creates the pipeline. Example (PowerShell, project root;
`loaded` is a LoadedModel from stt.modelloader and `kept` the kept NormalizedSamples of one split; the full working
example is in docs/COMPONENTS.md, C4):
    $env:PYTHONPATH = "src"
    python -c "from stt.featurepipeline import FeaturePipeline, FeaturesConfig; print(FeaturesConfig())"
Rules: the sampling rate, the audio window, the mel bins and the label limit are READ from the loaded model, never
hard-coded; audio at another rate is converted (and counted); stereo is averaged to mono (and counted); audio longer
than the window is NEVER truncated: the run FAILS (the SampleFilter must remove such clips first); labels come from
`train_text`; labels over the limit are removed and listed, and a gate fails above `features.max_dropped_ratio`;
the input audio files are only read.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from stt.featurepipeline.cache import FeatureCache, extractor_signature
from stt.featurepipeline.config import FeaturesConfig
from stt.featurepipeline.dataset import Item, LazyFeatureDataset
from stt.featurepipeline.errors import AudioReadError, FeaturePipelineError
from stt.featurepipeline.types import AudioReader, FeatureExample, FeatureReport, FeatureResult, LengthStats
from stt.modelloader import LoadedModel
from stt.textnormalizer import NormalizedSample

WINDOW_TOLERANCE_SEC = 1e-3


def _default_reader() -> AudioReader:
    try:
        from stt.featurepipeline.audio_adapter import SoundfileAudioReader
    except ImportError as exc:
        raise FeaturePipelineError(
            f"A library for audio is missing ({exc.name}). Repeat the home install test (O9) and install from requirements.lock."
        ) from exc
    return SoundfileAudioReader()


class FeaturePipeline:
    """Builds the feature dataset of one split.

    Inputs: the `features` config, a LoadedModel (only its FeatureExtractor and LabelTokenizer are used here), the seed
    (used only when `features.shuffle` is true) and optionally an AudioReader (default: soundfile and scipy).
    Errors: FeaturePipelineError for audio longer than the window, an empty `train_text`, an unreadable audio file, a
    missing audio library, zero examples, or too many labels over the limit.
    """

    def __init__(self, config: FeaturesConfig, model: LoadedModel, seed: int = 0, audio: AudioReader | None = None) -> None:
        self._config = config
        self._model = model
        self._seed = seed
        self._audio = audio

    def build(self, samples: Sequence[NormalizedSample], split: str) -> FeatureResult:
        """Check the samples, tokenize the labels, and return the lazy dataset and the report."""
        extractor = self._model.feature_extractor
        tokenizer = self._model.tokenizer
        reader = self._audio or _default_reader()
        if not samples:
            raise FeaturePipelineError(f"There are no samples in split '{split}' to build features from.")
        empty = [s.sample.id for s in samples if not s.train_text.strip()]
        if empty:
            raise FeaturePipelineError(
                f"{len(empty)} sample(s) of split '{split}' have an empty train_text (for example {', '.join(empty[:3])}). "
                "Remove them with the SampleFilter rule min_text_chars (filter.mode: apply)."
            )
        resampled = downmixed = 0
        too_long: list[tuple[float, str]] = []
        for s in samples:
            try:
                header = reader.header(s.sample.audio_path)
            except AudioReadError as exc:
                raise FeaturePipelineError(f"Cannot read the audio of sample {s.sample.id} ({s.sample.audio_path}): {exc}") from exc
            resampled += header.sample_rate != extractor.sampling_rate
            downmixed += header.channels > 1
            if header.duration_sec > extractor.window_seconds + WINDOW_TOLERANCE_SEC:
                too_long.append((header.duration_sec, s.sample.id))
        if too_long:
            longest = max(too_long)
            raise FeaturePipelineError(
                f"{len(too_long)} clip(s) of split '{split}' are longer than the model window of {extractor.window_seconds:g} s; "
                f"the longest is {longest[1]} with {longest[0]:.2f} s. Audio is never truncated. Fix: enable the rule "
                "filter.rules.max_duration and set filter.mode: apply, so the SampleFilter removes them first."
            )
        items: list[Item] = []
        lengths: list[int] = []
        removed: list[str] = []
        for s in samples:
            labels = tokenizer.encode_label(s.train_text)
            lengths.append(len(labels))
            if len(labels) > tokenizer.max_label_tokens:
                removed.append(s.sample.id)
            else:
                items.append(Item(s.sample.id, s.sample.audio_path, labels))
        if len(removed) / len(samples) > self._config.max_dropped_ratio:
            raise FeaturePipelineError(
                f"{len(removed)} of {len(samples)} labels of split '{split}' exceed the limit of {tokenizer.max_label_tokens} tokens "
                f"({len(removed) / len(samples):.1%}, above features.max_dropped_ratio {self._config.max_dropped_ratio:.1%}); "
                f"for example {', '.join(removed[:3])}. Read the label-length distribution and check that the timestamps are stripped; "
                "do not raise the ratio without a decision."
            )
        if not items:
            raise FeaturePipelineError(f"No examples are left in split '{split}' after the label-length check.")
        if self._config.shuffle:
            random.Random(self._seed).shuffle(items)
        cache = None
        if self._config.cache.enabled and self._config.cache.dir is not None:
            cache = FeatureCache(self._config.cache.dir, extractor_signature(extractor, self._config.downmix))
        dataset = LazyFeatureDataset(items, extractor, reader, cache)
        reused = computed = 0
        if cache is not None:
            reused, computed = dataset.warm(self._config.num_workers)
        array = np.asarray(lengths)
        report = FeatureReport(
            split=split,
            seen=len(samples),
            built=len(items),
            removed_for_label_length=len(removed),
            removed_label_ids=tuple(removed),
            resampled=resampled,
            downmixed=downmixed,
            label_lengths=LengthStats(len(lengths), int(array.min()), float(array.mean()), float(np.percentile(array, 95)), int(array.max())),
            limit_audio_sec=extractor.window_seconds,
            limit_label_tokens=tokenizer.max_label_tokens,
            target_rate=extractor.sampling_rate,
            cache_enabled=cache is not None,
            cache_reused=reused,
            cache_computed=computed,
        )
        return FeatureResult(dataset=dataset, report=report)

    def build_collator(self) -> Callable[[Sequence[FeatureExample]], Mapping[str, Any]]:
        """Return the batch maker (a callable from a list of FeatureExample to a dict of tensors); see stt.featurepipeline.collator.

        The Hugging Face adapter is imported here, only when a collator is wanted. The decoder start token is read
        from the model's own configuration.
        """
        from stt.featurepipeline.collator import FeatureCollator

        config = getattr(self._model.model, "config", None)
        start = getattr(config, "decoder_start_token_id", None)
        return FeatureCollator(int(start) if start is not None else None)

