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

import hashlib
import logging
import random
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from stt.config import ConfigError
from stt.featurepipeline.cache import FeatureCache, extractor_signature
from stt.featurepipeline.config import FeaturesConfig
from stt.featurepipeline.dataset import Item, LazyFeatureDataset
from stt.featurepipeline.encoder_cache import SELF_CHECK_CLIPS, SELF_CHECK_TOLERANCE, EncoderCache, Progress, bytes_per_clip, check_disk
from stt.featurepipeline.errors import AudioReadError, FeaturePipelineError
from stt.featurepipeline.types import AudioReader, FeatureExample, FeatureReport, FeatureResult, LengthStats
from stt.modelloader import LoadedModel
from stt.textnormalizer import NormalizedSample

WINDOW_TOLERANCE_SEC = 1e-3
ENCODER_CACHE_CONFIG_LIMITS = ("dropout", "activation_dropout", "attention_dropout", "encoder_layerdrop", "decoder_layerdrop")  # every one must be 0 for the cache to be exact

log = logging.getLogger("stt.featurepipeline")


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

    def __init__(
        self, config: FeaturesConfig, model: LoadedModel, seed: int = 0, audio: AudioReader | None = None, freeze_policy: str | None = None
    ) -> None:
        self._config = config
        self._model = model
        self._seed = seed
        self._audio = audio
        self._freeze_policy = freeze_policy  # `train.freeze_policy`; only the encoder cache looks at it (it needs `freeze_encoder`)
        self._encoder_fingerprint: str | None = None

    def build(self, samples: Sequence[NormalizedSample], split: str) -> FeatureResult:
        """Check the samples, tokenize the labels, and return the lazy dataset and the report."""
        extractor = self._model.feature_extractor
        tokenizer = self._model.tokenizer
        self._check_encoder_cache_settings()  # before any work: the refusals of the encoder cache cost nothing here and hours later
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
        encoder_cache = self._fill_encoder_cache(dataset, split)
        if encoder_cache is not None:
            dataset = LazyFeatureDataset(items, extractor, reader, cache, encoder_cache)
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

    def _check_encoder_cache_settings(self) -> None:
        """The refusals of the encoder cache (settings errors, exit code 2): a frozen encoder, no dropout or layerdrop, no augmentation, a Whisper model."""
        cfg = self._config.encoder_cache
        if not cfg.enabled:
            return
        if self._freeze_policy != "freeze_encoder":
            raise ConfigError(
                f"features.encoder_cache.enabled is true but train.freeze_policy is '{self._freeze_policy}'. The cache stores the output of the encoder, which is only valid when the encoder is NOT trained. "
                "Fix: set train.freeze_policy to freeze_encoder, or set features.encoder_cache.enabled to false."
            )
        native: Any = self._model.model
        if not hasattr(getattr(native, "model", None), "encoder"):
            raise ConfigError("features.encoder_cache.enabled needs a Whisper model (an object with model.encoder). Fix: use a Whisper model or set features.encoder_cache.enabled to false.")
        config = getattr(native, "config", None)
        active = {name: getattr(config, name, 0.0) for name in ENCODER_CACHE_CONFIG_LIMITS if float(getattr(config, name, 0.0) or 0.0) > 0.0}
        if active:
            raise ConfigError(
                f"features.encoder_cache.enabled is true but the model config has {', '.join(f'{k} {v}' for k, v in active.items())}. With dropout or layerdrop the encoder output is not the same at every step, "
                "so a cache would silently change the training. Fix: use a model with these values at 0 (the turbo model has), or set features.encoder_cache.enabled to false."
            )
        if bool(getattr(config, "apply_spec_augment", False)):
            raise ConfigError(
                "features.encoder_cache.enabled is true but the model config has apply_spec_augment true (an augmentation). A cache fixes the encoder output of each clip, so no augmentation can be used. "
                "Fix: set apply_spec_augment to false in the model's config.json, or set features.encoder_cache.enabled to false."
            )

    def _fingerprint(self) -> str:
        """Hash of the base encoder weights, the feature extractor and the downmix (computed once per pipeline object)."""
        if self._encoder_fingerprint is None:
            from stt.featurepipeline import encoder_adapter  # torch is imported here, only when the encoder cache is on

            weights = encoder_adapter.encoder_weights_fingerprint(self._model.model)
            extractor = extractor_signature(self._model.feature_extractor, self._config.downmix)
            self._encoder_fingerprint = hashlib.sha256(f"{weights}|{extractor}".encode()).hexdigest()
        return self._encoder_fingerprint

    def _fill_encoder_cache(self, dataset: LazyFeatureDataset, split: str) -> EncoderCache | None:
        """Make sure every clip of `dataset` has its encoder output on disk (atomic files, resumable, with progress and a disk check) and run the one-time self-check."""
        cfg = self._config.encoder_cache
        if not cfg.enabled or cfg.dir is None:
            return None
        from stt.featurepipeline import encoder_adapter

        native: Any = self._model.model
        cache = EncoderCache(cfg.dir, self._fingerprint(), cfg.dtype)
        wanted = [(item, cache.key(item.sample_id, item.audio_path)) for item in dataset.items]
        missing = [(item, key) for item, key in wanted if not cache.has(key)]
        if missing:
            frames, width = encoder_adapter.encoder_output_shape(native)
            needed = len(missing) * bytes_per_clip(frames, width, cfg.dtype)
            check_disk(cfg.dir, needed)
            log.info("Encoder cache (%s): %d of %d clips are missing, about %.2f GB in %s; computing them once.", split, len(missing), len(wanted), needed / 1e9, cache.folder)
            progress = Progress(len(missing), time.monotonic)
            for done, (item, key) in enumerate(missing, 1):
                cache.put(key, encoder_adapter.encode(native, dataset.compute_features(item)))
                progress.update(done)
        else:
            log.info("Encoder cache (%s): all %d clips are in %s.", split, len(wanted), cache.folder)
        if not cache.self_check_done():
            self._self_check(dataset, cache, native)
        return cache

    def _self_check(self, dataset: LazyFeatureDataset, cache: EncoderCache, native: Any) -> None:
        """Compare the Val loss through the encoder with the loss from the cached output for the first clips; stop with a clear error if they differ beyond the tolerance."""
        from stt.featurepipeline import encoder_adapter

        collate = self.build_collator()
        tolerance = SELF_CHECK_TOLERANCE[cache.dtype]
        differences: list[float] = []
        for item in dataset.items[:SELF_CHECK_CLIPS]:
            batch = collate([FeatureExample(item.sample_id, dataset.compute_features(item), list(item.labels))])
            cached = cache.get(cache.key(item.sample_id, item.audio_path))
            plain, through_cache = encoder_adapter.val_losses(native, dict(batch), cached)  # type: ignore[arg-type]
            relative = abs(plain - through_cache) / max(abs(plain), 1e-12)
            differences.append(relative)
            if relative > tolerance:
                raise FeaturePipelineError(
                    f"The encoder-cache self-check failed for sample {item.sample_id}: the Val loss is {plain:.6f} through the encoder and {through_cache:.6f} from the cache (relative difference {relative:.2e}, "
                    f"tolerance {tolerance:.0e} for {cache.dtype}). Do not use this cache. Fix: delete the folder {cache.folder}, or set features.encoder_cache.dtype to float32, "
                    "or set features.encoder_cache.enabled to false."
                )
        cache.mark_self_check(differences)
        log.info("Encoder cache self-check passed on %d clips (largest relative loss difference %.2e).", len(differences), max(differences) if differences else 0.0)

    def build_collator(self) -> Callable[[Sequence[FeatureExample]], Mapping[str, Any]]:
        """Return the batch maker (a callable from a list of FeatureExample to a dict of tensors); see stt.featurepipeline.collator.

        The Hugging Face adapter is imported here, only when a collator is wanted. The decoder start token is read
        from the model's own configuration.
        """
        from stt.featurepipeline.collator import FeatureCollator

        config = getattr(self._model.model, "config", None)
        start = getattr(config, "decoder_start_token_id", None)
        return FeatureCollator(int(start) if start is not None else None)

