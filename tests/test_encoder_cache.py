"""Tests of the encoder-output cache (DECISIONS 89): `features.encoder_cache.*`, offline, on the stored tiny test model and tiny synthetic wav files.

How to run (project root, virtual environment active):
    python -m unittest tests.test_encoder_cache -v
What is proved: the cache changes nothing in the results (identical Val loss, identical training history, identical decoded tokens with float32), every refusal fires with a settings error,
a stopped build continues, a changed key never reuses a file, and the default (off) behaves as before.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import soundfile
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stt.config import ConfigError  # noqa: E402
from stt.datasource import AudioSample  # noqa: E402
from stt.evaluator.hf_transcriber import HfTranscriber  # noqa: E402
from stt.evaluator.types import DecodeSettings  # noqa: E402
from stt.featurepipeline import FeaturePipeline, FeaturePipelineError, FeaturesConfig  # noqa: E402
from stt.featurepipeline.encoder_cache import EncoderCache, bytes_per_clip, check_disk  # noqa: E402
from stt.modelloader import LoadedModel, ModelConfig, create_loader  # noqa: E402
from stt.textnormalizer import NormalizedSample  # noqa: E402
from stt.trainercore import TrainConfig, TrainerConfigError, TrainerCore, TrainRequest, hyperparams_from_config  # noqa: E402

TINY = ROOT / "tests" / "assets" / "tiny_whisper"


def fresh_model() -> LoadedModel:
    return create_loader(ModelConfig(path=TINY)).load()


def features_config(directory: Path | None, dtype: str = "float32", enabled: bool = True) -> FeaturesConfig:
    values: dict[str, Any] = {"encoder_cache": {"enabled": enabled, "dtype": dtype}}
    if directory is not None:
        values["encoder_cache"]["dir"] = str(directory)
    return FeaturesConfig.model_validate(values)


class CacheTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.cache_dir = self.tmp / "cache"

    def samples(self, count: int, prefix: str = "s") -> list[NormalizedSample]:
        out = []
        for i in range(count):
            path = self.tmp / f"{prefix}{i}.wav"
            t = np.arange(int(1.5 * 16000)) / 16000
            soundfile.write(str(path), (0.3 * np.sin(2 * np.pi * (300 + 40 * i) * t)).astype("float32"), 16000)
            text = f"שלום עולם {i} מה שלומך"
            out.append(NormalizedSample(AudioSample(f"{prefix}{i}", path, "raw " + text, "g", 1.5, None, {}), text, text, None, ()))
        return out

    def pipeline(self, loaded: LoadedModel | None = None, dtype: str = "float32", freeze: str | None = "freeze_encoder", enabled: bool = True) -> FeaturePipeline:
        return FeaturePipeline(features_config(self.cache_dir, dtype, enabled), loaded or fresh_model(), seed=1, freeze_policy=freeze)


class ConfigTests(unittest.TestCase):
    def test_the_defaults_change_nothing(self) -> None:
        cfg = FeaturesConfig().encoder_cache
        self.assertEqual((cfg.enabled, cfg.dir, cfg.dtype), (False, None, "float16"))

    def test_a_folder_is_required_when_enabled(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            features_config(None)
        self.assertIn("features.encoder_cache.dir", str(ctx.exception))

    def test_an_unknown_dtype_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            features_config(Path("x"), dtype="bfloat16")


class IdenticalResultsTests(CacheTestCase):
    def batch(self, dataset: Any, collate: Any, indices: list[int]) -> dict[str, Any]:
        return dict(collate([dataset[i] for i in indices]))

    def test_the_default_off_gives_mel_examples_and_creates_no_folder(self) -> None:
        samples = self.samples(2)
        result = self.pipeline(enabled=False).build(samples, "val")
        self.assertFalse(result.dataset.uses_encoder_cache)  # type: ignore[attr-defined]
        self.assertIsNone(result.dataset[0].encoder_output)
        self.assertEqual(result.dataset[0].input_features.shape, (80, 3000))
        self.assertFalse(self.cache_dir.exists())

    def test_float32_cache_gives_a_bit_identical_val_loss_and_identical_decoded_tokens(self) -> None:
        samples = self.samples(3)
        loaded = fresh_model()
        off = FeaturePipeline(FeaturesConfig(), loaded, seed=1).build(samples, "val")
        on = self.pipeline(loaded).build(samples, "val")
        self.assertTrue(on.dataset.uses_encoder_cache)  # type: ignore[attr-defined]
        example = on.dataset[0]
        self.assertEqual((example.encoder_output.shape, example.encoder_output.dtype), ((1500, 4), np.float32))  # type: ignore[union-attr]
        collate = FeaturePipeline(FeaturesConfig(), loaded).build_collator()
        model: Any = loaded.model
        model.eval()
        with torch.no_grad():
            for i in range(3):
                b_off = self.batch(off.dataset, collate, [i])
                b_on = self.batch(on.dataset, collate, [i])
                self.assertIn("encoder_outputs", b_on)
                self.assertNotIn("input_features", b_on)
                self.assertTrue(torch.equal(model(**b_off).loss, model(**b_on).loss), "the Val loss must be bit-identical with a float32 cache")
        settings = DecodeSettings(1, 30, 2)
        tokens_off = HfTranscriber().transcribe(loaded, off.dataset, settings)
        tokens_on = HfTranscriber().transcribe(loaded, on.dataset, settings)
        self.assertEqual([(h.sample_id, h.text, h.hit_length_cap) for h in tokens_off], [(h.sample_id, h.text, h.hit_length_cap) for h in tokens_on])

    def test_float16_storage_changes_the_loss_only_a_little(self) -> None:
        samples = self.samples(2)
        loaded = fresh_model()
        off = FeaturePipeline(FeaturesConfig(), loaded, seed=1).build(samples, "val")
        on = self.pipeline(loaded, dtype="float16").build(samples, "val")
        collate = FeaturePipeline(FeaturesConfig(), loaded).build_collator()
        model: Any = loaded.model
        model.eval()
        with torch.no_grad():
            a = float(model(**self.batch(off.dataset, collate, [0])).loss)
            b = float(model(**self.batch(on.dataset, collate, [0])).loss)
        self.assertLess(abs(a - b) / abs(a), 2e-3)

    def test_the_training_history_is_identical_with_and_without_the_cache(self) -> None:
        train, val = self.samples(4, "t"), self.samples(2, "v")
        histories = []
        for use_cache in (False, True):
            loaded = fresh_model()
            pipe = self.pipeline(loaded) if use_cache else FeaturePipeline(FeaturesConfig(), loaded, seed=1)
            config = TrainConfig.model_validate(
                {"freeze_policy": "freeze_encoder", "cpu_threads": 1, "log_every_steps": 1, "eval_every_steps": 2, "save_every_steps": 4, "sinks": ["console"],
                 "hyperparams": {"max_steps": 4, "batch_size": 2, "grad_accum_steps": 1, "warmup_steps": 0, "learning_rate": 1e-3}}
            )
            request = TrainRequest(loaded, pipe.build(train, "train").dataset, pipe.build(val, "val").dataset, pipe.build_collator(), hyperparams_from_config(config, 7), self.tmp / f"run_{use_cache}")
            result = TrainerCore(config).train(request)
            histories.append([(r.step, r.train_loss, r.val_loss) for r in result.history])
            if use_cache:  # the trained model (frozen encoder) has exactly the base encoder, so it finds the same cache folder
                trained = create_loader(ModelConfig(path=result.model_dir)).load()
                self.assertEqual(self.pipeline(trained)._fingerprint(), self.pipeline(fresh_model())._fingerprint())
        self.assertEqual(histories[0], histories[1])  # the same numbers, to the last digit
        self.assertTrue(any(v is not None for _, _, v in histories[1]))

    def test_a_trained_model_with_a_frozen_encoder_finds_the_same_cache_but_a_changed_encoder_does_not(self) -> None:
        base = fresh_model()
        fp_base = self.pipeline(base)._fingerprint()
        self.assertEqual(self.pipeline(fresh_model())._fingerprint(), fp_base)  # same weights, same cache
        changed = fresh_model()
        encoder: Any = changed.model.model.encoder  # type: ignore[attr-defined]
        with torch.no_grad():
            next(encoder.parameters()).add_(1e-3)
        self.assertNotEqual(self.pipeline(changed)._fingerprint(), fp_base)  # an encoder that was trained gets its own cache folder


class RefusalTests(CacheTestCase):
    def test_a_policy_other_than_freeze_encoder_is_refused_before_any_work(self) -> None:
        for policy in ("none", None):
            with self.assertRaises(ConfigError) as ctx:
                self.pipeline(freeze=policy).build(self.samples(1), "train")
            self.assertIn("train.freeze_policy", str(ctx.exception))
            self.assertIn("freeze_encoder", str(ctx.exception))
        self.assertFalse(self.cache_dir.exists())

    def test_dropout_or_layerdrop_above_zero_is_refused(self) -> None:
        for name in ("dropout", "activation_dropout", "attention_dropout", "encoder_layerdrop", "decoder_layerdrop"):
            loaded = fresh_model()
            with mock.patch.object(loaded.model.config, name, 0.1), self.assertRaises(ConfigError) as ctx:  # type: ignore[attr-defined]
                self.pipeline(loaded).build(self.samples(1), "train")
            self.assertIn(name, str(ctx.exception))
            self.assertIn("features.encoder_cache.enabled", str(ctx.exception))

    def test_an_augmentation_is_refused(self) -> None:
        loaded = fresh_model()
        with mock.patch.object(loaded.model.config, "apply_spec_augment", True), self.assertRaises(ConfigError) as ctx:  # type: ignore[attr-defined]
            self.pipeline(loaded).build(self.samples(1), "train")
        self.assertIn("apply_spec_augment", str(ctx.exception))

    def test_the_trainer_refuses_cached_datasets_without_a_frozen_encoder(self) -> None:
        pipe = self.pipeline()
        data = pipe.build(self.samples(2), "train").dataset
        config = TrainConfig.model_validate({"freeze_policy": "none", "hyperparams": {"max_steps": 2, "warmup_steps": 0}})
        request = TrainRequest(fresh_model(), data, data, pipe.build_collator(), hyperparams_from_config(config, 1), self.tmp / "run")
        with self.assertRaises(TrainerConfigError) as ctx:
            TrainerCore(config).train(request)
        self.assertIn("train.freeze_policy", str(ctx.exception))
        self.assertFalse((self.tmp / "run").exists())

    def test_not_enough_disk_is_refused_with_the_numbers(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            check_disk(self.tmp, 10_000_000_000, free_bytes=2_000_000_000)
        message = str(ctx.exception)
        self.assertIn("10.00 GB", message)
        self.assertIn("2.00 GB", message)
        self.assertIn("short by 8.00 GB", message)
        usage = mock.Mock(free=1000)
        with mock.patch("shutil.disk_usage", return_value=usage), self.assertRaises(ConfigError):
            self.pipeline().build(self.samples(2), "train")
        self.assertEqual(bytes_per_clip(1500, 1280, "float16"), 3_840_000)
        self.assertEqual(bytes_per_clip(1500, 1280, "float32"), 7_680_000)

    def test_a_cache_that_disagrees_with_the_encoder_stops_the_build(self) -> None:
        import stt.featurepipeline.encoder_adapter as adapter

        with mock.patch.object(adapter, "encode", lambda model, features: np.zeros((1500, 4), dtype=np.float32) + 5.0), self.assertRaises(FeaturePipelineError) as ctx:
            self.pipeline().build(self.samples(3), "val")
        self.assertIn("self-check failed", str(ctx.exception))
        self.assertIn("Fix:", str(ctx.exception))

    def test_the_self_check_runs_once_per_cache_folder(self) -> None:
        import stt.featurepipeline.encoder_adapter as adapter

        samples = self.samples(3)
        self.pipeline().build(samples, "val")
        with mock.patch.object(adapter, "val_losses", side_effect=AssertionError("the self-check must not run again")):
            self.pipeline().build(samples, "val")


class BuildBehaviourTests(CacheTestCase):
    def test_a_stopped_build_continues_and_a_partial_file_is_never_read(self) -> None:
        import stt.featurepipeline.encoder_adapter as adapter

        samples = self.samples(5)
        real = adapter.encode
        calls = {"n": 0}

        def stop_after_two(model: Any, features: np.ndarray) -> np.ndarray:
            calls["n"] += 1
            if calls["n"] == 3:
                raise KeyboardInterrupt
            return real(model, features)

        with mock.patch.object(adapter, "encode", stop_after_two), self.assertRaises(KeyboardInterrupt):
            self.pipeline().build(samples, "train")
        folder = next(self.cache_dir.glob("enc_*"))
        self.assertEqual(len(list(folder.glob("*.npy"))), 2)  # the two finished clips are kept
        (folder / "deadbeef.123.tmp").write_bytes(b"half a file")  # a leftover temporary file of a killed process
        calls["n"] = 0
        counting = mock.Mock(side_effect=real)
        with mock.patch.object(adapter, "encode", counting):
            result = self.pipeline().build(samples, "train")
        self.assertEqual(counting.call_count, 3)  # only the missing clips were computed
        self.assertEqual(len(result.dataset), 5)
        self.assertTrue(all(result.dataset[i].encoder_output is not None for i in range(5)))

    def test_a_damaged_file_is_not_used_and_says_so(self) -> None:
        samples = self.samples(2)
        result = self.pipeline().build(samples, "train")
        folder = next(self.cache_dir.glob("enc_*"))
        victim = sorted(folder.glob("*.npy"))[0]
        victim.write_bytes(b"not an npy file")
        with self.assertRaises(FeaturePipelineError):
            for i in range(len(result.dataset)):
                result.dataset[i]

    def test_a_changed_key_never_reuses_a_file(self) -> None:
        samples = self.samples(2)
        loaded = fresh_model()
        self.pipeline(loaded, dtype="float32").build(samples, "train")
        self.pipeline(loaded, dtype="float16").build(samples, "train")
        self.assertEqual(len(list(self.cache_dir.glob("enc_*_float32"))), 1)
        self.assertEqual(len(list(self.cache_dir.glob("enc_*_float16"))), 1)  # another dtype, another folder
        cache = EncoderCache(self.cache_dir, "f" * 64, "float32")
        path = samples[0].sample.audio_path
        before = cache.key("s0", path)
        self.assertEqual(before, cache.key("s0", path))
        path.touch()  # the audio file changed
        import os

        os.utime(path, ns=(1, 1))
        self.assertNotEqual(before, cache.key("s0", path))
        self.assertNotEqual(cache.key("s0", path), cache.key("other", path))  # another sample id
        self.assertNotEqual(EncoderCache(self.cache_dir, "e" * 64, "float32").key("s0", path), cache.key("s0", path))  # another encoder

    def test_progress_is_logged(self) -> None:
        with self.assertLogs("stt.featurepipeline", level="INFO") as logs:
            self.pipeline().build(self.samples(2), "train")
        text = "\n".join(logs.output)
        self.assertIn("2 of 2 clips are missing", text)
        self.assertIn("clips done", text)
        self.assertIn("self-check passed", text)


if __name__ == "__main__":
    unittest.main()
