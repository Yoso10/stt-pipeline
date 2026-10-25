"""Unit tests of the FeaturePipeline component (stt.featurepipeline): the acceptance tests of the C4 brief, offline.

How to run (project root, virtual environment active):
    python -m unittest tests.test_featurepipeline -v
The tests use the stored tiny test model (tests/assets/tiny_whisper: the real tokenizer and feature extractor, random
weights) and tiny synthetic wav files generated in a temporary folder. Nothing is downloaded.
"""

from __future__ import annotations

import ast
import hashlib
import sys
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import soundfile
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stt.config import ConfigError, SectionRegistry, load_config  # noqa: E402
from stt.datasource import AudioSample  # noqa: E402
from stt.featurepipeline import (  # noqa: E402
    AudioHeader,
    AudioReadError,
    FeatureCache,
    FeaturePipeline,
    FeaturePipelineError,
    FeaturesConfig,
    LoadedAudio,
    extractor_signature,
    resolve_seed,
)
from stt.featurepipeline.audio_adapter import SoundfileAudioReader  # noqa: E402
from stt.featurepipeline.collator import FeatureCollator  # noqa: E402
from stt.modelloader import LoadedModel, ModelConfig, create_loader  # noqa: E402
from stt.textnormalizer import NormalizedSample  # noqa: E402

TINY = ROOT / "tests" / "assets" / "tiny_whisper"
_MODEL: LoadedModel | None = None


def model() -> LoadedModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = create_loader(ModelConfig(path=TINY)).load()
    return _MODEL


def tone(seconds: float, rate: int, channels: int = 1, freq: float = 330.0) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    wave = (0.3 * np.sin(2 * np.pi * freq * t)).astype("float32")
    return wave if channels == 1 else np.stack([wave, 0.5 * wave][:channels], axis=1)


def conf(**overrides: Any) -> FeaturesConfig:
    return FeaturesConfig.model_validate(overrides)


class FeatureTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def wav(self, name: str, seconds: float = 2.0, rate: int = 16000, channels: int = 1) -> Path:
        path = self.tmp / name
        soundfile.write(str(path), tone(seconds, rate, channels), rate)
        return path

    def sample(self, ident: str, path: Path, text: str = "שלום עולם, מה שלומך?") -> NormalizedSample:
        audio = AudioSample(ident, path, "raw " + text, "g", 1.0, None, {})
        return NormalizedSample(audio, text, text, None, ())

    def pipeline(self, **config: Any) -> FeaturePipeline:
        return FeaturePipeline(conf(**config), model(), seed=5)


class CountingReader:
    """A reader that wraps the real one and counts calls."""

    def __init__(self) -> None:
        self.real = SoundfileAudioReader()
        self.headers = 0
        self.loads = 0

    def header(self, path: Path) -> AudioHeader:
        self.headers += 1
        return self.real.header(path)

    def load(self, path: Path, target_rate: int) -> LoadedAudio:
        self.loads += 1
        return self.real.load(path, target_rate)


class BuildTests(FeatureTestCase):
    def test_1_wav_at_the_model_rate_gives_expected_features_and_labels(self) -> None:
        sample = self.sample("a", self.wav("a.wav"))
        result = self.pipeline().build([sample], "train")
        example = result.dataset[0]
        self.assertEqual((len(result.dataset), example.sample_id), (1, "a"))
        self.assertEqual((example.input_features.shape, example.input_features.dtype), ((80, 3000), np.float32))
        self.assertEqual(example.labels, model().tokenizer.encode_label(sample.train_text))
        self.assertEqual(example.labels[:4], [50258, 50279, 50359, 50363])
        report = result.report
        self.assertEqual((report.seen, report.built, report.removed_for_label_length, report.resampled, report.downmixed), (1, 1, 0, 0, 0))
        self.assertEqual((report.limit_audio_sec, report.limit_label_tokens, report.target_rate), (30.0, 448, 16000))
        self.assertEqual(report.label_lengths.minimum, len(example.labels))

    def test_2_other_rates_are_converted_and_stereo_is_downmixed_and_both_are_counted(self) -> None:
        samples = [
            self.sample("a", self.wav("a.wav")),
            self.sample("b", self.wav("b.wav", rate=8000)),
            self.sample("c", self.wav("c.wav", rate=44100)),
            self.sample("d", self.wav("d.wav", channels=2)),
            self.sample("e", self.wav("e.wav", rate=48000, channels=2)),
        ]
        result = self.pipeline().build(samples, "train")
        self.assertEqual((result.report.resampled, result.report.downmixed), (3, 2))
        for i in range(5):
            self.assertEqual(result.dataset[i].input_features.shape, (80, 3000))
        reader = SoundfileAudioReader()
        loaded = reader.load(self.tmp / "e.wav", 16000)
        self.assertTrue(loaded.resampled and loaded.downmixed)
        self.assertEqual(loaded.samples.ndim, 1)
        self.assertAlmostEqual(len(loaded.samples) / 16000, 2.0, places=2)
        self.assertEqual(loaded.samples.dtype, np.float32)

    def test_resampling_keeps_the_signal(self) -> None:
        reader = SoundfileAudioReader()
        original = reader.load(self.wav("a.wav", 1.0, 16000), 16000).samples
        converted = reader.load(self.wav("b.wav", 1.0, 44100), 16000).samples
        self.assertEqual(len(converted), 16000)
        self.assertLess(float(np.abs(original[2000:14000] - converted[2000:14000]).max()), 0.02)

    def test_3_a_clip_longer_than_the_window_fails_with_the_count_the_longest_and_the_fix(self) -> None:
        samples = [self.sample("ok", self.wav("ok.wav")), self.sample("long1", self.wav("l1.wav", 31.0)), self.sample("long2", self.wav("l2.wav", 33.5))]
        with self.assertRaises(FeaturePipelineError) as ctx:
            self.pipeline().build(samples, "train")
        message = str(ctx.exception)
        self.assertIn("2 clip(s)", message)
        self.assertIn("long2", message)
        self.assertIn("33.50 s", message)
        self.assertIn("filter.mode: apply", message)

    def test_3_a_clip_of_exactly_the_window_passes_and_a_short_one_is_padded(self) -> None:
        samples = [self.sample("full", self.wav("full.wav", 30.0)), self.sample("short", self.wav("short.wav", 0.5))]
        result = self.pipeline().build(samples, "train")
        self.assertEqual(result.dataset[0].input_features.shape, result.dataset[1].input_features.shape)
        self.assertEqual(result.dataset[1].input_features.shape, (80, 3000))

    def test_4_an_over_limit_label_is_removed_listed_and_the_gate_fails_above_the_ratio(self) -> None:
        long_text = "שלום עולם " * 300
        self.assertGreater(len(model().tokenizer.encode_label(long_text)), 448)
        samples = [self.sample(f"s{i}", self.wav(f"s{i}.wav")) for i in range(3)] + [self.sample("big", self.wav("big.wav"), long_text)]
        with self.assertRaises(FeaturePipelineError) as ctx:
            self.pipeline().build(samples, "train")
        self.assertIn("max_dropped_ratio", str(ctx.exception))
        self.assertIn("big", str(ctx.exception))
        result = self.pipeline(max_dropped_ratio=0.3).build(samples, "train")
        self.assertEqual((result.report.built, result.report.removed_for_label_length, result.report.removed_label_ids), (3, 1, ("big",)))
        self.assertEqual(result.dataset.sample_ids, ["s0", "s1", "s2"])  # type: ignore[attr-defined]
        self.assertGreater(result.report.label_lengths.maximum, 448)

    def test_label_at_the_limit_is_kept(self) -> None:
        tok = model().tokenizer
        text = "שלום "
        while len(tok.encode_label(text + "שלום ")) <= tok.max_label_tokens:
            text += "שלום "
        self.assertLessEqual(len(tok.encode_label(text)), 448)
        result = self.pipeline().build([self.sample("a", self.wav("a.wav"), text)], "train")
        self.assertEqual(result.report.removed_for_label_length, 0)

    def test_failures_empty_text_unreadable_audio_no_samples_and_nothing_left(self) -> None:
        with self.assertRaises(FeaturePipelineError) as ctx:
            self.pipeline().build([self.sample("a", self.wav("a.wav"), "  ")], "train")
        self.assertIn("min_text_chars", str(ctx.exception))
        bad = self.tmp / "bad.wav"
        bad.write_bytes(b"not audio at all")
        with self.assertRaises(FeaturePipelineError) as ctx2:
            self.pipeline().build([self.sample("broken", bad)], "train")
        self.assertIn("broken", str(ctx2.exception))
        self.assertIn("bad.wav", str(ctx2.exception))
        with self.assertRaises(FeaturePipelineError):
            self.pipeline().build([self.sample("gone", self.tmp / "missing.wav")], "train")
        with self.assertRaises(FeaturePipelineError):
            self.pipeline().build([], "train")
        with self.assertRaises(FeaturePipelineError) as ctx3:
            self.pipeline(max_dropped_ratio=1.0).build([self.sample("big", self.wav("big.wav"), "שלום עולם " * 300)], "train")
        self.assertIn("No examples are left", str(ctx3.exception))

    def test_a_missing_audio_library_fails_naming_it(self) -> None:
        with mock.patch.dict(sys.modules, {"stt.featurepipeline.audio_adapter": None}):
            with self.assertRaises(FeaturePipelineError) as ctx:
                FeaturePipeline(conf(), model()).build([self.sample("a", self.wav("a.wav"))], "train")
        self.assertIn("audio", str(ctx.exception))
        self.assertIn("O9", str(ctx.exception))

    def test_everything_numeric_is_read_from_the_model_not_hard_coded(self) -> None:
        class Small:
            sampling_rate = 8000
            window_seconds = 2.0
            num_mel_bins = 4

            def extract(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
                if len(audio) > 2.0 * sampling_rate:
                    raise ValueError("too long")
                return np.zeros((4, 10), dtype=np.float32)

        loaded = LoadedModel(model().model, Small(), model().tokenizer, model().info)
        pipeline = FeaturePipeline(conf(), loaded)
        result = pipeline.build([self.sample("a", self.wav("a.wav", 1.0, 16000))], "train")
        self.assertEqual((result.report.resampled, result.report.target_rate, result.report.limit_audio_sec), (1, 8000, 2.0))
        with self.assertRaises(FeaturePipelineError) as ctx:
            pipeline.build([self.sample("b", self.wav("b.wav", 3.0))], "train")
        self.assertIn("2 s", str(ctx.exception))

    def test_8_input_audio_files_are_byte_identical_afterwards(self) -> None:
        samples = [self.sample("a", self.wav("a.wav")), self.sample("b", self.wav("b.wav", rate=44100, channels=2))]
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.tmp.iterdir()}
        result = self.pipeline().build(samples, "train")
        _ = [result.dataset[i] for i in range(2)]
        self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.tmp.iterdir()})


class LazyAndOrderTests(FeatureTestCase):
    def test_nothing_is_decoded_until_an_item_is_read(self) -> None:
        reader = CountingReader()
        samples = [self.sample(f"s{i}", self.wav(f"s{i}.wav")) for i in range(3)]
        result = FeaturePipeline(conf(), model(), audio=reader).build(samples, "train")  # type: ignore[arg-type]
        self.assertEqual((reader.headers, reader.loads), (3, 0))
        result.dataset[1]
        self.assertEqual(reader.loads, 1)

    def test_indexing_and_errors_at_read_time(self) -> None:
        samples = [self.sample("a", self.wav("a.wav")), self.sample("b", self.wav("b.wav"))]
        dataset = self.pipeline_for(samples)
        self.assertEqual(dataset[-1].sample_id, "b")
        with self.assertRaises(IndexError):
            dataset[2]
        (self.tmp / "b.wav").unlink()
        with self.assertRaises(FeaturePipelineError) as ctx:
            dataset[1]
        self.assertIn("sample b", str(ctx.exception))

    def pipeline_for(self, samples: Sequence[NormalizedSample]) -> Any:
        return FeaturePipeline(conf(), model()).build(samples, "train").dataset

    def test_7_shuffle_uses_the_seed_and_is_off_by_default(self) -> None:
        samples = [self.sample(f"s{i:02d}", self.wav(f"s{i:02d}.wav", 0.3)) for i in range(12)]
        plain = FeaturePipeline(conf(), model(), seed=1).build(samples, "t").dataset.sample_ids  # type: ignore[attr-defined]
        self.assertEqual(plain, [f"s{i:02d}" for i in range(12)])
        a = FeaturePipeline(conf(shuffle=True), model(), seed=1).build(samples, "t").dataset.sample_ids  # type: ignore[attr-defined]
        b = FeaturePipeline(conf(shuffle=True), model(), seed=1).build(samples, "t").dataset.sample_ids  # type: ignore[attr-defined]
        c = FeaturePipeline(conf(shuffle=True), model(), seed=2).build(samples, "t").dataset.sample_ids  # type: ignore[attr-defined]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, plain)
        self.assertEqual(sorted(a), plain)

    def test_seed_falls_back_to_the_runtime_seed(self) -> None:
        self.assertEqual((resolve_seed(conf(), 42), resolve_seed(conf(seed=7), 42)), (42, 7))


class CollatorTests(FeatureTestCase):
    def test_5_labels_are_padded_with_minus_100_features_stack_and_order_is_kept(self) -> None:
        samples = [self.sample("a", self.wav("a.wav"), "שלום"), self.sample("b", self.wav("b.wav"), "שלום עולם, מה שלומך היום?"), self.sample("c", self.wav("c.wav"), "כן")]
        pipeline = FeaturePipeline(conf(), model())
        result = pipeline.build(samples, "train")
        collate = pipeline.build_collator()
        examples = [result.dataset[i] for i in range(3)]
        batch = collate(examples)
        self.assertEqual(set(batch), {"input_features", "labels"})
        self.assertEqual((batch["input_features"].shape, batch["input_features"].dtype), ((3, 80, 3000), torch.float32))
        labels = batch["labels"]
        self.assertEqual(labels.dtype, torch.long)
        self.assertEqual(labels.shape[1], max(len(e.labels) for e in examples) - 1)  # the start token is removed
        for row, example in enumerate(examples):
            expected = example.labels[1:]
            self.assertEqual(labels[row, : len(expected)].tolist(), expected)
            self.assertTrue((labels[row, len(expected) :] == -100).all())
            self.assertTrue(np.array_equal(batch["input_features"][row].numpy(), example.input_features))
        self.assertEqual(labels[0, 0].item(), 50279)  # <|he|> is now the first label: the model adds the start token itself
        self.assertEqual(labels[2, len(examples[2].labels) - 2].item(), 50257)  # the end token is kept

    def test_the_start_token_comes_from_the_model_configuration(self) -> None:
        collator = FeatureCollator(decoder_start_token_id=7)
        from stt.featurepipeline import FeatureExample

        ok = FeatureExample("a", np.zeros((2, 3), dtype=np.float32), [7, 1, 2])
        other = FeatureExample("b", np.zeros((2, 3), dtype=np.float32), [9, 1])
        batch = collator([ok, other])
        self.assertEqual(batch["labels"].tolist(), [[1, 2], [9, 1]])
        self.assertEqual(FeatureCollator(None)([ok])["labels"].tolist(), [[7, 1, 2]])
        pipeline = FeaturePipeline(conf(), model())
        self.assertIsInstance(pipeline.build_collator(), FeatureCollator)
        self.assertEqual(pipeline.build_collator()._start, 50258)  # type: ignore[attr-defined]

    def test_collator_rejects_empty_and_mismatched_batches(self) -> None:
        from stt.featurepipeline import FeatureExample

        collator = FeatureCollator(50258)
        with self.assertRaises(ValueError):
            collator([])
        with self.assertRaises(ValueError) as ctx:
            collator([FeatureExample("a", np.zeros((2, 3), dtype=np.float32), [1]), FeatureExample("b", np.zeros((2, 4), dtype=np.float32), [1])])
        self.assertIn("same shape", str(ctx.exception))

    def test_a_batch_runs_through_the_model_and_gives_a_finite_loss(self) -> None:
        sample = self.sample("a", self.wav("a.wav"))
        pipeline = FeaturePipeline(conf(), model())
        batch = pipeline.build_collator()([pipeline.build([sample], "train").dataset[0]])
        with torch.no_grad():
            out = model().model(**batch)  # type: ignore[operator]
        self.assertTrue(torch.isfinite(out.loss))


class CacheTests(FeatureTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.cache_dir = self.tmp / "cache"
        self.samples = [self.sample(f"s{i}", self.wav(f"s{i}.wav", 0.5 + i * 0.1)) for i in range(4)]

    def build(self, reader: CountingReader | None = None, **config: Any) -> Any:
        config.setdefault("cache", {"enabled": True, "dir": str(self.cache_dir)})
        return FeaturePipeline(conf(**config), model(), audio=reader).build(self.samples, "train")  # type: ignore[arg-type]

    def test_6_a_second_run_reuses_the_files_and_gives_the_same_features(self) -> None:
        first_reader = CountingReader()
        first = self.build(first_reader)
        self.assertEqual((first.report.cache_computed, first.report.cache_reused, first_reader.loads), (4, 0, 4))
        second_reader = CountingReader()
        second = self.build(second_reader)
        self.assertEqual((second.report.cache_computed, second.report.cache_reused, second_reader.loads), (0, 4, 0))
        for i in range(4):
            self.assertTrue(np.array_equal(first.dataset[i].input_features, second.dataset[i].input_features))
        self.assertEqual(second_reader.loads, 0)  # reading items does not decode either
        self.assertEqual(len(list(self.cache_dir.glob("*/*.npy"))), 4)

    def test_6_changed_extractor_settings_invalidate_the_cache(self) -> None:
        self.build()

        class Other:
            sampling_rate = 16000
            window_seconds = 30.0
            num_mel_bins = 80

            def extract(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
                return np.ones((80, 3000), dtype=np.float32) * float(len(audio) > 0)

        base = model().feature_extractor
        self.assertNotEqual(extractor_signature(base, "mean"), extractor_signature(Other(), "mean"))  # same numbers, different features
        self.assertNotEqual(extractor_signature(base, "mean"), extractor_signature(base, "other"))
        loaded = LoadedModel(model().model, Other(), model().tokenizer, model().info)
        reader = CountingReader()
        result = FeaturePipeline(conf(cache={"enabled": True, "dir": str(self.cache_dir)}), loaded, audio=reader).build(self.samples, "train")  # type: ignore[arg-type]
        self.assertEqual((result.report.cache_computed, result.report.cache_reused), (4, 0))
        self.assertEqual(len(list(self.cache_dir.glob("*/signature.json"))), 2)  # the old cache is kept, never reused

    def test_a_changed_audio_file_is_not_served_from_a_stale_cache(self) -> None:
        self.build()
        soundfile.write(str(self.tmp / "s0.wav"), tone(1.7, 16000, freq=880.0), 16000)
        result = self.build(CountingReader())
        self.assertEqual((result.report.cache_computed, result.report.cache_reused), (1, 3))

    def test_a_damaged_cache_file_is_rebuilt(self) -> None:
        self.build()
        victim = sorted(self.cache_dir.glob("*/*.npy"))[0]
        victim.write_bytes(b"garbage")
        result = self.build(CountingReader())
        self.assertEqual(result.report.cache_computed, 1)
        self.assertEqual(result.dataset[0].input_features.shape, (80, 3000))

    def test_a_cache_folder_that_cannot_be_written_is_skipped_not_an_error(self) -> None:
        blocker = self.tmp / "blocked"
        blocker.write_text("a file, not a folder", encoding="utf-8")
        result = self.build(cache={"enabled": True, "dir": str(blocker / "inside")})
        self.assertEqual(result.dataset[0].input_features.shape, (80, 3000))

    def test_threads_give_the_same_cache_as_one_at_a_time(self) -> None:
        serial = self.build()
        threaded = self.build(cache={"enabled": True, "dir": str(self.tmp / "cache2")}, num_workers=3)
        self.assertEqual(threaded.report.cache_computed, 4)
        for i in range(4):
            self.assertTrue(np.array_equal(serial.dataset[i].input_features, threaded.dataset[i].input_features))

    def test_cache_files_use_the_documented_layout(self) -> None:
        self.build()
        signature = extractor_signature(model().feature_extractor, "mean")
        cache = FeatureCache(self.cache_dir, signature)
        self.assertTrue(cache.folder.is_dir())
        self.assertEqual(cache.folder.name, signature[:16])
        self.assertFalse(list(cache.folder.glob("*.tmp.npy")))


class ConfigAndRulesTests(unittest.TestCase):
    def load(self, *overrides: str) -> FeaturesConfig:
        config = load_config(overrides=list(overrides), registry=SectionRegistry({"features": FeaturesConfig}), environ={}, env_file=None, target_environ={})
        return config.section("features", FeaturesConfig)

    def test_defaults_match_the_brief(self) -> None:
        c = self.load()
        self.assertEqual((c.num_workers, c.cache.enabled, c.cache.dir, c.max_dropped_ratio, c.downmix, c.shuffle, c.seed), (0, False, None, 0.05, "mean", False, None))

    def test_invalid_values_fail_at_config_validation(self) -> None:
        for bad, expected in (
            ("features.cache.enabled=true", "features.cache.dir"),
            ("features.num_workers=-1", "num_workers"),
            ("features.max_dropped_ratio=2", "max_dropped_ratio"),
            ("features.downmix=max", "downmix"),
            ("features.nonsense=1", "Unknown key"),
        ):
            with self.assertRaises(ConfigError, msg=bad) as ctx:
                self.load(bad)
            self.assertIn(expected, str(ctx.exception))
        self.assertEqual(self.load("features.cache.enabled=true", "features.cache.dir=somewhere").cache.dir, Path("somewhere"))

    def test_shipped_example_config_equals_the_defaults(self) -> None:
        config = load_config(ROOT / "configs" / "features_default.yaml", registry=SectionRegistry({"features": FeaturesConfig}), environ={}, env_file=None, target_environ={})
        self.assertEqual(config.section("features", FeaturesConfig), FeaturesConfig())

    def test_9_libraries_are_imported_only_inside_adapter_modules_and_there_is_no_network(self) -> None:
        allowed = {"torch": {"collator.py", "encoder_adapter.py"}, "soundfile": {"audio_adapter.py"}, "scipy": {"audio_adapter.py"}}
        forbidden = {"requests", "urllib3", "httpx", "huggingface_hub", "socket", "tools", "datasets", "transformers"}
        for path in (ROOT / "src" / "stt" / "featurepipeline").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    top = name.split(".")[0]
                    self.assertNotIn(top, forbidden, f"{path} imports {name}")
                    if top in allowed:
                        self.assertIn(path.name, allowed[top], f"{path} imports {name}")

    def test_the_pipeline_does_not_import_torch_at_all_until_a_collator_is_requested(self) -> None:
        import subprocess

        code = (
            "import sys; sys.path.insert(0, 'src'); import stt.featurepipeline; "
            "print([m for m in ('torch', 'transformers', 'scipy') if m in sys.modules])"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "[]")

    def test_the_audio_adapter_reports_unreadable_files(self) -> None:
        reader = SoundfileAudioReader()
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.wav"
            bad.write_bytes(b"xx")
            with self.assertRaises(AudioReadError):
                reader.header(bad)
            with self.assertRaises(AudioReadError):
                reader.load(bad, 16000)
            with self.assertRaises(AudioReadError):
                reader.header(Path(tmp) / "missing.wav")


if __name__ == "__main__":
    unittest.main()
