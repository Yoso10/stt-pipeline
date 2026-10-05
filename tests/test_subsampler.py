"""Unit tests of the Subsampler component (stt.subsampler): the acceptance tests of the C3b brief, offline, on tiny synthetic samples.

How to run (project root, virtual environment active):
    python -m unittest tests.test_subsampler -v
No audio, no model and no network are used: a sample is a NormalizedSample with a duration and a group.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stt.config import SectionRegistry, load_config  # noqa: E402
from stt.datasource import AudioSample  # noqa: E402
from stt.subsampler import (  # noqa: E402
    BUILT_IN_PRESETS,
    SizeConfig,
    SubsampleConfig,
    SubsampleConfigError,
    SubsampleError,
    SubsampleSize,
    SubsampleSpec,
    SubsampleStore,
    Subsampler,
    build_spec,
    default_registry,
    rank,
    resolve_subsample,
)
from stt.textnormalizer import NormalizedSample  # noqa: E402


def sample(ident: str, group: str, seconds: float = 60.0) -> NormalizedSample:
    audio = AudioSample(ident, Path(f"{ident}.wav"), "raw", group, seconds, None, {})
    return NormalizedSample(audio, "text " + ident, "text " + ident, None, ())


def make_splits(train_groups: int = 20, per_group: int = 5, val_groups: int = 4, test_groups: int = 4) -> dict[str, list[NormalizedSample]]:
    """Train: `train_groups` groups of `per_group` clips with different lengths (30 to 90 s); Val and Test: one-clip-per-group sets."""
    train = [sample(f"t{g:02d}_{i}", f"tg{g:02d}", 30.0 + 15.0 * i) for g in range(train_groups) for i in range(per_group)]
    return {
        "train": train,
        "val": [sample(f"v{g}", f"vg{g}") for g in range(val_groups)],
        "test": [sample(f"x{g}", f"xg{g}") for g in range(test_groups)],
    }


def spec(kind: str = "minutes", value: float | None = 60.0, strategy: str = "by_sample", scope: str = "train_only", val: str = "fixed", seed: int = 42, repeat: int = 0, minimum: int = 1) -> SubsampleSpec:
    return SubsampleSpec(SubsampleSize(kind, value), strategy, scope, val, seed, repeat, "train", "val", minimum)  # type: ignore[arg-type]


def ids(result_splits: dict[str, list[NormalizedSample]], name: str = "train") -> set[str]:
    return {n.sample.id for n in result_splits[name]}


class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.splits = make_splits()
        self.sub = Subsampler()

    def test_1_same_seed_same_subset_other_seed_other_subset_and_input_order_does_not_matter(self) -> None:
        first = self.sub.subsample(self.splits, spec())
        again = self.sub.subsample(self.splits, spec())
        self.assertEqual((ids(first.splits), first.fingerprint), (ids(again.splits), again.fingerprint))
        other = self.sub.subsample(self.splits, spec(seed=43))
        self.assertNotEqual(ids(first.splits), ids(other.splits))
        shuffled = {name: random.Random(5).sample(samples, len(samples)) for name, samples in self.splits.items()}
        reordered = self.sub.subsample(shuffled, spec())
        self.assertEqual(ids(first.splits), ids(reordered.splits))
        self.assertEqual(first.fingerprint, reordered.fingerprint)

    def test_2_nesting_a_smaller_subset_is_inside_a_larger_one_for_both_strategies(self) -> None:
        for strategy in ("by_sample", "by_group"):
            with self.subTest(strategy=strategy):
                subsets = [ids(self.sub.subsample(self.splits, spec("minutes", m, strategy)).splits) for m in (10.0, 30.0, 80.0)]
                self.assertTrue(subsets[0] < subsets[1] < subsets[2])
                counts = [ids(self.sub.subsample(self.splits, spec("samples", n, strategy)).splits) for n in (3, 12, 40)]
                self.assertTrue(counts[0] <= counts[1] <= counts[2])
                fractions = [ids(self.sub.subsample(self.splits, spec("fraction", f, strategy)).splits) for f in (0.1, 0.5)]
                self.assertTrue(fractions[0] < fractions[1])

    def test_3_sizes_in_minutes_samples_and_fraction_reach_the_target_within_one_unit_and_report_the_overshoot(self) -> None:
        total_minutes = sum(n.sample.duration_sec for n in self.splits["train"]) / 60.0
        cases = [("minutes", 30.0, "by_sample", "minutes", 30.0), ("samples", 17.0, "by_sample", "samples", 17.0), ("fraction", 0.1, "by_sample", "minutes", total_minutes * 0.1), ("minutes", 30.0, "by_group", "minutes", 30.0)]
        for kind, value, strategy, unit, requested in cases:
            with self.subTest(kind=kind, strategy=strategy):
                result = self.sub.subsample(self.splits, spec(kind, value, strategy))
                report = result.report
                self.assertEqual((report.unit, round(report.requested_value or 0, 6)), (unit, round(requested, 6)))
                self.assertGreaterEqual(report.overshoot or 0, -1e-9)  # reaches the target
                biggest_unit = 5.0 if strategy == "by_group" else 1.5  # a group (5 clips of 1 minute on average) or the longest clip, in minutes
                if unit == "samples":
                    biggest_unit = 5.0 if strategy == "by_group" else 1.0
                self.assertLessEqual(report.overshoot or 0, biggest_unit + 1e-9)  # at most one unit
                self.assertAlmostEqual((report.achieved_value or 0) - (report.requested_value or 0), report.overshoot or 0)
        self.assertIn("overshoot", self.sub.subsample(self.splits, spec()).report.summary())

    def test_4_by_group_never_splits_a_group_by_sample_can(self) -> None:
        by_group = self.sub.subsample(self.splits, spec("minutes", 30.0, "by_group"))
        chosen = by_group.splits["train"]
        for group in {n.sample.group_id for n in chosen}:
            expected = {n.sample.id for n in self.splits["train"] if n.sample.group_id == group}
            self.assertEqual({n.sample.id for n in chosen if n.sample.group_id == group}, expected)
        by_sample = self.sub.subsample(self.splits, spec("minutes", 30.0, "by_sample"))
        partly = [g for g in {n.sample.group_id for n in by_sample.splits["train"]} if len([n for n in by_sample.splits["train"] if n.sample.group_id == g]) < 5]
        self.assertTrue(partly)

    def test_5_train_only_leaves_val_and_test_identical_follow_reduces_val_and_all_splits_reduces_all_three(self) -> None:
        only = self.sub.subsample(self.splits, spec("fraction", 0.5))
        for name in ("val", "test"):
            self.assertEqual(len(only.splits[name]), len(self.splits[name]))
            self.assertTrue(all(a is b for a, b in zip(only.splits[name], self.splits[name], strict=True)))  # the same objects, the same order
        self.assertLess(len(only.splits["train"]), len(self.splits["train"]))
        follow = self.sub.subsample(self.splits, spec("fraction", 0.5, val="follow"))
        self.assertLess(len(follow.splits["val"]), len(self.splits["val"]))
        self.assertEqual(len(follow.splits["test"]), len(self.splits["test"]))  # the Test is never touched in train_only
        everything = self.sub.subsample(self.splits, spec("fraction", 0.5, "by_group", "all_splits"))
        for name in ("train", "val", "test"):
            self.assertLess(len(everything.splits[name]), len(self.splits[name]))
        self.assertTrue(all(s.reduced for s in everything.report.sizes))
        with self.assertRaises(SubsampleConfigError) as ctx:
            self.sub.subsample(self.splits, spec("minutes", 30.0, "by_group", "all_splits"))
        self.assertIn("fraction", str(ctx.exception))
        follow_minutes = self.sub.subsample(self.splits, spec("minutes", 30.0, val="follow"))
        self.assertEqual(len(follow_minutes.splits["val"]), 2)  # 30 of 100 train minutes: 30% of the 4 val minutes is 1.2, so 2 one-minute clips

    def test_6_a_size_larger_than_available_fails_with_both_numbers_and_all_returns_everything(self) -> None:
        with self.assertRaises(SubsampleError) as ctx:
            self.sub.subsample(self.splits, spec("minutes", 9999.0))
        self.assertIn("9999.00 minutes", str(ctx.exception))
        self.assertIn("exist", str(ctx.exception))
        with self.assertRaises(SubsampleError):
            self.sub.subsample(self.splits, spec("samples", 1000.0))
        everything = self.sub.subsample(self.splits, spec("all", None))
        self.assertEqual({k: ids(everything.splits, k) for k in self.splits}, {k: ids(self.splits, k) for k in self.splits})
        self.assertIsNone(everything.report.overshoot)
        self.assertTrue(everything.report.enabled)

    def test_7_fewer_than_the_minimum_samples_fails_and_an_empty_split_fails(self) -> None:
        with self.assertRaises(SubsampleError) as ctx:
            self.sub.subsample(self.splits, spec("samples", 5.0, minimum=10))
        self.assertIn("min_train_samples", str(ctx.exception))
        self.assertIn("5", str(ctx.exception))
        no_val = {**self.splits, "val": []}
        with self.assertRaises(SubsampleError) as ctx2:
            self.sub.subsample(no_val, spec("fraction", 0.5, val="follow"))
        self.assertIn("'val'", str(ctx2.exception))
        with self.assertRaises(SubsampleError):
            self.sub.subsample({**self.splits, "test": []}, spec("fraction", 0.5, "by_group", "all_splits"))

    def test_8_the_repeat_index_gives_different_subsets_of_the_same_size(self) -> None:
        draws = [ids(self.sub.subsample(self.splits, spec("samples", 20.0, repeat=r)).splits) for r in range(4)]
        self.assertEqual({len(d) for d in draws}, {20})
        self.assertEqual(len({frozenset(d) for d in draws}), 4)

    def test_a_single_group_in_train_gives_a_warning(self) -> None:
        result = self.sub.subsample(self.splits, spec("minutes", 1.0, "by_group"))
        self.assertTrue(any("ONE group" in w for w in result.report.warnings))
        self.assertIn("WARNING", result.report.summary())

    def test_unknown_strategy_and_scope_fail_listing_the_names(self) -> None:
        with self.assertRaises(SubsampleConfigError) as ctx:
            self.sub.subsample(self.splits, spec(strategy="by_speaker"))
        self.assertIn("by_group, by_sample", str(ctx.exception))
        with self.assertRaises(SubsampleConfigError) as ctx2:
            self.sub.subsample(self.splits, spec(scope="val_only"))
        self.assertIn("all_splits, train_only", str(ctx2.exception))
        with self.assertRaises(SubsampleConfigError):
            self.sub.subsample({"a": self.splits["train"]}, spec())  # no split called train

    def test_the_rank_is_stable_and_depends_on_seed_repeat_and_key(self) -> None:
        self.assertEqual(rank(1, 0, "a"), rank(1, 0, "a"))
        self.assertEqual(len({rank(1, 0, "a"), rank(2, 0, "a"), rank(1, 1, "a"), rank(1, 0, "b")}), 4)
        self.assertEqual(default_registry().names(), ["by_group", "by_sample"])

    def test_samples_are_never_modified(self) -> None:
        before = [(n, n.train_text, n.eval_text) for n in self.splits["train"]]
        self.sub.subsample(self.splits, spec("fraction", 0.3))
        self.assertEqual(before, [(n, n.train_text, n.eval_text) for n in self.splits["train"]])


class ValSliceTests(unittest.TestCase):
    """`subsample.val_samples`: a fixed-size Val slice (O78, owner approved)."""

    def setUp(self) -> None:
        self.splits = make_splits(val_groups=30, test_groups=30)
        self.sub = Subsampler()

    def slice_spec(self, size_kind: str = "all", value: float | None = None, val_samples: int | None = 10, **kw: object) -> SubsampleSpec:
        return replace(spec(size_kind, value, **kw), val_samples=val_samples)  # type: ignore[arg-type]

    def test_a_fixed_val_slice_with_the_train_split_untouched(self) -> None:
        result = self.sub.subsample(self.splits, self.slice_spec())
        self.assertEqual(len(result.splits["val"]), 10)
        self.assertTrue(all(a is b for a, b in zip(result.splits["train"], self.splits["train"], strict=True)))  # Train unchanged
        self.assertEqual(len(result.splits["test"]), len(self.splits["test"]))  # the Test is never touched
        sizes = {s.name: s.reduced for s in result.report.sizes}
        self.assertEqual(sizes, {"train": False, "val": True, "test": False})
        self.assertIn("Val slice of 10 samples", result.report.requested)

    def test_the_slice_does_not_depend_on_the_train_size_and_is_nested(self) -> None:
        a = ids(self.sub.subsample(self.splits, self.slice_spec()).splits, "val")
        b = ids(self.sub.subsample(self.splits, self.slice_spec("minutes", 30.0)).splits, "val")
        c = ids(self.sub.subsample(self.splits, self.slice_spec("fraction", 0.1)).splits, "val")
        self.assertEqual(a, b)
        self.assertEqual(a, c)  # the same Val slice for every Train size: the trials of a search are comparable
        small = ids(self.sub.subsample(self.splits, self.slice_spec(val_samples=4)).splits, "val")
        self.assertTrue(small < a)  # nested
        reduced = self.sub.subsample(self.splits, self.slice_spec("minutes", 30.0))
        self.assertLess(len(reduced.splits["train"]), len(self.splits["train"]))
        self.assertEqual(len(reduced.splits["val"]), 10)

    def test_a_slice_larger_than_val_fails_with_both_numbers_and_the_minimum_train_check_is_for_a_reduced_train_only(self) -> None:
        with self.assertRaises(SubsampleError) as ctx:
            self.sub.subsample(self.splits, self.slice_spec(val_samples=999))
        self.assertIn("999.00 samples", str(ctx.exception))
        self.assertIn("30.00 samples", str(ctx.exception))
        tiny_train = {**self.splits, "train": self.splits["train"][:3]}
        result = self.sub.subsample(tiny_train, self.slice_spec(minimum=10))
        self.assertEqual(len(result.splits["train"]), 3)  # Train is untouched, so the minimum for a REDUCED Train does not apply

    def test_a_slice_needs_train_only_and_a_val_split_and_the_config_refuses_follow(self) -> None:
        with self.assertRaises(SubsampleConfigError):
            self.sub.subsample(self.splits, self.slice_spec("fraction", 0.5, scope="all_splits"))
        with self.assertRaises(SubsampleConfigError):
            self.sub.subsample(self.splits, replace(self.slice_spec(), val_split=None))
        with self.assertRaises(ValueError) as ctx:
            SubsampleConfig.model_validate({"val": "follow", "val_samples": 10})
        self.assertIn("cannot be combined", str(ctx.exception))
        with self.assertRaises(ValueError):
            SubsampleConfig.model_validate({"val_samples": 0})
        self.assertIsNone(SubsampleConfig().val_samples)  # off by default

    def test_the_config_store_and_fingerprint_know_the_slice(self) -> None:
        config = SubsampleConfig.model_validate({"enabled": True, "val_samples": 10, "size": {"minutes": 30}, "min_train_samples": 1})
        spec_ = build_spec(config, 42, "train", "val")
        self.assertEqual(spec_.val_samples, 10)
        with tempfile.TemporaryDirectory() as tmp:
            on = SubsampleConfig.model_validate({"enabled": True, "val_samples": 10, "size": {"minutes": 30}, "min_train_samples": 1, "dir": tmp})
            result = resolve_subsample(on, 42, self.splits, "train", "val")
            self.assertTrue(Path(result.report.ids_location or "").name.endswith("_v10_s42_r0.json"))
            again = resolve_subsample(on, 42, self.splits, "train", "val")
            self.assertEqual(again.fingerprint, result.fingerprint)
        without = self.sub.subsample(self.splits, replace(spec_, val_samples=None))
        self.assertNotEqual(without.fingerprint, self.sub.subsample(self.splits, spec_).fingerprint)

    def test_a_val_slice_with_all_train_through_resolve_subsample(self) -> None:
        config = SubsampleConfig.model_validate({"enabled": True, "val_samples": 7})
        result = resolve_subsample(config, 42, self.splits, "train", "val")
        self.assertEqual((len(result.splits["train"]), len(result.splits["val"]), len(result.splits["test"])), (100, 7, 30))


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name) / "subsets"
        self.splits = make_splits()
        self.store = SubsampleStore()

    def test_9_store_round_trip_and_a_stale_fingerprint_fails(self) -> None:
        s = spec("minutes", 30.0)
        result = Subsampler().subsample(self.splits, s)
        chosen = {n: [x.sample.id for x in v] for n, v in result.splits.items()}
        path = self.store.save(self.dir, s, self.splits, chosen)
        self.assertEqual(path.name, "minutes_30_by_sample_train_only_fixed_s42_r0.json")
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual((document["spec"]["seed"], document["fingerprint"]), (42, result.fingerprint))
        loaded = self.store.load(self.dir, s, self.splits)
        self.assertEqual({k: [n.sample.id for n in v] for k, v in loaded.items()}, chosen)
        with self.assertRaises(SubsampleError):
            self.store.save(self.dir, s, self.splits, chosen)  # never overwritten
        changed = {**self.splits, "train": [replace(n, sample=replace(n.sample, duration_sec=n.sample.duration_sec + 1)) for n in self.splits["train"]]}
        with self.assertRaises(SubsampleError) as ctx:
            self.store.load(self.dir, s, changed)
        self.assertIn("STALE", str(ctx.exception))
        document["splits"]["train"] = document["splits"]["train"][:-1]  # a file edited by hand no longer matches its fingerprint
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(SubsampleError):
            self.store.load(self.dir, s, self.splits)
        path.write_text("{ not json", encoding="utf-8")
        with self.assertRaises(SubsampleError) as ctx2:
            self.store.load(self.dir, s, self.splits)
        self.assertIn("cannot be read", str(ctx2.exception))

    def test_resolve_subsample_makes_stores_and_reuses_the_subset_and_fails_when_stale(self) -> None:
        config = SubsampleConfig.model_validate({"enabled": True, "size": {"minutes": 30}, "dir": str(self.dir), "min_train_samples": 1})
        first = resolve_subsample(config, 42, self.splits, "train", "val")
        self.assertIsNotNone(first.report.ids_location)
        self.assertTrue(Path(first.report.ids_location or "").is_file())
        second = resolve_subsample(config, 42, self.splits, "train", "val")
        self.assertEqual((second.fingerprint, ids(second.splits)), (first.fingerprint, ids(first.splits)))
        self.assertIn("ids stored in", second.report.summary())
        changed = {**self.splits, "val": self.splits["val"][:-1]}
        with self.assertRaises(SubsampleError) as ctx:
            resolve_subsample(config, 42, changed, "train", "val")
        self.assertIn("STALE", str(ctx.exception))
        other_seed = resolve_subsample(config, 7, self.splits, "train", "val")  # another spec: another file
        self.assertNotEqual(ids(other_seed.splits), ids(first.splits))
        self.assertEqual(len(list(self.dir.glob("*.json"))), 2)


class ModeTests(unittest.TestCase):
    def test_10_disabled_mode_returns_the_input_unchanged_and_says_so(self) -> None:
        splits = make_splits()
        result = resolve_subsample(SubsampleConfig(), 42, splits, "train", "val")
        self.assertFalse(result.report.enabled)
        for name, samples in splits.items():
            self.assertTrue(all(a is b for a, b in zip(result.splits[name], samples, strict=True)))
            self.assertEqual(len(result.splits[name]), len(samples))
        self.assertIn("disabled", result.report.summary())
        self.assertIsNone(result.report.ids_location)

    def test_presets_have_the_documented_values_and_unknown_ones_fail_listing_the_names(self) -> None:
        self.assertEqual(sorted(BUILT_IN_PRESETS), ["full", "half", "smoke", "tiny", "work_like"])
        expect = {"smoke": ("samples", 20.0, "by_sample", "train_only"), "tiny": ("minutes", 30.0, "by_sample", "train_only"), "half": ("fraction", 0.5, "by_sample", "train_only"), "work_like": ("fraction", 0.1, "by_group", "all_splits"), "full": ("all", None, "by_sample", "train_only")}
        for name, (kind, value, strategy, scope) in expect.items():
            built = build_spec(SubsampleConfig(preset=name), 42, "train", "val")
            self.assertEqual((built.size.kind, built.size.value, built.strategy, built.scope), (kind, value, strategy, scope), name)
        with self.assertRaises(SubsampleConfigError) as ctx:
            build_spec(SubsampleConfig(preset="quarter"), 42, "train", "val")
        self.assertIn("half, smoke, tiny, work_like", str(ctx.exception).replace("full, ", ""))
        own = SubsampleConfig.model_validate({"preset": "quarter", "presets": {"quarter": {"size": {"fraction": 0.25}, "strategy": "by_group"}}})
        built = build_spec(own, 42, "train", "val")
        self.assertEqual((built.size, built.strategy, built.scope), (SubsampleSize("fraction", 0.25), "by_group", "train_only"))
        self.assertEqual(sorted(own.presets), ["quarter"])  # only the presets you list; the built-in ones stay in code

    def test_the_work_like_preset_reduces_all_three_splits_by_group_to_a_tenth(self) -> None:
        splits = make_splits(train_groups=40, val_groups=20, test_groups=20)
        result = resolve_subsample(SubsampleConfig.model_validate({"enabled": True, "preset": "work_like", "min_train_samples": 1}), 42, splits, "train", "val")
        for name in ("train", "val", "test"):
            self.assertLess(len(result.splits[name]), len(splits[name]) * 0.25)
        self.assertEqual({s.name for s in result.report.sizes if s.reduced}, {"train", "val", "test"})

    def test_config_defaults_and_validation(self) -> None:
        c = SubsampleConfig()
        self.assertEqual((c.enabled, c.size, c.strategy, c.scope, c.val, c.repeats, c.repeat_index, c.min_train_samples, c.preset, c.dir), (False, "all", "by_sample", "train_only", "fixed", 1, 0, 10, None, None))
        self.assertEqual(SubsampleConfig.model_validate({"size": {"minutes": 30}}).size, SizeConfig(minutes=30))
        for bad in ({"size": {"fraction": 1.5}}, {"size": {"minutes": -1}}, {"size": {"minutes": 1, "samples": 2}}, {"size": {}}, {"size": "some"}, {"scope": "everything"}, {"repeats": 2, "repeat_index": 2}, {"min_train_samples": 0}, {"nope": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                SubsampleConfig.model_validate(bad)

    def test_the_yaml_example_loads_and_shows_exactly_the_defaults(self) -> None:
        loaded = load_config(ROOT / "configs" / "subsample_default.yaml", env_file=None, target_environ={}, registry=SectionRegistry({"subsample": SubsampleConfig}))
        self.assertEqual(loaded.section("subsample", SubsampleConfig), SubsampleConfig())

    def test_no_network_and_no_model_imports(self) -> None:
        code = "import sys; sys.path.insert(0, 'src'); import stt.subsampler; print(any(m.split('.')[0] in ('torch', 'transformers', 'scipy') for m in sys.modules))"
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False")
        for path in sorted((ROOT / "src" / "stt" / "subsampler").glob("*.py")):  # the component itself imports no heavy library
            self.assertNotRegex(path.read_text(encoding="utf-8"), r"(?m)^\s*(import|from)\s+(torch|transformers|numpy|scipy|soundfile|socket|urllib|requests)")


if __name__ == "__main__":
    unittest.main()
