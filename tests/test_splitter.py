"""Unit tests of the Splitter component (stt.splitter): the acceptance tests of the C3 brief, offline.

How to run (project root, virtual environment active):
    python -m unittest tests.test_splitter -v
The tests build small synthetic AudioSample lists in memory; no audio file exists and nothing is downloaded.
"""

from __future__ import annotations

import ast
import csv
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stt.config import ConfigError, SectionRegistry, load_config  # noqa: E402
from stt.datasource import AudioSample  # noqa: E402
from stt.splitter import (  # noqa: E402
    GroupedRatioSplitter,
    HashGroupSplitter,
    PredefinedSplitter,
    SplitConfig,
    SplitError,
    SplitStore,
    TimeBlockSplitter,
    default_registry,
    resolve_seed,
    resolve_split,
)
from stt.splitter.checks import verify_result  # noqa: E402
from stt.splitter.keys import group_keys  # noqa: E402


def sample(i: str, group: str, duration: float = 10.0, split: str | None = None, **extra: Any) -> AudioSample:
    return AudioSample(i, Path(f"{i}.wav"), "text", group, duration, split, extra)


def groups_data(n_groups: int = 20, per_group: int = 5, **kwargs: Any) -> list[AudioSample]:
    return [sample(f"g{g:03d}_s{k}", f"g{g:03d}", **kwargs) for g in range(n_groups) for k in range(per_group)]


def cfg(**overrides: Any) -> SplitConfig:
    base: dict[str, Any] = {"min_samples_per_split": 1, "min_groups_per_split": 1}
    base.update(overrides)
    return SplitConfig.model_validate(base)


def assignment(result: Any) -> dict[str, str]:
    return {s.id: name for name, group in result.splits.items() for s in group}


class GroupedRatioTests(unittest.TestCase):
    def test_1_twenty_groups_no_leak_union_equals_input_ratios_in_tolerance(self) -> None:
        data = groups_data()
        result = GroupedRatioSplitter(cfg(), seed=1).split(data)
        self.assertEqual(sorted(s.id for g in result.splits.values() for s in g), sorted(s.id for s in data))
        group_to_splits: dict[str, set[str]] = {}
        for name, group in result.splits.items():
            for s in group:
                group_to_splits.setdefault(s.group_id, set()).add(name)
        self.assertTrue(all(len(v) == 1 for v in group_to_splits.values()))
        for stat in result.report.stats:
            self.assertLessEqual(abs(stat.actual_ratio - stat.requested_ratio), 0.05)
        self.assertEqual([s.name for s in result.report.stats], ["train", "val", "test"])
        self.assertEqual(result.report.total_groups, 20)

    def test_2_determinism_shuffled_input_same_result_other_seed_different(self) -> None:
        data = groups_data()
        first = GroupedRatioSplitter(cfg(), seed=7).split(data)
        shuffled = data[:]
        random.Random(0).shuffle(shuffled)
        again = GroupedRatioSplitter(cfg(), seed=7).split(shuffled)
        self.assertEqual(assignment(first), assignment(again))
        self.assertEqual(first.fingerprint, again.fingerprint)
        other = GroupedRatioSplitter(cfg(), seed=8).split(data)
        self.assertNotEqual(assignment(first), assignment(other))
        self.assertNotEqual(first.fingerprint, other.fingerprint)

    def test_ratio_unit_samples_and_duration_differ(self) -> None:
        data = [sample(f"a{i}", f"a{i}", duration=1.0) for i in range(8)] + [sample("b0", "b0", duration=20.0), sample("c0", "c0", duration=20.0)]
        by_dur = GroupedRatioSplitter(cfg(ratios=[0.5, 0.25, 0.25], ratio_tolerance=0.5), seed=1).split(data)
        by_n = GroupedRatioSplitter(cfg(ratios=[0.5, 0.25, 0.25], ratio_unit="samples", ratio_tolerance=0.5), seed=1).split(data)
        self.assertNotEqual(assignment(by_dur), assignment(by_n))

    def test_ratio_miss_beyond_tolerance_fails(self) -> None:
        data = [sample(f"s{i}", f"g{i}") for i in range(3)]
        with self.assertRaises(SplitError) as ctx:
            GroupedRatioSplitter(cfg(ratio_tolerance=0.01, on_oversized_group="warn"), seed=1).split(data)
        self.assertIn("ratio_tolerance", str(ctx.exception))

    def test_report_has_the_required_content(self) -> None:
        data = groups_data(source_split=None) + [sample("x_1", "gx", split="eval", meta_creator="z"), sample("x_2", "gx", split="eval", meta_creator="z")]
        report = GroupedRatioSplitter(cfg(speaker_column="meta_creator", on_oversized_group="warn"), seed=3).split(data).report
        self.assertEqual(report.total_samples, 102)
        self.assertEqual(len(report.largest_groups), 5)
        self.assertGreater(report.largest_group_share, 0)
        self.assertIn("ok", report.group_overlap_check)
        self.assertIn("(none)", report.source_split_table)
        self.assertIn("speakers (meta_creator)", report.speaker_summary)
        self.assertEqual(len(report.data_fingerprint), 64)
        self.assertIn("samples 102", report.summary())


class HashGroupTests(unittest.TestCase):
    def test_3_adding_a_group_does_not_move_existing_groups(self) -> None:
        data = groups_data(60, 2)
        before = assignment(HashGroupSplitter(cfg(mode="hash_group", ratio_tolerance=1.0), seed=5).split(data))
        grown = data + [sample(f"new{k}_s{j}", f"new{k}") for k in range(20) for j in range(2)]
        after = assignment(HashGroupSplitter(cfg(mode="hash_group", ratio_tolerance=1.0), seed=5).split(grown))
        self.assertTrue(all(after[i] == name for i, name in before.items()))
        self.assertGreater(len(after), len(before))

    def test_hash_is_independent_of_input_order_and_depends_on_seed(self) -> None:
        data = groups_data(60, 2)
        shuffled = data[:]
        random.Random(1).shuffle(shuffled)
        c = cfg(mode="hash_group", ratio_tolerance=1.0)
        self.assertEqual(assignment(HashGroupSplitter(c, 5).split(data)), assignment(HashGroupSplitter(c, 5).split(shuffled)))
        self.assertNotEqual(assignment(HashGroupSplitter(c, 5).split(data)), assignment(HashGroupSplitter(c, 6).split(data)))

    def test_few_groups_fail_the_minimum_gates_instead_of_continuing(self) -> None:
        with self.assertRaises(SplitError) as ctx:
            HashGroupSplitter(
                cfg(mode="hash_group", min_groups_per_split=3, min_samples_per_split=1, on_oversized_group="warn"), seed=1
            ).split(groups_data(6, 2))
        self.assertTrue("Minimum gate" in str(ctx.exception) or "empty" in str(ctx.exception))

    def test_uneven_hash_split_is_reported_as_a_note(self) -> None:
        result = HashGroupSplitter(cfg(mode="hash_group", ratio_tolerance=0.0), seed=5).split(groups_data(60, 2))
        self.assertTrue(any("WARNING" in n for n in result.report.notes))


class PredefinedTests(unittest.TestCase):
    def data(self) -> list[AudioSample]:
        return [sample(f"tr{g}_{k}", f"tr{g}", split="train") for g in range(10) for k in range(2)] + [
            sample(f"te{g}_{k}", f"te{g}", split="test") for g in range(3) for k in range(2)
        ]

    def test_4_uses_source_split_and_carves_val_from_train_by_group(self) -> None:
        result = PredefinedSplitter(cfg(mode="predefined", ratios=[0.7, 0.1, 0.2]), seed=1).split(self.data())
        self.assertTrue(all(s.source_split == "test" for s in result.splits["test"]))
        self.assertEqual(len(result.splits["test"]), 6)
        self.assertTrue(result.splits["val"])
        self.assertTrue(all(s.source_split == "train" for s in result.splits["val"] + result.splits["train"]))
        val_groups = {s.group_id for s in result.splits["val"]}
        self.assertFalse(val_groups & {s.group_id for s in result.splits["train"]})
        self.assertTrue(any("carved" in n for n in result.report.notes))

    def test_source_map_renames_source_splits(self) -> None:
        data = [s if s.source_split != "test" else sample(s.id, s.group_id, split="eval") for s in self.data()]
        with self.assertRaises(SplitError) as ctx:
            PredefinedSplitter(cfg(mode="predefined"), seed=1).split(data)
        self.assertIn("source_map", str(ctx.exception))
        result = PredefinedSplitter(cfg(mode="predefined", predefined={"source_map": {"eval": "test"}}), seed=1).split(data)
        self.assertEqual(len(result.splits["test"]), 6)

    def test_overlap_fails_or_drops_from_train(self) -> None:
        data = self.data() + [sample("te0_extra", "tr1", split="test")]
        with self.assertRaises(SplitError) as ctx:
            PredefinedSplitter(cfg(mode="predefined"), seed=1).split(data)
        self.assertIn("more than one source split", str(ctx.exception))
        result = PredefinedSplitter(cfg(mode="predefined", on_overlap="drop_from_train"), seed=1).split(data)
        self.assertEqual({s.id for s in result.dropped_overlap}, {"tr1_0", "tr1_1"})
        self.assertIn("te0_extra", {s.id for s in result.splits["test"]})
        self.assertIn("removed from 'train'", result.report.group_overlap_check)
        self.assertEqual(result.report.dropped_overlap, 2)

    def test_missing_source_split_fails(self) -> None:
        with self.assertRaises(SplitError) as ctx:
            PredefinedSplitter(cfg(mode="predefined"), seed=1).split([sample("a", "g1", split=None)])
        self.assertIn("source_split", str(ctx.exception))


class TimeBlockTests(unittest.TestCase):
    def recording(self, n: int = 40, group: str = "rec1", step: float = 10.0) -> list[AudioSample]:
        return [sample(f"{group}_{i:03d}", group, duration=step, meta_seek=i * step) for i in range(n)]

    def test_5_contiguous_blocks_guard_gap_dropped_and_reported(self) -> None:
        data = self.recording()
        result = TimeBlockSplitter(cfg(mode="time_block", time_block={"guard_gap_sec": 25}), seed=1).split(data)
        self.assertTrue(result.dropped_guard)
        self.assertEqual(result.report.dropped_guard, len(result.dropped_guard))
        spans = {}
        for name, group in result.splits.items():
            starts = [s.extra["meta_seek"] for s in group]
            spans[name] = (min(starts), max(s.extra["meta_seek"] + s.duration_sec for s in group))
            self.assertEqual(sorted(starts), list(starts))
        ordered = [spans["train"], spans["val"], spans["test"]]
        for (_, end_a), (start_b, _) in zip(ordered, ordered[1:]):
            self.assertGreaterEqual(start_b - end_a, 25 - 1e-9)
        everyone = [s.id for g in result.splits.values() for s in g] + [s.id for s in result.dropped_guard]
        self.assertEqual(sorted(everyone), sorted(s.id for s in data))
        self.assertIn("by design", result.report.group_overlap_check)

    def test_zero_gap_drops_nothing(self) -> None:
        result = TimeBlockSplitter(cfg(mode="time_block", time_block={"guard_gap_sec": 0}), seed=1).split(self.recording())
        self.assertEqual(result.dropped_guard, [])

    def test_missing_order_column_fails_with_examples(self) -> None:
        data = self.recording(5) + [sample("nope", "rec1")]
        with self.assertRaises(SplitError) as ctx:
            TimeBlockSplitter(cfg(mode="time_block"), seed=1).split(data)
        self.assertIn("meta_seek", str(ctx.exception))
        self.assertIn("nope", str(ctx.exception))

    def test_every_recording_is_cut_separately(self) -> None:
        data = self.recording(100, "a") + self.recording(100, "b")
        result = TimeBlockSplitter(cfg(mode="time_block", time_block={"guard_gap_sec": 10}), seed=1).split(data)
        for group in ("a", "b"):
            for name in ("train", "val", "test"):
                self.assertTrue(any(s.group_id == group for s in result.splits[name]), (group, name))


class GatesAndKeysTests(unittest.TestCase):
    def test_6_empty_input_fails(self) -> None:
        with self.assertRaises(SplitError):
            GroupedRatioSplitter(cfg(), 1).split([])

    def test_6_duplicate_ids_fail(self) -> None:
        with self.assertRaises(SplitError) as ctx:
            GroupedRatioSplitter(cfg(), 1).split([sample("a", "g1"), sample("a", "g2")])
        self.assertIn("Duplicate", str(ctx.exception))

    def test_6_minimum_gates_name_counts_and_options(self) -> None:
        with self.assertRaises(SplitError) as ctx:
            GroupedRatioSplitter(cfg(min_samples_per_split=30, min_groups_per_split=2), 1).split(groups_data())
        message = str(ctx.exception)
        self.assertIn("Minimum gate", message)
        self.assertIn("finer group key", message)
        with self.assertRaises(SplitError) as ctx2:
            GroupedRatioSplitter(cfg(min_samples_per_split=1, min_groups_per_split=5), 1).split(groups_data())
        self.assertIn("groups (minimum 5)", str(ctx2.exception))

    def test_6_oversized_group_fails_or_warns(self) -> None:
        data = groups_data(10, 2) + [sample(f"big_{i}", "big") for i in range(20)]
        with self.assertRaises(SplitError) as ctx:
            GroupedRatioSplitter(cfg(), 1).split(data)
        self.assertIn("big", str(ctx.exception))
        result = GroupedRatioSplitter(cfg(on_oversized_group="warn", ratio_tolerance=0.3), 1).split(data)
        self.assertTrue(any("WARNING" in n for n in result.report.notes))

    def test_a_single_group_cannot_fill_three_splits(self) -> None:
        with self.assertRaises(SplitError):
            GroupedRatioSplitter(cfg(on_oversized_group="warn", ratio_tolerance=1.0), 1).split([sample(f"s{i}", "only") for i in range(10)])

    def test_7_regex_group_key_extracts_the_group(self) -> None:
        data = [sample(f"s{i}", f"lesson{i}", meta_source=f"Tractate_{i % 4}-page{i}.srt") for i in range(8)]
        keys = group_keys(data, "meta_source", r"^(\w+?)_(\d)-")
        self.assertEqual(sorted(set(keys.values())), ["Tractate"])
        keys = group_keys(data, "meta_source", r"_(\d)-")
        self.assertEqual(sorted(set(keys.values())), ["0", "1", "2", "3"])
        result = GroupedRatioSplitter(
            cfg(group_key={"column": "meta_source", "regex": r"_(\d)-"}, on_oversized_group="warn", ratio_tolerance=1.0, ratios=[0.5, 0.25, 0.25]), 1
        ).split(data)
        self.assertEqual(result.report.total_groups, 4)

    def test_7_non_matching_or_missing_value_fails_with_examples(self) -> None:
        data = [sample("ok", "g", meta_source="T_1-x"), sample("bad", "g", meta_source="nothing"), sample("none", "g")]
        with self.assertRaises(SplitError) as ctx:
            group_keys(data, "meta_source", r"_(\d)-")
        message = str(ctx.exception)
        self.assertIn("bad", message)
        self.assertIn("none", message)
        self.assertNotIn("ok:", message)

    def test_group_key_fields(self) -> None:
        s = [sample("a", "g1", split="train", k=3)]
        self.assertEqual(group_keys(s, "id", None), {"a": "a"})
        self.assertEqual(group_keys(s, "source_split", None), {"a": "train"})
        self.assertEqual(group_keys(s, "k", None), {"a": "3"})
        self.assertEqual(group_keys(s, None, None), {"a": "g1"})


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name) / "v1"
        self.data = groups_data()
        self.result = GroupedRatioSplitter(cfg(), 4).split(self.data)

    def test_8_round_trip_gives_an_equal_result(self) -> None:
        store = SplitStore()
        store.save(self.result, self.dir)
        loaded = store.load(self.dir, list(reversed(self.data)))
        self.assertEqual(assignment(loaded), assignment(self.result))
        self.assertEqual(loaded.fingerprint, self.result.fingerprint)
        self.assertEqual(loaded.report.params, self.result.report.params)
        self.assertEqual([s.name for s in loaded.report.stats], ["train", "val", "test"])

    def test_8_fingerprint_mismatch_fails_with_new_version_advice(self) -> None:
        store = SplitStore()
        store.save(self.result, self.dir)
        with self.assertRaises(SplitError) as ctx:
            store.load(self.dir, self.data + [sample("extra", "gx")])
        self.assertIn("split.version", str(ctx.exception))
        changed = [sample(s.id, s.group_id, duration=s.duration_sec + 1) for s in self.data]
        with self.assertRaises(SplitError):
            store.load(self.dir, changed)

    def test_8_csv_export_matches_the_json(self) -> None:
        store = SplitStore()
        path = store.save(self.result, self.dir)
        csv_path = store.export_csv(self.result, self.dir / "split.csv")
        saved = json.loads(path.read_text(encoding="utf-8"))
        from_json = {i: name for name, ids in saved["splits"].items() for i in ids}
        with csv_path.open(encoding="utf-8", newline="") as handle:
            from_csv = {row["id"]: row["split"] for row in csv.DictReader(handle)}
        self.assertEqual(from_csv, from_json)
        self.assertEqual(saved["schema_version"], 1)
        self.assertEqual(saved["data_fingerprint"], self.result.data_fingerprint)

    def test_save_never_overwrites_and_missing_or_damaged_files_fail(self) -> None:
        store = SplitStore()
        store.save(self.result, self.dir)
        with self.assertRaises(SplitError) as ctx:
            store.save(self.result, self.dir)
        self.assertIn("split.version", str(ctx.exception))
        with self.assertRaises(SplitError):
            store.load(self.dir.parent / "nothing", self.data)
        (self.dir / "split.json").write_text("{not json", encoding="utf-8")
        with self.assertRaises(SplitError):
            store.load(self.dir, self.data)

    def test_hand_edited_split_file_is_detected(self) -> None:
        store = SplitStore()
        path = store.save(self.result, self.dir)
        doc = json.loads(path.read_text(encoding="utf-8"))
        moved = doc["splits"]["train"].pop()
        doc["splits"]["test"].append(moved)
        path.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaises(SplitError):
            store.load(self.dir, self.data)

    def test_time_block_split_round_trips_too(self) -> None:
        data = [sample(f"r_{i:03d}", "r", meta_seek=i * 10.0) for i in range(40)]
        result = TimeBlockSplitter(cfg(mode="time_block"), 1).split(data)
        store = SplitStore()
        store.save(result, self.dir)
        loaded = store.load(self.dir, data)
        self.assertEqual(assignment(loaded), assignment(result))
        self.assertEqual({s.id for s in loaded.dropped_guard}, {s.id for s in result.dropped_guard})

    def test_resolve_split_saves_then_reuses_and_never_overwrites(self) -> None:
        config = cfg(dir=str(self.dir.parent / "splits"), export_csv=True)
        first = resolve_split(config, 9, self.data)
        self.assertTrue((self.dir.parent / "splits" / "v1" / "split.json").is_file())
        self.assertTrue((self.dir.parent / "splits" / "v1" / "split.csv").is_file())
        with mock.patch.object(GroupedRatioSplitter, "assign", side_effect=AssertionError("must reuse")):
            again = resolve_split(config, 9, self.data)
        self.assertEqual(assignment(first), assignment(again))
        with self.assertRaises(SplitError) as ctx:
            resolve_split(cfg(dir=str(self.dir.parent / "splits"), reuse=False), 9, self.data)
        self.assertIn("never overwritten", str(ctx.exception))
        with self.assertRaises(SplitError):
            resolve_split(config, 9, self.data[:-1])
        newer = resolve_split(cfg(dir=str(self.dir.parent / "splits"), version="v2"), 9, self.data[:-5])
        self.assertTrue((self.dir.parent / "splits" / "v2" / "split.json").is_file())
        self.assertNotEqual(newer.fingerprint, first.fingerprint)

    def test_resolve_split_without_dir_writes_nothing(self) -> None:
        with mock.patch("pathlib.Path.write_text", side_effect=AssertionError("no files")):
            resolve_split(cfg(), 1, self.data)


class InvariantTests(unittest.TestCase):
    def test_9_every_registered_splitter_satisfies_the_hard_checks(self) -> None:
        registry = default_registry()
        self.assertEqual(registry.names(), ["grouped_ratio", "hash_group", "predefined", "time_block"])
        data = [
            sample(f"g{g:02d}_{k}", f"g{g:02d}", split="train" if g < 16 else "test", meta_seek=float(k * 10)) for g in range(20) for k in range(6)
        ]
        for mode in registry.names():
            with self.subTest(mode=mode):
                config = cfg(mode=mode, ratios=[0.6, 0.2, 0.2], ratio_tolerance=0.3, on_oversized_group="warn", time_block={"guard_gap_sec": 0})
                splitter = registry.create(config, 11)
                result = splitter.split(data)
                keys = {s.id: s.group_id for s in data}
                verify_result(result, data, keys, config, groups_may_span_splits=(mode == "time_block"))
                ids = [s.id for g in result.splits.values() for s in g] + [s.id for s in result.dropped_guard + result.dropped_overlap]
                self.assertEqual(sorted(ids), sorted(s.id for s in data))
                self.assertEqual(len(ids), len(set(ids)))
                self.assertTrue(all(result.splits[n] for n in config.names))
                self.assertEqual(splitter.split(list(reversed(data))).fingerprint, result.fingerprint)

    def test_verify_result_catches_a_leak_and_a_lost_sample(self) -> None:
        data = groups_data(20, 2)
        result = GroupedRatioSplitter(cfg(), 1).split(data)
        keys = {s.id: s.group_id for s in data}
        leaked_splits = {name: list(group) for name, group in result.splits.items()}
        leaked_splits["test"].append(leaked_splits["train"][0].__class__(**{**vars(leaked_splits["train"][0]), "id": "dup"}))
        keys["dup"] = leaked_splits["train"][0].group_id
        from dataclasses import replace

        broken = replace(result, splits=leaked_splits)
        with self.assertRaises(SplitError):
            verify_result(broken, [*data, leaked_splits["test"][-1]], keys, cfg(), False)
        lost = replace(result, splits={n: g[:-1] for n, g in result.splits.items()})
        with self.assertRaises(SplitError):
            verify_result(lost, data, keys, cfg(), False)

    def test_unknown_mode_lists_registered_modes(self) -> None:
        with self.assertRaises(SplitError) as ctx:
            default_registry().create(cfg(mode="magic"), 1)
        self.assertIn("grouped_ratio, hash_group, predefined, time_block", str(ctx.exception))

    def test_10_inputs_unmodified_and_no_file_access_by_a_splitter(self) -> None:
        data = groups_data()
        before = [(s.id, s.group_id, s.duration_sec, dict(s.extra)) for s in data]
        order = [s.id for s in data]
        with mock.patch("builtins.open", side_effect=AssertionError("no file access")), mock.patch(
            "pathlib.Path.open", side_effect=AssertionError("no file access")
        ):
            GroupedRatioSplitter(cfg(), 1).split(data)
        self.assertEqual(before, [(s.id, s.group_id, s.duration_sec, dict(s.extra)) for s in data])
        self.assertEqual(order, [s.id for s in data])

    def test_10_no_network_audio_or_tools_imports(self) -> None:
        for path in (ROOT / "src" / "stt" / "splitter").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    self.assertNotIn(
                        name.split(".")[0],
                        {"requests", "urllib3", "httpx", "huggingface_hub", "socket", "tools", "soundfile", "pyarrow", "scipy", "torch"},
                        f"{path} imports {name}",
                    )


class ConfigTests(unittest.TestCase):
    def test_defaults_and_seed_fallback(self) -> None:
        config = SplitConfig()
        self.assertEqual((config.names, config.ratios, config.mode), (("train", "val", "test"), (0.8, 0.1, 0.1), "grouped_ratio"))
        self.assertEqual(resolve_seed(config, 42), 42)
        self.assertEqual(resolve_seed(SplitConfig(seed=5), 42), 5)

    def test_invalid_values_fail_through_the_config_loader(self) -> None:
        registry = SectionRegistry({"split": SplitConfig})

        def load(*overrides: str) -> Any:
            return load_config(overrides=list(overrides), registry=registry, environ={}, env_file=None, target_environ={})

        cfg_ok = load("split.ratios=[0.5,0.5]", 'split.names=["a","b"]')
        self.assertEqual(cfg_ok.section("split", SplitConfig).ratios, (0.5, 0.5))
        for bad, expected in (
            (("split.ratios=[0.5,0.4,0.2]",), "sum to 1"),
            (("split.ratios=[0.5,0.5]",), "must match"),
            (('split.names=["a","a","b"]',), "unique"),
            (("split.ratios=[1.2,-0.1,-0.1]",), "greater than 0"),
            (("split.group_key.regex=(",), "regular expression"),
            (("split.mode_typo=x",), "Unknown key"),
            (("split.version=a/b",), "version"),
        ):
            with self.assertRaises(ConfigError, msg=str(bad)) as ctx:
                load(*bad)
            self.assertIn(expected, str(ctx.exception))

    def test_shipped_example_config_loads(self) -> None:
        registry = SectionRegistry({"split": SplitConfig})
        config = load_config(ROOT / "configs" / "split_default.yaml", registry=registry, environ={}, env_file=None, target_environ={})
        self.assertEqual(config.section("split", SplitConfig), SplitConfig())


if __name__ == "__main__":
    unittest.main()
