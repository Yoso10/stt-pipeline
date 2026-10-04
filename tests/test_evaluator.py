"""Unit tests of the Evaluator component (stt.evaluator): the acceptance tests of the C7 brief, offline.

How to run (project root, virtual environment active):
    python -m unittest tests.test_evaluator -v
Most tests use a fake Transcriber with scripted outputs (no model needed). The adapter tests use the stored tiny test model
(tests/assets/tiny_whisper, random weights) on random features. Nothing is downloaded.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stt.config import SectionRegistry, load_config  # noqa: E402
from stt.evaluator import (  # noqa: E402
    Comparator,
    DecodeSettings,
    EvalConfig,
    EvalItem,
    EvalRequest,
    EvalResult,
    Evaluator,
    EvaluatorConfigError,
    EvaluatorError,
    GroupErrors,
    PairedGroupBootstrap,
    RawHypothesis,
    WordErrorRate,
    CharErrorRate,
    build_review_list,
    default_metric_registry,
    default_significance_registry,
    edit_distance,
    make_decode_settings,
    resample_deltas,
    word_diff,
)
from stt.featurepipeline import FeatureExample  # noqa: E402
from stt.modelloader import LoadedModel, ModelConfig, create_loader  # noqa: E402
from stt.textnormalizer import TextConfig, build_profile  # noqa: E402

TINY = ROOT / "tests" / "assets" / "tiny_whisper"
_MODEL: LoadedModel | None = None
EVAL_PROFILE = build_profile(TextConfig(), "eval")


def model() -> LoadedModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = create_loader(ModelConfig(path=TINY)).load()
    return _MODEL


class FakeDataset:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> FeatureExample:
        return FeatureExample(self.ids[index], np.zeros((80, 3000), dtype=np.float32), [1])


class FakeTranscriber:
    """Returns scripted texts by sample id. `hits` is the set of ids that report a hit length cap."""

    def __init__(self, texts: Mapping[str, str], hits: frozenset[str] = frozenset()) -> None:
        self.texts = texts
        self.hits = hits
        self.calls = 0

    def transcribe(self, model: LoadedModel, dataset: Any, settings: DecodeSettings) -> list[RawHypothesis]:
        self.calls += 1
        return [RawHypothesis(dataset[i].sample_id, self.texts[dataset[i].sample_id], dataset[i].sample_id in self.hits) for i in range(len(dataset))]


def item(sample_id: str, reference: str, group: str = "g", **attributes: Any) -> EvalItem:
    return EvalItem(sample_id, group, reference, Path(f"audio/{sample_id}.mp3"), attributes)


def settings() -> DecodeSettings:
    return DecodeSettings(1, 100, 4)


def evaluate(items: list[EvalItem], texts: Mapping[str, str], config: EvalConfig | None = None, version: int = 1, hits: frozenset[str] = frozenset()) -> EvalResult:
    config = config or EvalConfig()
    request = EvalRequest(items, FakeDataset([i.sample_id for i in items]), model(), "val", version, EVAL_PROFILE, settings())
    return Evaluator(config, FakeTranscriber(texts, hits), seed=42).evaluate(request)


def grouped(n_groups: int, files_per_group: int, base_errors: int, cand_errors: int) -> tuple[EvalResult, EvalResult]:
    """Two results over the same files: the baseline gets `base_errors` wrong words in every file, the candidate `cand_errors`."""
    items = [item(f"g{g}f{f}", "אחת שתיים שלוש ארבע חמש", f"g{g}") for g in range(n_groups) for f in range(files_per_group)]
    words = "אחת שתיים שלוש ארבע חמש".split()

    def texts(errors: int) -> dict[str, str]:
        return {i.sample_id: " ".join(["x"] * errors + words[errors:]) for i in items}

    return evaluate(items, texts(base_errors)), evaluate(items, texts(cand_errors))


class MetricTests(unittest.TestCase):
    def test_1_wer_and_cer_match_hand_computed_values(self) -> None:
        wer, cer = WordErrorRate(), CharErrorRate()
        cases = [  # reference, hypothesis, word errors, words, char errors
            ("שלום עולם טוב", "שלום עולם טוב", 0, 3, 0),
            ("שלום עולם טוב", "שלום עולם רע", 1, 3, 3),  # one substitution; "טוב"->"רע": 3 characters, 3 edits
            ("שלום עולם טוב", "שלום טוב", 1, 3, 5),  # the word "עולם " is missing: 5 characters
            ("שלום עולם טוב", "שלום עולם יפה טוב", 1, 3, 4),  # the word "יפה " is added: 4 characters
            ("בבית הספר", "הבית הספר", 1, 2, 1),  # prefix error: one word wrong but one character
            ("שלום עולם", "", 2, 2, 9),  # empty output: every word deleted
        ]
        for ref, hyp, word_errors, words, char_errors in cases:
            with self.subTest(ref=ref, hyp=hyp):
                self.assertEqual((wer.score(ref, hyp).errors, wer.score(ref, hyp).units), (word_errors, words))
                self.assertEqual(cer.score(ref, hyp).errors, char_errors)
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance([], ["a", "b"]), 2)

    def test_2_same_words_with_different_punctuation_score_zero_through_the_same_profile(self) -> None:
        result = evaluate([item("a", EVAL_PROFILE("שלום עולם"))], {"a": "<|0.00|>שלום, עולם.<|1.00|>"})
        self.assertEqual((result.wer, result.cer, result.files[0].hypothesis), (0.0, 0.0, "שלום עולם"))
        self.assertEqual(result.files[0].hypothesis_raw, "<|0.00|>שלום, עולם.<|1.00|>")

    def test_3_corpus_wer_differs_from_the_mean_of_file_wers(self) -> None:
        items = [item("a", "אחת"), item("b", "א ב ג ד ה ו ז ח ט")]
        result = evaluate(items, {"a": "x", "b": "א ב ג ד ה ו ז ח ט"})
        self.assertAlmostEqual(result.wer, 1 / 10)
        self.assertAlmostEqual(result.wer_mean_per_file, 0.5)
        self.assertEqual((result.n_files, result.n_words), (2, 10))
        self.assertEqual(result.share_files_wer_below, {0.10: 0.5, 0.20: 0.5})

    def test_4_empty_output_is_a_full_error_and_flagged_and_an_empty_reference_is_excluded(self) -> None:
        items = [item("a", "אחת שתיים"), item("b", "שלוש"), item("c", "<|0.5|>.")]  # c is empty after normalization
        result = evaluate(items, {"a": "", "b": "שלוש", "c": "משהו"}, version=3)
        a = result.files[0]
        self.assertEqual((a.wer, a.word_errors, a.empty_output), (1.0, 2, True))
        self.assertEqual(dict(result.flags), {"empty_output": 1, "hit_length_cap": 0, "excluded_empty_reference": 0})
        items[2] = item("c", "")
        result = evaluate(items, {"a": "", "b": "שלוש", "c": "משהו"})
        self.assertEqual((result.n_files, result.excluded_ids, result.flags["excluded_empty_reference"]), (2, ("c",), 1))
        self.assertAlmostEqual(result.wer, 2 / 3)
        self.assertEqual(result.eval_version, 1)

    def test_5_the_length_cap_flags_a_looping_output_and_the_per_second_cap_is_derived_from_the_duration(self) -> None:
        result = evaluate([item("a", "אחת"), item("b", "שתיים")], {"a": "אחת", "b": "שתיים " * 50}, hits=frozenset({"b"}))
        self.assertEqual([f.hit_length_cap for f in result.files], [False, True])
        self.assertEqual(result.flags["hit_length_cap"], 1)
        config = EvalConfig.model_validate({"generation": {"max_tokens_per_audio_sec": 5.0}})
        caps = make_decode_settings(config, model(), [item("a", "x", duration=2.0), item("b", "y", duration=1000.0)])
        self.assertEqual(dict(caps.token_caps or {}), {"a": 10, "b": model().tokenizer.max_label_tokens})
        with self.assertRaises(EvaluatorError) as ctx:
            make_decode_settings(config, model(), [item("a", "x")])
        self.assertIn("duration", str(ctx.exception))
        self.assertIsNone(make_decode_settings(EvalConfig(), model(), [item("a", "x")]).token_caps)
        self.assertEqual(make_decode_settings(EvalConfig(), model(), []).max_new_tokens, model().tokenizer.max_label_tokens)

    def test_attributes_and_ids_pass_through_and_the_fake_needs_no_model_call(self) -> None:
        transcriber = FakeTranscriber({"a": "אחת"})
        request = EvalRequest([item("a", "אחת", "lesson1", duration=3.5, quality=0.2)], FakeDataset(["a"]), model(), "val", 1, EVAL_PROFILE, settings())
        result = Evaluator(EvalConfig(), transcriber, 1).evaluate(request)
        self.assertEqual((transcriber.calls, result.files[0].group_id, dict(result.files[0].attributes)), (1, "lesson1", {"duration": 3.5, "quality": 0.2}))
        self.assertEqual(result.model_info.name, "tiny_whisper")
        self.assertEqual(result.decode_settings.token_caps, None)
        self.assertIn("WER 0.0000", result.summary())


class ComparatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.comparator = Comparator(EvalConfig(), 42)

    def test_6_better_identical_and_worse_candidates(self) -> None:
        baseline, better = grouped(10, 2, 3, 0)
        result = self.comparator.compare(baseline, better)
        self.assertEqual((result.verdict, result.n_groups, result.min_effect_met), ("improved", 10, True))
        self.assertAlmostEqual(result.delta_wer or 0, 0.6)
        self.assertGreater(result.ci_low or 0, 0)
        same = self.comparator.compare(baseline, baseline)
        self.assertEqual((same.verdict, same.delta_wer), ("no_clear_difference", 0.0))
        worse = self.comparator.compare(better, baseline)
        self.assertEqual(worse.verdict, "worse")
        self.assertLess(worse.delta_wer or 0, 0)
        self.assertIn("worse", worse.explanation)

    def test_7_bootstrap_is_deterministic_and_resamples_whole_groups(self) -> None:
        big, small = GroupErrors(3, 0, 3), GroupErrors(0, 0, 1)  # a group of three improved files and a group of one unchanged file
        deltas = resample_deltas([big, small], 500, 7)
        self.assertEqual(set(np.round(deltas, 6)), {1.0, 0.75, 0.0})  # only whole-group combinations exist: AA, AB/BA, BB
        np.testing.assert_array_equal(deltas, resample_deltas([big, small], 500, 7))
        self.assertFalse(np.array_equal(deltas, resample_deltas([big, small], 500, 8)))
        test = PairedGroupBootstrap(0.9, True, 200)
        self.assertEqual(test.interval([big, small] * 4, 42), test.interval([big, small] * 4, 42))

    def test_8_a_different_eval_version_is_not_comparable_and_different_file_sets_fail(self) -> None:
        a, _ = grouped(6, 1, 1, 0)
        items = [item(f"g{g}f0", "אחת שתיים שלוש ארבע חמש", f"g{g}") for g in range(6)]
        other_version = evaluate(items, {i.sample_id: "אחת" for i in items}, version=2)
        result = self.comparator.compare(a, other_version)
        self.assertEqual((result.verdict, result.delta_wer, result.ci_low, result.ci_high), ("not_comparable", None, None, None))
        fewer = evaluate(items[:5], {i.sample_id: "אחת" for i in items[:5]})
        with self.assertRaises(EvaluatorError) as ctx:
            self.comparator.compare(a, fewer)
        self.assertIn("not scored on the same files", str(ctx.exception))

    def test_9_fewer_groups_than_the_minimum_gives_insufficient_groups_with_the_plain_difference(self) -> None:
        baseline, better = grouped(3, 2, 2, 0)
        result = self.comparator.compare(baseline, better)
        self.assertEqual((result.verdict, result.n_groups, result.ci_low, result.ci_high), ("insufficient_groups", 3, None, None))
        self.assertAlmostEqual(result.delta_wer or 0, 0.4)
        self.assertIn("minimum is 5", result.explanation)

    def test_two_sided_and_minimum_effect(self) -> None:
        baseline, better = grouped(10, 2, 1, 0)  # a 20 percent improvement
        two_sided = Comparator(EvalConfig.model_validate({"significance": {"one_sided": False}}), 42).compare(baseline, better)
        self.assertEqual(two_sided.verdict, "improved")
        demanding = Comparator(EvalConfig.model_validate({"significance": {"min_effect_abs": 0.5}}), 42).compare(baseline, better)
        self.assertEqual((demanding.verdict, demanding.min_effect_met), ("no_clear_difference", False))


class ReviewTests(unittest.TestCase):
    def results(self) -> EvalResult:
        items = [item(f"f{i:02d}", "אחת שתיים שלוש ארבע", f"g{i % 4}") for i in range(30)]
        texts = {i.sample_id: " ".join(["x"] * (n % 5) + "אחת שתיים שלוש ארבע".split()[n % 5 :]) for n, i in enumerate(items)}
        return evaluate(items, texts, EvalConfig.model_validate({"review": {"n_random": 6, "n_worst": 3}}))

    def test_10_the_review_list_has_the_configured_numbers_audio_paths_and_is_deterministic(self) -> None:
        result = self.results()
        entries = result.review.entries
        self.assertEqual([e.reason for e in entries], ["worst"] * 3 + ["random"] * 6)
        self.assertEqual(len({e.sample_id for e in entries}), 9)
        self.assertTrue(all(e.audio_path == Path(f"audio/{e.sample_id}.mp3") for e in entries))
        self.assertEqual([e.wer for e in entries[:3]], sorted((e.wer for e in entries[:3]), reverse=True))
        self.assertEqual(max(f.wer for f in result.files), entries[0].wer)
        again = self.results().review.entries
        self.assertEqual([e.sample_id for e in entries], [e.sample_id for e in again])
        paths = {f.sample_id: Path("a") for f in result.files}
        other_seed = build_review_list(result.files, paths, EvalConfig.model_validate({"review": {"n_random": 6, "n_worst": 3}}).review, 99)
        self.assertNotEqual([e.sample_id for e in entries[3:]], [e.sample_id for e in other_seed.entries[3:]])

    def test_the_review_list_is_capped_by_the_number_of_files(self) -> None:
        result = evaluate([item("a", "אחת"), item("b", "שתיים")], {"a": "x", "b": "שתיים"}, EvalConfig.model_validate({"review": {"n_random": 20, "n_worst": 10}}))
        self.assertEqual(len(result.review.entries), 2)

    def test_word_diff_format(self) -> None:
        self.assertEqual(word_diff("שלום עולם טוב", "שלום עולם רע"), "שלום עולם [טוב->רע]")
        self.assertEqual(word_diff("אחת שתיים שלוש", "אחת שלוש"), "אחת [-שתיים-] שלוש")
        self.assertEqual(word_diff("אחת שלוש", "אחת שתיים שלוש"), "אחת [+שתיים+] שלוש")
        self.assertEqual(word_diff("אחת", "אחת"), "אחת")


class FailureTests(unittest.TestCase):
    def test_unknown_metric_and_method_fail_listing_the_registered_names(self) -> None:
        with self.assertRaises(EvaluatorConfigError) as ctx:
            Evaluator(EvalConfig(metrics=["wer", "bleu"]), FakeTranscriber({}), 1)
        self.assertIn("cer, wer", str(ctx.exception))
        with self.assertRaises(EvaluatorConfigError) as ctx2:
            Comparator(EvalConfig.model_validate({"significance": {"method": "wilcoxon"}}), 1)
        self.assertIn("paired_group_bootstrap", str(ctx2.exception))
        with self.assertRaises(EvaluatorConfigError):
            Evaluator(EvalConfig(metrics=["cer"]), FakeTranscriber({}), 1)

    def test_all_hypotheses_empty_fails_naming_the_likely_cause(self) -> None:
        with self.assertRaises(EvaluatorError) as ctx:
            evaluate([item("a", "אחת"), item("b", "שתיים")], {"a": "", "b": " . "})
        self.assertIn("model.language", str(ctx.exception))

    def test_mismatching_hypotheses_and_empty_item_list_fail(self) -> None:
        request = EvalRequest([item("a", "אחת"), item("b", "שתיים")], FakeDataset(["a", "z"]), model(), "val", 1, EVAL_PROFILE, settings())
        with self.assertRaises(EvaluatorError) as ctx:
            Evaluator(EvalConfig(), FakeTranscriber({"a": "אחת", "z": "x"}), 1).evaluate(request)
        self.assertIn("'b'", str(ctx.exception))
        with self.assertRaises(EvaluatorError):
            evaluate([], {})

    def test_sanity_limit(self) -> None:
        result = evaluate([item("a", "אחת שתיים")], {"a": "x y"})
        Evaluator(EvalConfig(), FakeTranscriber({}), 1).check_sanity(result)  # off by default
        guarded = Evaluator(EvalConfig.model_validate({"sanity": {"max_baseline_wer": 0.6}}), FakeTranscriber({}), 1)
        with self.assertRaises(EvaluatorError) as ctx:
            guarded.check_sanity(result)
        self.assertIn("1.0000", str(ctx.exception))
        self.assertIn("0.6000", str(ctx.exception))

    def test_config_defaults_yaml_and_validation(self) -> None:
        c = EvalConfig()
        self.assertEqual((c.metrics, c.batch_size, c.generation.num_beams, c.thresholds.wer_below), (["wer", "cer"], 4, 1, [0.10, 0.20]))
        s = c.significance
        self.assertEqual((s.method, s.confidence, s.one_sided, s.n_resamples, s.min_effect_abs, s.min_groups), ("paired_group_bootstrap", 0.9, True, 2000, 0.0, 5))
        self.assertEqual((c.review.n_random, c.review.n_worst, c.sanity.max_baseline_wer, c.generation.max_new_tokens), (20, 10, None, None))
        loaded = load_config(ROOT / "configs" / "eval_default.yaml", env_file=None, target_environ={}, registry=SectionRegistry({"eval": EvalConfig}))
        self.assertEqual(loaded.section("eval", EvalConfig), c)
        for bad in ({"metrics": []}, {"batch_size": 0}, {"thresholds": {"wer_below": [1.5]}}, {"significance": {"confidence": 1.0}}, {"nope": 1}):
            with self.assertRaises(ValueError):
                EvalConfig.model_validate(bad)
        self.assertEqual(default_metric_registry().names(), ["cer", "wer"])
        self.assertEqual(default_significance_registry().names(), ["paired_group_bootstrap"])


class AdapterTests(unittest.TestCase):
    def dataset(self, n: int) -> Any:
        rng = np.random.default_rng(0)
        examples = [FeatureExample(f"s{i}", rng.standard_normal((80, 3000)).astype(np.float32) * 0.1, [1]) for i in range(n)]

        class Data:
            def __len__(self) -> int:
                return n

            def __getitem__(self, index: int) -> FeatureExample:
                return examples[index]

        return Data()

    def test_the_hugging_face_adapter_transcribes_in_batches_restores_the_mode_and_flags_the_cap(self) -> None:
        from stt.evaluator.hf_transcriber import HfTranscriber

        loaded = model()
        loaded.model.train()  # type: ignore[attr-defined]
        before = torch.get_num_threads()
        data = self.dataset(3)
        with mock.patch("socket.socket.connect", side_effect=AssertionError("network used")):
            out = HfTranscriber(threads=1).transcribe(loaded, data, DecodeSettings(1, 3, 2, 1.0, {"s0": 1, "s1": 3, "s2": 3}))
        self.assertEqual([h.sample_id for h in out], ["s0", "s1", "s2"])
        self.assertTrue(all(h.hit_length_cap for h in out))  # random weights never end a sentence within 3 tokens
        self.assertLessEqual(len(loaded.tokenizer.encode_label(out[0].text)) - 5, 1 + 1)  # the first output is cut at its own cap of 1 token
        self.assertTrue(loaded.model.training)  # type: ignore[attr-defined]
        self.assertEqual(torch.get_num_threads(), before)
        loaded.model.eval()  # type: ignore[attr-defined]

    def test_the_adapter_evaluates_end_to_end_and_maps_out_of_memory_to_an_error(self) -> None:
        from stt.evaluator.hf_transcriber import HfTranscriber

        loaded = model()
        items = [item(f"s{i}", "שלום עולם", f"g{i}") for i in range(3)]
        request = EvalRequest(items, self.dataset(3), loaded, "val", 1, EVAL_PROFILE, DecodeSettings(1, 4, 2))
        result = Evaluator(EvalConfig(), HfTranscriber(1), 1).evaluate(request)
        self.assertEqual(result.n_files, 3)
        self.assertGreater(result.seconds, 0)
        with mock.patch.object(loaded.model, "generate", side_effect=RuntimeError("DefaultCPUAllocator: not enough memory")):
            with self.assertRaises(EvaluatorError) as ctx:
                HfTranscriber(1).transcribe(loaded, self.dataset(2), DecodeSettings(1, 4, 2))
        self.assertIn("eval.batch_size", str(ctx.exception))


class ImportRuleTests(unittest.TestCase):
    def test_only_the_adapter_module_imports_torch_and_transformers(self) -> None:
        heavy = {"torch", "transformers", "accelerate", "safetensors"}
        offenders = []
        for path in sorted((ROOT / "src" / "stt" / "evaluator").glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                if any(m.split(".")[0] in heavy for m in modules) and path.name != "hf_transcriber.py":
                    offenders.append(path.name)
        self.assertEqual(offenders, [])

    def test_importing_the_package_does_not_import_torch(self) -> None:
        code = "import sys; sys.path.insert(0, 'src'); import stt.evaluator; print('torch' in sys.modules, 'transformers' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False False")


if __name__ == "__main__":
    unittest.main()
