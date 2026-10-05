"""Unit tests of the EvalResult / ComparisonResult JSON codec (stt.evaluator.codec): a result written and read again is EQUAL to the original, offline.

How to run (project root, virtual environment active):
    python -m unittest tests.test_evaluator_codec -v
One result is made by hand with every optional field set; one comes out of the real Evaluator with a scripted transcriber on the stored tiny test model.
Nothing is downloaded and no Test split is evaluated.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stt.evaluator import (  # noqa: E402
    Comparator,
    ComparisonResult,
    DecodeSettings,
    EvalConfig,
    EvalItem,
    EvalRequest,
    EvalResult,
    Evaluator,
    EvaluatorError,
    FileScore,
    RawHypothesis,
    ReviewEntry,
    ReviewList,
    comparison_from_json,
    comparison_to_json,
    eval_result_from_json,
    eval_result_to_json,
)
from stt.featurepipeline import FeatureExample  # noqa: E402
from stt.modelloader import LoadedModel, ModelConfig, ModelInfo, create_loader  # noqa: E402
from stt.textnormalizer import TextConfig, build_profile  # noqa: E402

TINY = ROOT / "tests" / "assets" / "tiny_whisper"


def by_hand() -> EvalResult:
    """A result with every optional field set: float keys, token caps, attributes of every type, an empty-output file with no CER, a review list."""
    files = (
        FileScore("a1", "g1", "שלום עולם טוב", "שלום עולם רע", "שלום עולם רע.", 3, 1, 1 / 3, 0.0833333333333333, False, False, {"duration": 12.5, "ref_words": 3, "flag_x": True, "note": "ok"}, {"conf": 0.25}),
        FileScore("a2", "g2", "אחת", "", "", 1, 1, 1.0, None, True, True, {}, {}),
    )
    entries = (ReviewEntry("a2", "g2", Path("audio") / "a2.mp3", "אחת", "", 1.0, "[-אחת-]", "worst"), ReviewEntry("a1", "g1", Path("audio") / "a1.mp3", "שלום עולם טוב", "שלום עולם רע", 1 / 3, "שלום עולם [טוב->רע]", "random"))
    return EvalResult(
        split_name="val",
        eval_version=3,
        model_info=ModelInfo("exp_0004_baseline", Path("models") / "whisper-tiny", 37_760_640, "float32", 80, "he", "transcribe", "exp_0004/config_full.yaml"),
        wer=0.5,
        wer_mean_per_file=0.6666666666666666,
        cer=0.1234567890123456,
        n_files=2,
        n_words=4,
        share_files_wer_below={0.1: 0.0, 0.3: 0.5, 0.5: 0.5},
        files=files,
        decode_settings=DecodeSettings(1, 440, 4, 8.0, {"a1": 100, "a2": 40}),
        seconds=12.3456789,
        flags={"empty_output": 1, "hit_length_cap": 1, "excluded_empty_reference": 0},
        excluded_ids=("z9",),
        review=ReviewList(entries),
        extra_metrics={"mer": 0.4},
    )


class FakeDataset:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> FeatureExample:
        return FeatureExample(self.ids[index], np.zeros((80, 3000), dtype=np.float32), [1])


class FakeTranscriber:
    def __init__(self, texts: Mapping[str, str]) -> None:
        self.texts = texts

    def transcribe(self, model: LoadedModel, dataset: Any, settings: DecodeSettings) -> list[RawHypothesis]:
        return [RawHypothesis(dataset[i].sample_id, self.texts[dataset[i].sample_id]) for i in range(len(dataset))]


def from_the_evaluator(texts: Mapping[str, str]) -> EvalResult:
    model = create_loader(ModelConfig(path=TINY)).load()
    reference = "אחת שתיים שלוש ארבע חמש"
    items = [EvalItem(f"g{g}f{f}", f"g{g}", reference, Path(f"audio/g{g}f{f}.mp3"), {"duration": 10.0 + f, "ref_words": 5}) for g in range(6) for f in range(2)]
    request = EvalRequest(items, FakeDataset([i.sample_id for i in items]), model, "val", 1, build_profile(TextConfig(), "eval"), DecodeSettings(1, 100, 4, None, {i.sample_id: 50 for i in items}))
    return Evaluator(EvalConfig(), FakeTranscriber({i.sample_id: texts[i.sample_id] for i in items}), seed=42).evaluate(request)


class RoundTripTests(unittest.TestCase):
    def test_a_hand_made_result_with_every_optional_field_comes_back_equal(self) -> None:
        original = by_hand()
        again = eval_result_from_json(eval_result_to_json(original))
        self.assertEqual(again, original)
        self.assertEqual(again.decode_settings.token_caps, {"a1": 100, "a2": 40})  # compare=False in the dataclass, so checked here
        self.assertEqual(list(again.share_files_wer_below), [0.1, 0.3, 0.5])  # the float keys and their order
        self.assertEqual(again.files[1].cer, None)
        self.assertEqual(again.review.entries[0].audio_path, Path("audio") / "a2.mp3")
        self.assertEqual(again.model_info.path, Path("models") / "whisper-tiny")
        self.assertIsInstance(again.files[0].attributes["ref_words"], int)  # types survive: int, float, bool, str
        self.assertIs(again.files[0].attributes["flag_x"], True)
        self.assertIsInstance(again.files[0].attributes["duration"], float)

    def test_numbers_are_kept_with_full_precision_and_the_text_is_written_twice_the_same(self) -> None:
        original = by_hand()
        text = eval_result_to_json(original)
        again = eval_result_from_json(text)
        self.assertEqual((again.wer_mean_per_file, again.cer, again.seconds), (original.wer_mean_per_file, original.cer, original.seconds))
        self.assertEqual(eval_result_to_json(again), text)  # a second encode gives the identical text

    def test_hebrew_is_written_as_readable_text_not_escapes(self) -> None:
        text = eval_result_to_json(by_hand())
        self.assertIn("שלום עולם", text)
        self.assertNotIn("\\u05", text)

    def test_a_result_of_the_real_evaluator_comes_back_equal_and_still_compares(self) -> None:
        texts = {f"g{g}f{f}": ("אחת שתיים שלוש ארבע חמש" if g % 2 else "אחת שתיים x ארבע y") for g in range(6) for f in range(2)}
        original = from_the_evaluator(texts)
        again = eval_result_from_json(eval_result_to_json(original))
        self.assertEqual(again, original)
        self.assertEqual(again.decode_settings.token_caps, original.decode_settings.token_caps)
        self.assertEqual(Comparator(EvalConfig(), 42).compare(again, again), Comparator(EvalConfig(), 42).compare(original, original))
        better = from_the_evaluator({k: "אחת שתיים שלוש ארבע חמש" for k in texts})
        a = Comparator(EvalConfig(), 42).compare(original, better)
        b = Comparator(EvalConfig(), 42).compare(eval_result_from_json(eval_result_to_json(original)), eval_result_from_json(eval_result_to_json(better)))
        self.assertEqual(a, b)  # the verdict made from stored results is the verdict made from the live ones


class ComparisonTests(unittest.TestCase):
    def test_every_verdict_shape_round_trips_including_the_missing_numbers(self) -> None:
        cases = [
            ComparisonResult(0.05, 0.01, 0.09, 0.9, True, 12, True, "improved", "better"),
            ComparisonResult(-0.02, -0.06, 0.02, 0.9, False, 9, False, "no_clear_difference", "unclear"),
            ComparisonResult(None, None, None, 0.9, True, 0, False, "not_comparable", "other files"),
            ComparisonResult(0.01, None, None, 0.9, True, 2, False, "insufficient_groups", "too few"),
        ]
        for original in cases:
            with self.subTest(verdict=original.verdict):
                self.assertEqual(comparison_from_json(comparison_to_json(original)), original)


class DamagedFileTests(unittest.TestCase):
    def test_a_damaged_or_foreign_text_fails_with_an_evaluator_error_that_says_what_to_do(self) -> None:
        good = json.loads(eval_result_to_json(by_hand()))
        missing = {k: v for k, v in good.items() if k != "files"}
        another_version = {**good, "schema_version": 99}
        another_codec = {**good, "codec": "something.else"}
        cases = {"not JSON": "{oops", "a list": "[1, 2]", "missing key": json.dumps(missing), "another version": json.dumps(another_version), "another codec": json.dumps(another_codec)}
        for name, text in cases.items():
            with self.subTest(case=name), self.assertRaises(EvaluatorError) as ctx:
                eval_result_from_json(text)
            self.assertTrue(any(word in str(ctx.exception) for word in ("evaluate stage", "document")), str(ctx.exception))
        with self.assertRaises(EvaluatorError):
            comparison_from_json(eval_result_to_json(by_hand()))  # an EvalResult text is not a comparison
        with self.assertRaises(EvaluatorError):
            eval_result_from_json(comparison_to_json(ComparisonResult(0.0, 0.0, 0.0, 0.9, True, 1, False, "improved", "x")))

    def test_the_codec_module_imports_no_model_library(self) -> None:
        code = "import sys; sys.path.insert(0, 'src'); import stt.evaluator.codec; print('torch' in sys.modules, 'transformers' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False False")


if __name__ == "__main__":
    unittest.main()
