"""Unit tests of the SampleFilter component (stt.samplefilter): the acceptance tests of the C1b brief, offline.

How to run (project root, virtual environment active):
    python -m unittest tests.test_samplefilter -v
The tests use small synthetic samples in memory and a fake token counter; no file is read, no tokenizer is imported.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stt.config import ConfigError, SectionRegistry, load_config  # noqa: E402
from stt.datasource import AudioSample  # noqa: E402
from stt.samplefilter import (  # noqa: E402
    FilterConfig,
    FilterError,
    FilterGateError,
    MaxBadSegments,
    MaxDuration,
    MaxLabelTokens,
    MinDuration,
    MinQualityScore,
    MinTextChars,
    RequireGolden,
    RuleRegistry,
    RuleSpec,
    RuleVerdict,
    SampleFilter,
    default_registry,
)
from stt.textnormalizer import NormalizedSample  # noqa: E402


def ns(i: str, duration: float = 10.0, eval_text: str = "שלום עולם", train_text: str | None = None, **extra: Any) -> NormalizedSample:
    audio = AudioSample(i, Path(f"{i}.wav"), "raw", "g", duration, None, extra)
    return NormalizedSample(audio, eval_text if train_text is None else train_text, eval_text, None, ())


def conf(**overrides: Any) -> FilterConfig:
    return FilterConfig.model_validate(overrides)


def ids(items: Any) -> list[str]:
    return [getattr(x, "sample_id", None) or x.sample.id for x in items]


class RuleTests(unittest.TestCase):
    """Test 1: each rule passes and fails the right samples (table-driven)."""

    def table(self, rule: Any, cases: list[tuple[NormalizedSample, bool]]) -> None:
        for sample, expected in cases:
            with self.subTest(sample=sample.sample.id):
                self.assertEqual(rule.check(sample).passed, expected)

    def test_min_duration(self) -> None:
        self.table(MinDuration(1.0), [(ns("a", 0.99), False), (ns("b", 1.0), True), (ns("c", 29.0), True)])

    def test_max_duration_with_float_tolerance(self) -> None:
        self.table(MaxDuration(30.0), [(ns("a", 30.0), True), (ns("b", 30.0000001), True), (ns("c", 30.01), False), (ns("d", 3.0), True)])

    def test_min_text_chars_measures_the_chosen_text(self) -> None:
        self.table(MinTextChars(2), [(ns("a", eval_text="א"), False), (ns("b", eval_text="לא"), True), (ns("c", eval_text="  "), False), (ns("d", eval_text=""), False)])
        on_train = MinTextChars(2, "train_text")
        self.assertTrue(on_train.check(ns("e", eval_text="", train_text="[רעש]")).passed)
        self.assertFalse(MinTextChars(2).check(ns("e", eval_text="", train_text="[רעש]")).passed)

    def test_min_quality_score(self) -> None:
        rule = MinQualityScore(0.3, "q")
        self.table(rule, [(ns("a", q=0.29), False), (ns("b", q=0.3), True), (ns("c", q=1), True), (ns("d", q="x"), False), (ns("e", q=True), False)])
        self.assertIn("below", rule.check(ns("a", q=0.1)).reason)

    def test_max_bad_segments(self) -> None:
        self.table(MaxBadSegments(2, "b"), [(ns("a", b=3), False), (ns("b", b=2), True), (ns("c", b=0), True)])

    def test_require_golden(self) -> None:
        self.table(RequireGolden("g"), [(ns("a", g=True), True), (ns("b", g=False), False), (ns("c", g="True"), False)])

    def test_max_label_tokens_uses_the_injected_counter_on_train_text(self) -> None:
        seen: list[str] = []

        def counter(text: str) -> int:
            seen.append(text)
            return len(text.split())

        rule = MaxLabelTokens(3, counter)
        self.table(rule, [(ns("a", train_text="א ב ג"), True), (ns("b", train_text="א ב ג ד"), False)])
        self.assertEqual(seen, ["א ב ג", "א ב ג ד"])
        self.assertEqual(rule.check(ns("c", eval_text="א", train_text="א ב ג ד ה")).reason, "train_text has 5 tokens (maximum 3)")


class FilterBehaviourTests(unittest.TestCase):
    def data(self) -> list[NormalizedSample]:
        return [
            ns("a"),
            ns("b", duration=0.5),
            ns("c", eval_text="א"),
            ns("d", duration=31.0),
            ns("e", duration=0.5, eval_text=""),
            ns("f"),
        ]

    def test_2_report_mode_keeps_everything_but_counts_would_be_rejections(self) -> None:
        result = SampleFilter(conf(min_kept_ratio=0.3)).filter(self.data(), "train")
        self.assertEqual(ids(result.kept), ["a", "b", "c", "d", "e", "f"])
        self.assertEqual(result.rejected, [])
        self.assertEqual(ids(result.would_reject), ["b", "c", "d", "e"])
        report = result.report
        self.assertEqual((report.mode, report.seen, report.kept, report.would_reject), ("report", 6, 6, 4))
        self.assertAlmostEqual(report.kept_ratio, 2 / 6)
        counts = {r.name: r.failed for r in report.rules_run}
        self.assertEqual(counts, {"min_duration": 2, "max_duration": 1, "min_text_chars": 2})

    def test_2_apply_mode_removes_and_names_every_failed_rule(self) -> None:
        result = SampleFilter(conf(mode="apply", min_kept_ratio=0.0)).filter(self.data(), "train")
        self.assertEqual(ids(result.kept), ["a", "f"])
        self.assertEqual(ids(result.rejected), ["b", "c", "d", "e"])
        by_id = {r.sample_id: r for r in result.rejected}
        self.assertEqual(by_id["e"].rules, ("min_duration", "min_text_chars"))
        self.assertEqual(len(by_id["e"].reasons), 2)
        self.assertIn("0.50 s", by_id["b"].reasons[0])
        self.assertEqual(result.report.kept, 2)

    def test_3_applies_to_train_does_not_run_for_test_and_quality_default_is_train_only(self) -> None:
        config = conf(rules={"min_quality_score": {"threshold": 0.5, "field": "q"}, "min_duration": {"threshold": 5, "applies_to": ["train"]}})
        data = [ns("a", q=0.1, duration=2.0), ns("b", q=0.9)]
        filt = SampleFilter(config)
        train = filt.filter(data, "train")
        self.assertEqual(ids(train.would_reject), ["a"])
        self.assertEqual({r.name for r in train.report.rules_run}, {"min_duration", "max_duration", "min_text_chars", "min_quality_score"})
        test = filt.filter(data, "test")
        self.assertEqual(test.would_reject, [])
        self.assertEqual({r.name for r in test.report.rules_run}, {"max_duration", "min_text_chars"})
        self.assertEqual(set(test.report.rules_not_applied), {"min_duration", "min_quality_score"})

    def test_kind_override_and_star_apply_a_quality_rule_everywhere(self) -> None:
        config = conf(rules={"max_bad_segments": {"threshold": 1, "field": "b", "applies_to": ["*"]}})
        result = SampleFilter(config).filter([ns("a", b=5), ns("b", b=0)], "test")
        self.assertEqual(ids(result.would_reject), ["a"])
        config = conf(rules={"min_duration": {"kind": "quality", "threshold": 5}})
        self.assertEqual(SampleFilter(config).filter([ns("a", duration=1.0), ns("b")], "val").would_reject, [])
        self.assertEqual(ids(SampleFilter(config).filter([ns("a", duration=1.0), ns("b")], "train").would_reject), ["a"])

    def test_4_missing_field_fails_naming_rule_and_field_and_disabling_removes_the_failure(self) -> None:
        config = conf(rules={"min_quality_score": {"threshold": 0.2, "field": "meta_quality_score"}})
        with self.assertRaises(FilterError) as ctx:
            SampleFilter(config).filter([ns("a"), ns("b")], "train")
        message = str(ctx.exception)
        self.assertIn("min_quality_score", message)
        self.assertIn("extra:meta_quality_score", message)
        self.assertIn("enabled: false", message)
        disabled = conf(rules={"min_quality_score": {"threshold": 0.2, "field": "meta_quality_score", "enabled": False}})
        self.assertEqual(SampleFilter(disabled).filter([ns("a"), ns("b")], "train").would_reject, [])
        SampleFilter(config).filter([ns("a")], "test")  # not run for test: no field needed

    def test_enabled_rule_without_threshold_field_or_counter_fails_when_built(self) -> None:
        for rules, expected in (
            ({"min_quality_score": {"enabled": True, "field": "q"}}, "no threshold"),
            ({"min_quality_score": {"threshold": 1}}, "field is not set"),
            ({"require_golden": {"enabled": True}}, "field is not set"),
            ({"max_label_tokens": {"threshold": 440}}, "token counter"),
        ):
            with self.subTest(rules=rules):
                with self.assertRaises(FilterError) as ctx:
                    SampleFilter(conf(rules=rules))
                self.assertIn(expected, str(ctx.exception))

    def test_5_min_kept_ratio_gate_fails_with_per_rule_breakdown_and_keeps_the_result(self) -> None:
        data = [ns("a"), *(ns(f"s{i}", duration=0.1) for i in range(3))]
        with self.assertRaises(FilterGateError) as ctx:
            SampleFilter(conf(min_kept_ratio=0.5)).filter(data, "train")
        message = str(ctx.exception)
        self.assertIn("min_kept_ratio", message)
        self.assertIn("min_duration: 3", message)
        self.assertEqual(ctx.exception.result.report.would_reject, 3)
        SampleFilter(conf(min_kept_ratio=0.25)).filter(data, "train")  # exactly at the limit passes

    def test_gate_checks_the_would_be_ratio_in_report_mode_too(self) -> None:
        with self.assertRaises(FilterGateError):
            SampleFilter(conf(mode="report")).filter([ns("a", duration=0.1), ns("b", duration=0.1)], "val")

    def test_empty_split_after_apply_names_the_split_and_dominant_rule(self) -> None:
        with self.assertRaises(FilterGateError) as ctx:
            SampleFilter(conf(mode="apply", min_kept_ratio=0.0)).filter([ns("a", duration=0.1), ns("b", duration=0.2, eval_text="")], "test")
        message = str(ctx.exception)
        self.assertIn("'test'", message)
        self.assertIn("min_duration", message)

    def test_empty_input_returns_an_empty_result(self) -> None:
        result = SampleFilter(conf(mode="apply")).filter([], "train")
        self.assertEqual((result.kept, result.rejected, result.report.seen, result.report.kept_ratio), ([], [], 0, 1.0))

    def test_6_kept_order_equals_input_order_and_inputs_are_unmodified(self) -> None:
        data = [ns("z"), ns("m", duration=0.1), ns("a"), ns("k")]
        before = [(s.sample.id, s.sample.duration_sec, s.train_text, s.eval_text) for s in data]
        result = SampleFilter(conf(mode="apply", min_kept_ratio=0.0)).filter(data, "train")
        self.assertEqual(ids(result.kept), ["z", "a", "k"])
        self.assertTrue(all(any(k is d for d in data) for k in result.kept))
        self.assertEqual(before, [(s.sample.id, s.sample.duration_sec, s.train_text, s.eval_text) for s in data])

    def test_filter_splits_runs_every_split(self) -> None:
        results = SampleFilter(conf(mode="apply")).filter_splits({"train": [ns("a"), ns("b")], "test": [ns("c")]})
        self.assertEqual(list(results), ["train", "test"])
        self.assertEqual(ids(results["test"].kept), ["c"])

    def test_7_a_new_rule_class_works_through_the_registry_without_editing_the_filter(self) -> None:
        class MaxWords:
            name = "max_words"
            kind = "structural"
            required_fields = ("eval_text",)

            def __init__(self, limit: float) -> None:
                self.limit = limit

            def check(self, sample: NormalizedSample) -> RuleVerdict:
                n = len(sample.eval_text.split())
                return RuleVerdict(n <= self.limit, f"{n} words")

        base = default_registry()
        specs = {n: base.spec(n) for n in base.names()}
        specs["max_words"] = RuleSpec("structural", False, None, None, True, False, lambda p: MaxWords(p.threshold or 0))
        registry = RuleRegistry(specs)
        filt = SampleFilter(conf(rules={"max_words": {"threshold": 2}}), registry=registry)
        result = filt.filter([ns("a", eval_text="א ב"), ns("b", eval_text="א ב ג")], "train")
        self.assertEqual(ids(result.would_reject), ["b"])
        self.assertEqual(result.would_reject[0].reasons, ("3 words",))

    def test_8_unknown_rule_name_fails_and_lists_registered_names(self) -> None:
        with self.assertRaises(FilterError) as ctx:
            SampleFilter(conf(rules={"min_duraton": {"threshold": 1}}))
        message = str(ctx.exception)
        self.assertIn("min_duraton", message)
        self.assertIn("max_label_tokens, min_duration", message.replace("max_duration, ", ""))
        self.assertEqual(len(default_registry().names()), 7)

    def test_token_rule_with_a_fake_counter_in_the_filter(self) -> None:
        filt = SampleFilter(conf(rules={"max_label_tokens": {"threshold": 3}}), token_counter=lambda t: len(t.split()))
        result = filt.filter([ns("a", train_text="א ב"), ns("b", train_text="א ב ג ד")], "val")
        self.assertEqual(ids(result.would_reject), ["b"])
        self.assertIn("max_label_tokens", filt.active_rules)
        self.assertNotIn("max_label_tokens", SampleFilter(conf()).active_rules)


class DefaultsAndConfigTests(unittest.TestCase):
    def load(self, *overrides: str) -> FilterConfig:
        config = load_config(overrides=list(overrides), registry=SectionRegistry({"filter": FilterConfig}), environ={}, env_file=None, target_environ={})
        return config.section("filter", FilterConfig)

    def test_defaults_are_permissive_and_match_the_brief(self) -> None:
        c = self.load()
        self.assertEqual((c.mode, c.min_kept_ratio, c.rules), ("report", 0.4, {}))
        self.assertEqual(SampleFilter(c).active_rules, ("max_duration", "min_duration", "min_text_chars"))

    def test_default_thresholds_are_the_brief_values(self) -> None:
        filt = SampleFilter(conf())
        result = filt.filter([ns("a", duration=1.0), ns("b", duration=0.99), ns("c", duration=30.0), ns("d", duration=30.1), ns("e", eval_text="אב"), ns("f", eval_text="א")], "train")
        self.assertEqual(ids(result.would_reject), ["b", "d", "f"])

    def test_partial_override_keeps_the_other_rules(self) -> None:
        c = self.load("filter.rules.min_duration.threshold=5")
        filt = SampleFilter(c)
        self.assertEqual(filt.active_rules, ("max_duration", "min_duration", "min_text_chars"))
        self.assertEqual(ids(filt.filter([ns("a", duration=4.0), ns("b", duration=31.0), ns("c", duration=6.0), ns("d", duration=7.0)], "train").would_reject), ["a", "b"])

    def test_threshold_enables_a_quality_rule_and_enabled_false_disables_a_default_rule(self) -> None:
        c = self.load("filter.rules.min_quality_score.threshold=0.3", "filter.rules.min_quality_score.field=q", "filter.rules.min_text_chars.enabled=false")
        self.assertEqual(SampleFilter(c).active_rules, ("max_duration", "min_duration", "min_quality_score"))

    def test_invalid_values_fail_at_config_validation(self) -> None:
        for bad, expected in (
            ("filter.rules.min_duration.threshold=-1", "greater than or equal to 0"),
            ("filter.rules.min_duration.threshold=abc", "threshold"),
            ("filter.mode=maybe", "filter.mode"),
            ("filter.min_kept_ratio=1.5", "min_kept_ratio"),
            ("filter.rules.min_duration.nonsense=1", "Unknown key"),
            ("filter.rules.min_duration.kind=fancy", "kind"),
            ("filter.rules.min_text_chars.measure_on=raw", "measure_on"),
        ):
            with self.assertRaises(ConfigError, msg=bad) as ctx:
                self.load(bad)
            self.assertIn(expected, str(ctx.exception))

    def test_shipped_example_config_loads_and_builds_a_filter(self) -> None:
        config = load_config(ROOT / "configs" / "filter_default.yaml", registry=SectionRegistry({"filter": FilterConfig}), environ={}, env_file=None, target_environ={})
        filt = SampleFilter(config.section("filter", FilterConfig))
        self.assertEqual(filt.active_rules, ("max_duration", "min_duration", "min_text_chars"))


class ImportRuleTests(unittest.TestCase):
    def test_no_file_access_no_network_no_tokenizer_imports(self) -> None:
        with mock.patch("builtins.open", side_effect=AssertionError("no file access")), mock.patch("pathlib.Path.open", side_effect=AssertionError("no file access")):
            SampleFilter(conf(mode="apply", min_kept_ratio=0.0)).filter([ns("a"), ns("b", duration=0.1)], "train")
        forbidden = {"requests", "urllib3", "httpx", "huggingface_hub", "socket", "tools", "soundfile", "pyarrow", "scipy", "torch", "transformers", "tokenizers"}
        for path in (ROOT / "src" / "stt" / "samplefilter").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    self.assertNotIn(name.split(".")[0], forbidden, f"{path} imports {name}")


if __name__ == "__main__":
    unittest.main()
