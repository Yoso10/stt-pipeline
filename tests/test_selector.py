"""Unit tests of the Selector component (stt.selector): the acceptance tests of the C9 brief, offline, with fake EvalResults and a fake one-time Test evaluation.

How to run (project root, virtual environment active):
    python -m unittest tests.test_selector -v
The EvalResults are made by the real Evaluator from scripted outputs (tests/selector_fakes.py), so the real Comparator compares them. The Test callable is a fake
that counts its calls. No file, no model weights and no network are used.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from selector_fakes import FakeTest, candidate, make_result, must_not_be_called  # noqa: E402

from stt.config import SectionRegistry, load_config  # noqa: E402
from stt.evaluator import Comparator, ComparisonResult, EvalConfig, EvaluatorError  # noqa: E402
from stt.selector import (  # noqa: E402
    SelectConfig,
    SelectionRequest,
    Selector,
    SelectorError,
    TestAlreadyEvaluatedError,
    TestOnceGuard,
)

COMPARATOR = Comparator(EvalConfig(), 42)


def request(baseline_errors: int, candidates: list, test, **kw) -> SelectionRequest:  # type: ignore[no-untyped-def]
    return SelectionRequest(make_result("val", baseline_errors, name="baseline"), candidates, kw.pop("comparator", COMPARATOR), test, **kw)


def select(req: SelectionRequest, **config):  # type: ignore[no-untyped-def]
    return Selector(SelectConfig.model_validate(config)).select(req)


class WinnerTests(unittest.TestCase):
    def test_1_a_clearly_better_candidate_wins_and_appears_first_in_the_table(self) -> None:
        test = FakeTest({None: 4, 1: 1})
        result = select(request(4, [candidate(0, 4), candidate(1, 1), candidate(2, 3)], test))
        self.assertEqual((result.verdict, result.winner.trial_id), ("winner", 1))  # type: ignore[union-attr]
        self.assertEqual([row.trial_id for row in result.comparison_table], [1, 2, 0])  # the winner first, then the eligible, then the others
        self.assertEqual((result.comparison_table[0].note, result.comparison_table[0].rank), ("WINNER", 1))
        self.assertTrue(result.ranking[0].eligible)
        self.assertIn("is the winner", result.reason)

    def test_2_no_eligible_candidate_gives_baseline_retained_and_the_test_is_never_evaluated(self) -> None:
        result = select(request(4, [candidate(0, 4), candidate(1, 6)], must_not_be_called))  # equal and worse
        self.assertEqual((result.verdict, result.winner, result.test_status, result.test_comparison), ("baseline_retained", None, "not_evaluated", None))
        self.assertIn("The baseline stays", result.reason)
        self.assertIn("legitimate outcome", result.reason)
        self.assertEqual(len(result.comparison_table), 2)  # every finalist is still in the table
        self.assertFalse(any(r.eligible for r in result.ranking))

    def test_3_a_tie_within_the_epsilon_is_decided_by_the_lower_val_loss(self) -> None:
        a, b = candidate(0, 1, val_loss=0.9), candidate(1, 2, val_loss=0.5)  # Val WER 0.10 and 0.20
        wide = select(request(5, [a, b], FakeTest({None: 5, 1: 2, 0: 1})), tie_epsilon=0.15)
        self.assertEqual(wide.winner.trial_id, 1)  # type: ignore[union-attr]  # a tie: the lower Val loss wins although its WER is higher
        self.assertIn("tie margin", wide.reason + wide.ranking[0].reason)
        narrow = select(request(5, [a, b], FakeTest({None: 5, 0: 1})), tie_epsilon=0.001)
        self.assertEqual(narrow.winner.trial_id, 0)  # type: ignore[union-attr]  # no tie: the lowest Val WER wins
        same = select(request(5, [candidate(3, 1, val_loss=0.5), candidate(2, 1, val_loss=0.5)], FakeTest({None: 5, 2: 1})), tie_epsilon=0.001)
        self.assertEqual(same.winner.trial_id, 2)  # type: ignore[union-attr]  # an exact tie in WER and loss: the lower trial id (deterministic)

    def test_4_val_improved_and_test_no_clear_difference_keeps_the_winner_and_flags_it(self) -> None:
        result = select(request(4, [candidate(0, 1)], FakeTest({None: 4, 0: 4})))
        self.assertEqual((result.verdict, result.winner.trial_id, result.test_status), ("winner", 0, "not_confirmed"))  # type: ignore[union-attr]
        self.assertEqual(result.test_comparison.verdict, "no_clear_difference")  # type: ignore[union-attr]
        self.assertIn("does NOT confirm", result.reason)
        self.assertIn("FLAGGED", result.reason)
        self.assertAlmostEqual(result.test_winner.wer, 0.4)  # type: ignore[union-attr]  # the Test numbers are beside the Val numbers
        self.assertAlmostEqual(result.test_baseline.wer, 0.4)  # type: ignore[union-attr]
        worse = select(request(4, [candidate(0, 1)], FakeTest({None: 4, 0: 7})))
        self.assertEqual((worse.winner.trial_id, worse.test_status, worse.test_comparison.verdict), (0, "not_confirmed", "worse"))  # type: ignore[union-attr]

    def test_the_test_confirms_a_real_winner(self) -> None:
        result = select(request(4, [candidate(0, 1)], FakeTest({None: 4, 0: 2})))
        self.assertEqual((result.test_status, result.test_comparison.verdict), ("confirmed", "improved"))  # type: ignore[union-attr]
        self.assertIn("The Test confirms it", result.reason)

    def test_5_the_test_callable_is_called_once_for_the_winner_and_once_for_the_baseline_and_a_second_call_fails(self) -> None:
        test = FakeTest({None: 4, 1: 1})
        result = select(request(4, [candidate(0, 4), candidate(1, 1)], test))
        self.assertEqual(test.calls, [1, None])  # exactly once for the winner and once for the baseline
        self.assertEqual(result.test_status, "confirmed")
        guard = TestOnceGuard(FakeTest({None: 4, 1: 1}))
        select(request(4, [candidate(1, 1)], guard))
        self.assertEqual(guard.calls, ["trial_1", "baseline"])
        with self.assertRaises(TestAlreadyEvaluatedError) as ctx:  # a second selection run with the same guard: the Test was already looked at
            select(request(4, [candidate(1, 1)], guard))
        self.assertIn("ONCE", str(ctx.exception))
        with self.assertRaises(TestAlreadyEvaluatedError):
            guard(None)
        failing = TestOnceGuard(FakeTest({}, fail=RuntimeError("boom")))
        with self.assertRaises(RuntimeError):
            failing(None)
        with self.assertRaises(TestAlreadyEvaluatedError):  # a failed Test evaluation also counts as a look at the Test
            failing(None)

    def test_the_test_is_never_asked_for_a_split_other_than_the_configured_one_and_never_for_val_results(self) -> None:
        with self.assertRaises(SelectorError) as ctx:
            select(request(4, [candidate(0, 1)], FakeTest({None: 4, 0: 1}, split="val")))
        self.assertIn("'test'", str(ctx.exception))
        custom = select(request(4, [candidate(0, 1)], FakeTest({None: 4, 0: 1}, split="holdout"), test_split="holdout"))
        self.assertEqual(custom.test_status, "confirmed")
        test_named = replace(candidate(0, 1), val=make_result("test", 1))
        with self.assertRaises(SelectorError) as ctx2:
            select(request(4, [test_named], must_not_be_called))
        self.assertIn("Val result", str(ctx2.exception))
        with self.assertRaises(SelectorError):
            select(SelectionRequest(make_result("test", 4), [candidate(0, 1)], COMPARATOR, must_not_be_called))

    def test_a_failing_test_evaluation_keeps_the_winner_and_reports_the_error(self) -> None:
        result = select(request(4, [candidate(0, 1)], FakeTest({}, fail=RuntimeError("out of memory"))))
        self.assertEqual((result.verdict, result.test_status, result.winner.trial_id), ("winner", "not_evaluated", 0))  # type: ignore[union-attr]
        self.assertIn("out of memory", result.test_error or "")
        self.assertIn("FAILED", result.reason)

    def test_test_confirmation_false_does_not_evaluate_the_test(self) -> None:
        result = select(request(4, [candidate(0, 1)], must_not_be_called), test_confirmation=False)
        self.assertEqual((result.verdict, result.test_status), ("winner", "not_evaluated"))
        self.assertIn("switched off", result.reason)


class ComparabilityTests(unittest.TestCase):
    def test_6_a_different_eval_version_or_file_set_fails_and_not_comparable_candidates_are_ineligible(self) -> None:
        other_version = replace(candidate(0, 1), val=make_result("val", 1, version=2))
        with self.assertRaises(SelectorError) as ctx:
            select(request(4, [other_version], must_not_be_called))
        self.assertIn("eval_version 2", str(ctx.exception))
        self.assertIn("baseline uses 1", str(ctx.exception))
        other_files = candidate(0, 1, groups=8)
        with self.assertRaises(SelectorError) as ctx2:
            select(request(4, [other_files], must_not_be_called))
        self.assertIn("not scored on the same files", str(ctx2.exception))

        class NotComparable:
            def compare(self, baseline, cand):  # type: ignore[no-untyped-def]
                return ComparisonResult(None, None, None, 0.9, True, 0, False, "not_comparable", "different profiles")

        result = select(request(4, [candidate(0, 1)], must_not_be_called, comparator=NotComparable()))
        self.assertEqual(result.verdict, "baseline_retained")
        self.assertFalse(result.ranking[0].eligible)
        self.assertIn("not comparable", result.ranking[0].reason)

    def test_7_insufficient_groups_follows_the_configured_policy(self) -> None:
        few = candidate(0, 1, groups=3)
        base = make_result("val", 4, groups=3, name="baseline")
        req = SelectionRequest(base, [few], COMPARATOR, FakeTest({None: 4, 0: 1}, groups=3))
        plain = select(req)
        self.assertEqual((plain.verdict, plain.winner.trial_id), ("winner", 0))  # type: ignore[union-attr]
        self.assertIn("no confidence statement", plain.comparison_table[0].note)
        self.assertTrue(any("only 3 group(s)" in w for w in plain.warnings))
        self.assertIn("plain WER difference", plain.ranking[0].reason)
        rejected = select(SelectionRequest(base, [few], COMPARATOR, must_not_be_called), on_insufficient_groups="reject")
        self.assertEqual(rejected.verdict, "baseline_retained")
        self.assertIn("select.on_insufficient_groups is reject", rejected.ranking[0].reason)
        worse = select(SelectionRequest(base, [candidate(1, 7, groups=3)], COMPARATOR, must_not_be_called))
        self.assertEqual(worse.verdict, "baseline_retained")  # a plain difference that is negative is not eligible

    def test_min_verdict_no_clear_difference_makes_equal_candidates_eligible(self) -> None:
        result = select(request(4, [candidate(0, 4)], FakeTest({None: 4, 0: 4})), min_verdict="no_clear_difference")
        self.assertEqual((result.verdict, result.winner.trial_id), ("winner", 0))  # type: ignore[union-attr]

    def test_selection_failures_name_the_cause(self) -> None:
        with self.assertRaises(SelectorError) as ctx:
            select(request(4, [], must_not_be_called))
        self.assertIn("no finalist", str(ctx.exception))
        with self.assertRaises(SelectorError) as ctx2:
            select(request(4, [candidate(1, 1), candidate(1, 2)], must_not_be_called))
        self.assertIn("Duplicate trial ids", str(ctx2.exception))
        self.assertIn("[1]", str(ctx2.exception))
        no_val = replace(candidate(0, 1), val=None)  # type: ignore[arg-type]
        with self.assertRaises(SelectorError) as ctx3:
            select(request(4, [no_val], must_not_be_called))
        self.assertIn("trial 0 has no Val evaluation", str(ctx3.exception))

        class Broken:
            def compare(self, baseline, cand):  # type: ignore[no-untyped-def]
                raise EvaluatorError("the files differ")

        with self.assertRaises(SelectorError):
            select(request(4, [candidate(0, 1)], must_not_be_called, comparator=Broken()))


class TableAndPassThroughTests(unittest.TestCase):
    def test_8_the_table_has_every_finalist_with_all_columns_in_a_fixed_order(self) -> None:
        cands = [candidate(0, 4, val_loss=0.9), candidate(1, 2, val_loss=0.8), candidate(2, 1, val_loss=0.7), candidate(3, 6, val_loss=0.6)]
        result = select(request(4, cands, FakeTest({None: 4, 2: 1})))
        table = result.comparison_table
        self.assertEqual([row.trial_id for row in table], [2, 1, 0, 3])  # winner, then the other eligible by Val WER, then the rest by Val WER
        self.assertEqual([row.rank for row in table], [1, 2, 3, 4])
        row = table[0]
        fields = {"rank", "trial_id", "params", "val_wer", "val_cer", "delta_wer", "ci_low", "ci_high", "verdict", "val_loss", "steps", "eligible", "note"}
        self.assertEqual(set(row.__dataclass_fields__), fields)
        self.assertEqual((row.trial_id, row.verdict, row.val_loss, row.steps, row.params["grad_accum_steps"]), (2, "improved", 0.7, 100, 4))
        self.assertAlmostEqual(float(row.params["learning_rate"]), 3e-05)
        self.assertAlmostEqual(row.val_wer, 0.1)
        self.assertAlmostEqual(row.delta_wer or 0, 0.3)
        self.assertIsNotNone(row.ci_low)
        self.assertIsNotNone(row.val_cer)
        self.assertEqual([r.eligible for r in table], [True, True, False, False])
        again = select(request(4, cands, FakeTest({None: 4, 2: 1})))
        self.assertEqual(table, again.comparison_table)  # deterministic

    def test_9_the_review_list_is_passed_through_unchanged(self) -> None:
        cands = [candidate(0, 1), candidate(1, 3)]
        result = select(request(4, cands, FakeTest({None: 4, 0: 1})))
        self.assertIs(result.review_list, cands[0].val.review)  # the winner's list, the very same object
        stays = request(4, [candidate(5, 4)], must_not_be_called)
        self.assertIs(select(stays).review_list, stays.baseline_val.review)  # when the baseline stays: the baseline's list

    def test_config_defaults_and_the_yaml_example(self) -> None:
        c = SelectConfig()
        self.assertEqual((c.min_verdict, c.tie_epsilon, c.on_insufficient_groups, c.test_confirmation), ("improved", 0.001, "plain_difference", True))
        loaded = load_config(ROOT / "configs" / "select_default.yaml", env_file=None, target_environ={}, registry=SectionRegistry({"select": SelectConfig}))
        self.assertEqual(loaded.section("select", SelectConfig), c)
        for bad in ({"min_verdict": "worse"}, {"tie_epsilon": -1}, {"on_insufficient_groups": "ignore"}, {"nope": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                SelectConfig.model_validate(bad)


class ImportRuleTests(unittest.TestCase):
    def test_10_no_file_access_no_network_and_no_model_imports(self) -> None:
        forbidden = {"torch", "transformers", "socket", "urllib", "requests", "shutil", "tempfile", "subprocess"}
        calls = []
        for path in sorted((ROOT / "src" / "stt" / "selector").glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                if any(m.split(".")[0] in forbidden for m in modules):
                    calls.append((path.name, modules))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"open", "input"}:
                    calls.append((path.name, node.func.id))
                if isinstance(node, ast.Attribute) and node.attr in {"read_text", "write_text", "read_bytes", "write_bytes", "mkdir", "unlink"}:
                    calls.append((path.name, node.attr))
        self.assertEqual(calls, [])
        code = "import sys; sys.path.insert(0, 'src'); import stt.selector; print('torch' in sys.modules, 'transformers' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False False")
        with mock.patch("socket.socket.connect", side_effect=AssertionError("network used")):
            select(request(4, [candidate(0, 1)], FakeTest({None: 4, 0: 2})))


if __name__ == "__main__":
    unittest.main()
