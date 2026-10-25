"""Tests of the convergence view of the report (DECISIONS 90): the verdict from the loss curves, the rules "what to change next", the page of a tiny `train` + `evaluate` experiment.

How to run (project root, virtual environment active):
    python -m unittest tests.test_convergence -v
Offline. The verdict logic is tested on synthetic curves; the page is made from a real tiny experiment (synthetic clips, the stored tiny test model, about 15 seconds).
"""

from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import tiny_flow  # noqa: E402
from stt.evaluator import ComparisonResult  # noqa: E402
from stt.reportbuilder import (  # noqa: E402
    ABSOLUTE_WER_TARGET_BAND,
    ClipRow,
    FinalTestView,
    ModelNumbers,
    ReportBuilder,
    ReportConfig,
    ReportData,
    RunInfoView,
    TrainedRunView,
    verdict_from_points,
)
from stt.reportbuilder import convergence  # noqa: E402
from stt.trainercore import StepMetrics  # noqa: E402


def record(step: int, train: float | None = None, val: float | None = None) -> StepMetrics:
    return StepMetrics(step=step, epoch_fraction=0.0, train_loss=train, val_loss=val, learning_rate=1e-5, grad_norm=1.0, seconds_per_step=1.0, elapsed_sec=float(step), eta_sec=None)


class VerdictTests(unittest.TestCase):
    def test_still_improving_when_the_last_point_is_the_lowest_and_fell_by_more_than_the_margin(self) -> None:
        v = verdict_from_points([(25, 2.0), (50, 1.5), (75, 1.2), (100, 1.0)], [(25, 2.2), (100, 1.1)])
        self.assertEqual((v.kind, v.step), ("improving", 100))
        self.assertIn("Still improving", v.english)
        self.assertIn("עדיין משתפר", v.hebrew)

    def test_plateau_when_the_end_changes_by_less_than_the_margin(self) -> None:
        v = verdict_from_points([(25, 1.0), (50, 0.995), (75, 0.99), (100, 0.9895)], [(25, 1.5), (100, 1.4)])
        self.assertEqual(v.kind, "plateau")
        self.assertIn("Plateau", v.english)

    def test_the_margin_is_a_named_constant_and_a_fall_below_it_is_a_plateau(self) -> None:
        self.assertEqual(convergence.STILL_IMPROVING_MARGIN, 0.01)
        self.assertEqual(verdict_from_points([(1, 2.0), (2, 1.0), (3, 0.995)], []).kind, "plateau")  # fell by 0.5%
        self.assertEqual(verdict_from_points([(1, 2.0), (2, 1.0), (3, 0.98)], []).kind, "improving")  # fell by 2%

    def test_overfitting_when_val_rose_after_the_best_step_while_train_fell(self) -> None:
        v = verdict_from_points([(25, 2.0), (50, 1.0), (75, 1.3), (100, 1.4)], [(25, 2.5), (50, 2.0), (75, 1.5), (100, 1.2)])
        self.assertEqual((v.kind, v.step), ("overfitting", 50))
        self.assertIn("rose after step 50", v.english)
        self.assertIn("התאמת יתר", v.hebrew)

    def test_a_rising_val_loss_with_a_rising_train_loss_is_not_called_overfitting(self) -> None:
        v = verdict_from_points([(25, 2.0), (50, 1.0), (100, 1.4)], [(25, 2.0), (50, 2.1), (100, 2.4)])
        self.assertNotEqual(v.kind, "overfitting")

    def test_a_small_rise_is_not_overfitting(self) -> None:
        v = verdict_from_points([(25, 2.0), (50, 1.0), (100, 1.005)], [(25, 2.5), (50, 2.0), (100, 1.2)])
        self.assertEqual(v.kind, "plateau")

    def test_fewer_than_two_val_points_say_so_and_name_the_key(self) -> None:
        for points in ([], [(10, 1.0)]):
            v = verdict_from_points(points, [(1, 2.0)])
            self.assertEqual(v.kind, "too_few_points")
            self.assertIn("train.eval_every_steps", v.english)

    def test_the_verdict_reads_a_metrics_history(self) -> None:
        history = [record(5, train=3.0), record(10, train=2.5, val=2.0), record(10, val=2.0), record(20, train=2.0, val=1.5)]
        self.assertEqual(convergence.verdict_of(history).kind, "improving")


class RuleTests(unittest.TestCase):
    def run_view(self, history: list[StepMetrics], base: float, trained: float, verdict: str = "improved") -> TrainedRunView:
        cmp = ComparisonResult(base - trained, 0.0, 0.1, 0.9, True, 7, True, verdict, "x")  # type: ignore[arg-type]
        return TrainedRunView(history, ModelNumbers(base, 0.3, 10, 100), ModelNumbers(trained, 0.3, 10, 100, empty_output=2, hit_length_cap=3), cmp)

    def test_each_verdict_gets_its_advice_in_both_languages(self) -> None:
        improving = [record(10, 3.0, 2.0), record(20, 2.5, 1.5), record(30, 2.0, 1.0)]
        flat = [record(10, 3.0, 1.0), record(20, 2.5, 0.999), record(30, 2.0, 0.998)]
        over = [record(10, 3.0, 2.0), record(20, 2.5, 1.0), record(30, 2.0, 1.5)]
        for history, word in ((improving, "MORE steps"), (flat, "HIGHER learning rate"), (over, "FEWER steps")):
            lines = convergence.next_steps(convergence.verdict_of(history), self.run_view(history, 0.8, 0.6))
            self.assertTrue(any(word in en for en, _ in lines), (word, lines))
            self.assertTrue(all(he for _, he in lines))

    def test_good_loss_but_bad_wer_points_at_empty_outputs_and_labels(self) -> None:
        history = [record(10, 3.0, 2.0), record(20, 2.5, 1.0)]
        lines = convergence.next_steps(convergence.verdict_of(history), self.run_view(history, 0.8, 0.85, verdict="no_clear_difference"))
        text = " ".join(en for en, _ in lines)
        self.assertIn("Good loss but bad WER", text)
        self.assertIn("2 empty", text)
        self.assertIn("labels", text)
        clean = convergence.next_steps(convergence.verdict_of(history), self.run_view(history, 0.8, 0.6))
        self.assertFalse(any("Good loss but bad WER" in en for en, _ in clean))


class PageTests(unittest.TestCase):
    def view(self, final: FinalTestView | None = None, clips: int = 40) -> ReportData:
        history = [record(5, 3.0), record(10, 2.5, 2.0), record(15, 2.2), record(20, 2.0, 1.5)]
        rows = [ClipRow(f"c{i}", "g", "שלום עולם", "בסיס", "מאומן", 1.0, i / clips) for i in range(clips)]
        cmp = ComparisonResult(0.2, 0.1, 0.3, 0.9, True, 7, True, "improved", "better")  # type: ignore[arg-type]
        run = TrainedRunView(history, ModelNumbers(0.8, 0.4, 101, 4697), ModelNumbers(0.6, 0.3, 101, 4697), cmp, rows, 20, final)
        return ReportData(RunInfoView("exp_0003", "2026-10-26T10:00:00"), ReportConfig(target_relative_wer_reduction=0.10), trained_run=run)

    def test_the_page_has_the_chart_the_captions_in_both_languages_the_band_and_the_tables(self) -> None:
        out = ReportBuilder().build(self.view())
        html = out.html
        self.assertIn("trained", out.sections)
        self.assertIn("clips", out.sections)
        self.assertNotIn("final_test", out.sections)
        self.assertIn("best checkpoint: step 20", html)
        self.assertIn("Solid blue line", html)
        self.assertIn("הקו הכחול המלא", html)
        self.assertIn(f"owner&#x27;s target band {ABSOLUTE_WER_TARGET_BAND[0]:g}-{ABSOLUTE_WER_TARGET_BAND[1]:g}", html)
        self.assertIn("What to change next", html)
        self.assertEqual(html.count("<h3>15 worst") + html.count("<h3>15 best"), 2)
        self.assertEqual(len(re.findall(r">c\d+<", html)), 30)  # 15 worst + 15 best clip ids
        self.assertIn("Relative reduction of the WER", html)
        self.assertNotIn("<script", html)
        self.assertNotIn("http://", html.replace("http://www.w3.org/2000/svg", ""))
        self.assertNotIn("FINAL - Test", html)  # the Test is not shown when test_result.json does not exist

    def test_the_final_test_section_appears_only_with_its_view_and_under_the_named_heading(self) -> None:
        final = FinalTestView("test", ModelNumbers(0.9, 0.5, 97, 4559), ModelNumbers(0.7, 0.4, 97, 4559), 0.222, 0.2, 0.1, 0.3, 0.9, 8, "improved", "confirmed", "better")
        html = ReportBuilder().build(self.view(final)).html
        self.assertIn("FINAL - Test, scored once", html)
        self.assertIn("Test: CONFIRMED", html)
        self.assertIn("0.9000", html)


class TinyExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.work = Path(cls._tmp.name)
        cls.base, cls.experiment = tiny_flow.make_experiment(cls.work, cache=True)
        code, out, err = tiny_flow.run_cli("report", "--experiment", cls.experiment, *cls.base)
        assert code == 0, err
        cls.html = (cls.work / "e" / cls.experiment / "reports" / "report.html").read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_report_of_a_real_tiny_train_and_evaluate_experiment_has_the_convergence_view(self) -> None:
        self.assertIn("Training: did it converge", self.html)
        self.assertIn("best checkpoint: step 8", self.html)
        self.assertIn("Baseline against trained model", self.html)
        self.assertIn("Worst and best Val clips", self.html)
        self.assertNotIn("<script", self.html)

    def test_it_shows_no_test_numbers_before_final_test(self) -> None:
        self.assertNotIn("FINAL - Test", self.html)
        self.assertFalse((self.work / "e" / self.experiment / "reports" / "test_result.json").exists())

    def test_the_train_loss_is_stored_in_the_metrics_file_the_report_reads(self) -> None:
        text = (self.work / "e" / self.experiment / "metrics" / "trial_001" / "metrics.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(text[0].split(",")[2:4], ["train_loss", "val_loss"])
        self.assertGreaterEqual(len(text), 8)


if __name__ == "__main__":
    unittest.main()
