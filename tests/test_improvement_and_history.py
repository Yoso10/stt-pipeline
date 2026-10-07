"""Tests of the improvement line and the history chart (Wave 3 task 7; DECISIONS 76 and 78): the relative reduction of the Val WER against the baseline, the optional target, the `run` summary line,
the verdict section and the history chart of the report, and the loader of the history from the registry cards.

How to run (project root, virtual environment active):
    python -m unittest tests.test_improvement_and_history -v
The data is built by hand (the helpers of tests/test_reportbuilder.py); nothing is downloaded and nothing is recomputed: the line and the chart only show stored numbers.
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_reportbuilder import build, full_data, top_of  # noqa: E402

from stt.orchestrator import improvement as orchestrator_improvement  # noqa: E402
from stt.orchestrator.cli import _say_verdict  # noqa: E402
from stt.orchestrator.reportdata import _history, _Reader  # noqa: E402
from stt.orchestrator.types import RunOutcome, StageResult  # noqa: E402
from stt.reportbuilder import HistoryPoint, ReportBuilder, ReportConfig  # noqa: E402
from stt.reportbuilder import improvement as report_improvement  # noqa: E402

PREFIX = "Relative reduction of the Val WER against the baseline: "
CASES: list[tuple[float | None, float | None, float | None, bool | None]] = [
    (0.60, 0.54, 0.10, True), (0.60, 0.54, 0.10, False), (0.60, 0.58, 0.10, True), (0.60, 0.54, None, None), (0.0, 0.1, 0.1, True), (None, 0.5, 0.1, None), (0.5, None, 0.1, None),
    (0.6, 0.7, 0.1, False), (float("nan"), 0.5, 0.1, None), (0.60, 0.60, 0.1, False), (0.5, 0.0, 1.0, True), (0.6, 0.54, 0.1, None),
]


class LineTests(unittest.TestCase):
    def test_the_line_in_every_case_of_the_owner_wording(self) -> None:
        line = orchestrator_improvement.improvement_line
        self.assertEqual(line(0.60, 0.54, 0.10, True), PREFIX + "10.0% (target: 10%): reached")  # the owner's example: 0.60 -> 0.54
        self.assertEqual(line(0.60, 0.58, 0.10, True), PREFIX + "3.3% (target: 10%): not reached")
        self.assertEqual(line(0.60, 0.54, None, True), PREFIX + "10.0% (no target set).")
        self.assertEqual(line(0.0, 0.1, 0.1, True), PREFIX + "not defined (the baseline Val WER is 0).")
        self.assertEqual(line(0.6, 0.7, 0.1, True), PREFIX + "-16.7% (target: 10%): not reached")  # worse than the baseline: a negative reduction
        self.assertEqual(line(None, 0.5, 0.1, None), PREFIX + "not available (the Val WER of the baseline or of the best model is not stored).")
        self.assertIn("not available", line(float("nan"), 0.5, 0.1, None))

    def test_a_reached_target_that_is_not_a_proven_improvement_says_point_estimate_only(self) -> None:
        line = orchestrator_improvement.improvement_line
        text = line(0.60, 0.54, 0.10, False)
        self.assertTrue(text.startswith(PREFIX + "10.0% (target: 10%): reached."))
        self.assertIn("Point estimate only", text)
        self.assertNotIn("Point estimate only", line(0.60, 0.54, 0.10, True))  # proven: no caveat
        self.assertNotIn("Point estimate only", line(0.60, 0.54, 0.10, None))  # unknown verdict: no claim either way
        self.assertNotIn("Point estimate only", line(0.60, 0.58, 0.10, False))  # not reached: nothing to qualify

    def test_the_report_and_the_terminal_give_the_same_text_for_the_same_numbers(self) -> None:
        for case in CASES:
            with self.subTest(case=case):
                self.assertEqual(orchestrator_improvement.improvement_line(*case), report_improvement.improvement_line(*case))

    def test_the_reduction_is_computed_from_the_stored_numbers_only(self) -> None:
        self.assertAlmostEqual(orchestrator_improvement.relative_reduction(0.5, 0.4) or 0.0, 0.2)
        self.assertIsNone(orchestrator_improvement.relative_reduction(0.0, 0.1))
        self.assertIsNone(orchestrator_improvement.relative_reduction(None, 0.1))
        self.assertEqual(ReportConfig().target_relative_wer_reduction, None)  # the target is optional: nothing changes unless it is set
        with self.assertRaises(ValueError):
            ReportConfig(target_relative_wer_reduction=0.0)
        with self.assertRaises(ValueError):
            ReportConfig(target_relative_wer_reduction=1.5)


class RunSummaryTests(unittest.TestCase):
    def outcome(self, summary: dict[str, Any], verdict: str = "winner") -> RunOutcome:
        return RunOutcome(0, verdict, "exp_0001", {"select_and_register": StageResult("ok", summary, [])}, "completed")

    def said(self, outcome: RunOutcome, target: float | None) -> list[str]:
        with self.assertLogs("stt.orchestrator", level="INFO") as logs:
            _say_verdict(outcome, target)
        return [r.getMessage() for r in logs.records]

    def summary(self, **over: Any) -> dict[str, Any]:
        base = {"reason": "Trial 1 wins.", "test_status": "confirmed", "entry_id": "model_0001", "registry_dir": "registry", "baseline_val_wer": 0.60, "winner_trial": 1, "winner_val_wer": 0.54,
                "best_trial": 1, "best_val_wer": 0.54, "best_val_verdict": "improved"}
        base.update(over)
        return base

    def test_the_verdict_is_followed_by_the_improvement_line_with_the_target(self) -> None:
        lines = self.said(self.outcome(self.summary()), 0.10)
        self.assertTrue(lines[0].startswith("VERDICT: winner."))
        self.assertEqual(lines[-1], PREFIX + "10.0% (target: 10%): reached")

    def test_no_target_set_and_not_reached_and_point_estimate_only(self) -> None:
        self.assertEqual(self.said(self.outcome(self.summary()), None)[-1], PREFIX + "10.0% (no target set).")
        self.assertEqual(self.said(self.outcome(self.summary(best_val_wer=0.58)), 0.10)[-1], PREFIX + "3.3% (target: 10%): not reached")
        unproven = self.said(self.outcome(self.summary(test_status="not_confirmed")), 0.10)[-1]
        self.assertIn("reached.", unproven)
        self.assertIn("Point estimate only", unproven)

    def test_a_baseline_that_stays_uses_the_best_finalist_and_a_zero_baseline_is_not_defined(self) -> None:
        kept = self.summary(winner_trial="none", winner_val_wer="none", best_val_wer=0.57, best_val_verdict="no_clear_difference", test_status="not_evaluated")
        self.assertEqual(self.said(self.outcome(kept, "baseline_retained"), 0.10)[-1], PREFIX + "5.0% (target: 10%): not reached")
        self.assertIn("not defined", self.said(self.outcome(self.summary(baseline_val_wer=0.0)), 0.10)[-1])

    def test_an_older_stored_result_without_the_numbers_says_not_available(self) -> None:
        old = {"reason": "x", "test_status": "confirmed", "entry_id": "model_0001", "registry_dir": "registry"}
        self.assertIn("not available", self.said(self.outcome(old), 0.10)[-1])


class ReportLineTests(unittest.TestCase):
    def test_the_verdict_section_shows_the_line_from_the_stored_numbers(self) -> None:
        top = top_of(build(full_data()))  # baseline Val WER 0.3, the winner 0.1 (hand-made data)
        self.assertIn("Relative reduction of the Val WER against the baseline: 66.7% (no target set).", top)

    def test_the_target_reached_and_not_reached(self) -> None:
        self.assertIn("(target: 10%): reached", top_of(build(full_data(settings=ReportConfig(target_relative_wer_reduction=0.10)))))
        self.assertIn("(target: 90%): not reached", top_of(build(full_data(settings=ReportConfig(target_relative_wer_reduction=0.90)))))

    def test_a_reached_target_without_a_proven_improvement_says_point_estimate_only(self) -> None:
        data = full_data(settings=ReportConfig(target_relative_wer_reduction=0.10))
        weak = replace(data, selection=replace(data.selection, test_status="not_confirmed"))  # type: ignore[type-var]
        self.assertIn("Point estimate only", top_of(build(weak)))
        self.assertNotIn("Point estimate only", top_of(build(data)))

    def test_a_baseline_that_stays_uses_the_best_finalist(self) -> None:
        text = top_of(build(full_data(winner=False, settings=ReportConfig(target_relative_wer_reduction=0.10))))
        self.assertIn("Relative reduction of the Val WER against the baseline: 3.3% (target: 10%): not reached", text)  # 0.3 -> 0.29


class HistoryChartTests(unittest.TestCase):
    POINTS = (
        HistoryPoint("model_0001", "exp_0001", "model", 0.60, 0.55),
        HistoryPoint("model_0002", "exp_0002", "baseline_retained", 0.60, 0.61),
        HistoryPoint("model_0003", "exp_0003", "model", 0.60, 0.50),
    )

    def section(self, html: str) -> str:
        return html.split('<h2 id="history">')[1].split("<h2")[0]

    def test_the_chart_has_a_baseline_line_a_best_line_and_the_dashed_target_line_with_a_table(self) -> None:
        html = build(full_data(history=self.POINTS, settings=ReportConfig(target_relative_wer_reduction=0.10)))
        part = self.section(html)
        self.assertEqual(part.count("<polyline"), 3)
        self.assertEqual(part.count("stroke-dasharray"), 1)  # only the target is dashed
        self.assertIn("target: 10% below the baseline", part)
        self.assertIn("baseline Val WER", part)
        self.assertIn("best model Val WER", part)
        self.assertIn("<td>baseline kept</td>", part)  # the retained baseline is marked
        self.assertIn("8.3%", part)  # (0.60 - 0.55) / 0.60 of the first experiment
        self.assertIn("16.7%", part)  # the third
        self.assertIn("-1.7%", part)  # the retained baseline: its best finalist is worse than the baseline
        self.assertIn("reached", part)
        self.assertIn("not reached", part)
        self.assertIn("nothing is recomputed", part)

    def test_without_a_target_there_is_no_target_line_and_no_target_columns(self) -> None:
        part = self.section(build(full_data(history=self.POINTS)))
        self.assertEqual(part.count("<polyline"), 2)
        self.assertEqual(part.count("stroke-dasharray"), 0)
        self.assertNotIn("target Val WER", part)
        self.assertIn("No target is set", part)

    def test_missing_history_is_named_not_an_empty_chart(self) -> None:
        part = self.section(build(full_data()))
        self.assertIn("Not available in this run: the history of the registered experiments", part)
        self.assertNotIn("<svg", part)
        named = self.section(build(full_data(history_problem="no registry was given: pass --set registry.dir=...")))
        self.assertIn("no registry was given", named)
        self.assertNotIn("<svg", named)

    def test_one_point_works_and_a_missing_number_is_a_dash(self) -> None:
        part = self.section(build(full_data(history=(HistoryPoint("model_0001", "exp_0001", "model", 0.6, None),))))
        self.assertIn("<svg", part)
        self.assertIn("<td class=\"num\">-</td>", part)

    def test_the_history_is_the_first_section_after_the_summary_and_is_listed_as_produced(self) -> None:
        output = ReportBuilder().build(full_data(history=self.POINTS))
        self.assertEqual(output.sections[:2], ["summary", "history"])
        self.assertNotIn("history", ReportBuilder().build(full_data()).sections)  # no data: not produced, named in the gaps line


class LoaderTests(unittest.TestCase):
    def registry(self, cards: dict[str, Any], entries: list[SimpleNamespace] | None = None) -> Any:
        listing = entries if entries is not None else [SimpleNamespace(entry_id=k, experiment_id=f"exp_{i + 1:04d}") for i, k in enumerate(cards)]

        def card(entry_id: str) -> Any:
            value = cards[entry_id]
            if isinstance(value, Exception):
                raise value
            return value

        return SimpleNamespace(entries=lambda: listing, card=card)

    def test_a_registered_model_a_retained_baseline_and_the_problems_are_read_from_the_cards(self) -> None:
        cards = {
            "model_0001": SimpleNamespace(kind="model", val=SimpleNamespace(wer=0.55, baseline_wer=0.60), finalists=[]),
            "model_0002": SimpleNamespace(kind="baseline_retained", val=SimpleNamespace(wer=0.60, baseline_wer=None), finalists=[SimpleNamespace(val_wer=0.62), SimpleNamespace(val_wer=0.61)]),
            "model_0003": RuntimeError("damaged"),
            "model_0004": SimpleNamespace(kind="model", val=None, finalists=[]),
        }
        reader = _Reader()
        points, problem = _history(self.registry(cards), reader)
        self.assertEqual([(p.entry_id, p.kind, p.baseline_wer, p.best_wer) for p in points], [("model_0001", "model", 0.60, 0.55), ("model_0002", "baseline_retained", 0.60, 0.61)])
        self.assertIn("model_0003 (card unreadable)", problem or "")
        self.assertIn("model_0004 (no Val result in its card)", problem or "")
        self.assertTrue(any("model_0003" in p for p in reader.problems))
        self.assertEqual(reader.found["history"], "2 registered experiment(s)")

    def test_no_registry_an_empty_registry_and_a_registry_that_cannot_list_are_named(self) -> None:
        reader = _Reader()
        points, problem = _history(None, reader)
        self.assertEqual((points, "no registry was given" in (problem or "")), ((), True))
        points, problem = _history(self.registry({}), _Reader())
        self.assertEqual((points, problem), ((), "the registry has no entry yet"))
        points, problem = _history(SimpleNamespace(card=lambda e: None), _Reader())
        self.assertIn("cannot list its entries", problem or "")

        def broken() -> list[Any]:
            raise OSError("disk")

        points, problem = _history(SimpleNamespace(entries=broken, card=lambda e: None), _Reader())
        self.assertIn("could not be listed", problem or "")


if __name__ == "__main__":
    unittest.main()
