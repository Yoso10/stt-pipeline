"""Unit tests of the ReportBuilder component (stt.reportbuilder): the acceptance tests of the C14 brief, offline, on synthetic results.

How to run (project root, virtual environment active):
    python -m unittest tests.test_reportbuilder -v
The data is built by hand (no model, no audio, no network): per-file scores, a stored selection, a search, training histories. Nothing is downloaded and the builder never recomputes a WER.
"""

from __future__ import annotations

import ast
import logging
import re
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stt.config import SectionRegistry, load_config  # noqa: E402
from stt.evaluator import ComparisonResult, DecodeSettings, EvalResult, FileScore, ReviewEntry, ReviewList  # noqa: E402
from stt.modelloader import ModelInfo  # noqa: E402
from stt.reportbuilder import (  # noqa: E402
    BenchmarkView,
    CurvePointView,
    HintsConfig,
    HtmlProgressSink,
    LearningCurveView,
    ProgressConfig,
    ReportBuilder,
    ReportConfig,
    ReportConfigError,
    ReportData,
    RuleRegistry,
    RunInfoView,
    SearchView,
    SelectionView,
    TrialView,
    correlation_row,
    default_rule_registry,
    ranks,
    render_progress,
    spearman,
    thin,
)
from stt.reportbuilder import svg  # noqa: E402
from stt.reportbuilder.hints import EdgeOfRange, FailedOrPruned, GainConcentrated, LengthCapOrEmpty, NoLearning, Overfitting, ShortClipWeakness, UnstableTraining  # noqa: E402
from stt.selector import ComparisonRow  # noqa: E402
from stt.trainercore import StepMetrics  # noqa: E402

SETTINGS = ReportConfig()
HEBREW = "שלום עולם טוב"


def file_score(i: int, group: int, errors: int, duration: float = 10.0, words: int = 10, hit: bool = False, empty: bool = False, **attrs: float | int | bool) -> FileScore:
    attributes: dict[str, float | int | bool | str] = {"duration": duration, "ref_words": words, "label_tokens": words * 2, **attrs}
    return FileScore(f"f{i:03d}", f"g{group:02d}", HEBREW, HEBREW, HEBREW, words, errors, errors / words, None, empty, hit, attributes)


def result(name: str, files: list[FileScore], split: str = "val", review: tuple[ReviewEntry, ...] = ()) -> EvalResult:
    words = sum(f.ref_words for f in files)
    wer = sum(f.word_errors for f in files) / words
    return EvalResult(
        split, 1, ModelInfo(name, Path(name), 0, "float32", 80, "he", "transcribe", None), wer, wer, 0.2, len(files), words, {0.1: 0.25, 0.2: 0.5}, tuple(files), DecodeSettings(1, 100, 4), 1.0,
        {"empty_output": 0, "hit_length_cap": 0, "excluded_empty_reference": 0}, (), ReviewList(review),
    )


def files_with(errors: list[int], groups: int = 10, **kw: Any) -> list[FileScore]:
    return [file_score(i, i % groups, e, duration=2.0 + (i % 14) * 2.0, **kw) for i, e in enumerate(errors)]


def row(rank: int, trial: int, wer: float, delta: float | None, lo: float | None, hi: float | None, verdict: str = "improved", eligible: bool = True) -> ComparisonRow:
    return ComparisonRow(rank, trial, {"learning_rate": 1e-5 * trial, "grad_accum_steps": 4}, wer, 0.2, delta, lo, hi, verdict, 0.9 + 0.01 * trial, 100, eligible, "WINNER" if rank == 1 and eligible else "not eligible")


def history(vals: list[float | None], trains: list[float]) -> list[StepMetrics]:
    return [StepMetrics(i * 10, i / 10, trains[i], vals[i], 1e-5, 1.0, 0.5, 5.0 * i, 10.0) for i in range(len(vals))]


def info() -> RunInfoView:
    return RunInfoView("exp_0007", "2026-10-06T12:00:00", "exp_0007/config_full.yaml", "whisper-tiny", "model_0003", "d" * 64, "s" * 64, None, "t" * 64, 2, {"torch": "2.14.1"}, "0.0.0", 42, 335.0)


def full_data(winner: bool = True, test: str = "confirmed", settings: ReportConfig = SETTINGS, **over: Any) -> ReportData:
    base = result("baseline", files_with([3] * 40))
    win = result("exp_0007_final_trial_001_step_00100", files_with([1] * 40), review=(ReviewEntry("f001", "g01", Path("audio") / "f001.wav", HEBREW, "שלום <b>עולם</b>", 0.3, "שלום [עולם->x]", "worst"),))
    second = result("exp_0007_final_trial_002_step_00100", files_with([2] * 40))
    table = (row(1, 1, 0.1, 0.2, 0.15, 0.25), row(2, 2, 0.2, 0.1, 0.05, 0.14)) if winner else (row(1, 1, 0.29, 0.01, -0.02, 0.04, "no_clear_difference", False),)
    comparison = ComparisonResult(0.15, 0.1, 0.2, 0.9, True, 10, True, "improved", "Improved at the 90% level.")
    selection = SelectionView("winner" if winner else "baseline_retained", 1 if winner else None, test if winner else "not_evaluated", "Trial 1 wins on Val." if winner else "The baseline stays.", table, (), None, comparison if test != "not_evaluated" else None, 0.9)
    tests = {"baseline": result("baseline", files_with([3] * 10), "test"), "trial_1": result("winner", files_with([1] * 10), "test")} if winner and test != "not_evaluated" else {}
    search = SearchView("grid", 42, "n_trials", {"learning_rate": {"kind": "log_float", "low": 1e-5, "high": 1e-4}, "grad_accum_steps": {"kind": "categorical", "choices": [2, 4]}},
                        tuple(TrialView(i, {"learning_rate": 1e-5 * (i + 1), "grad_accum_steps": 2 + 2 * (i % 2)}, "complete", 1.0 + 0.1 * i, 100, 30.0) for i in range(5)), (0, 1, 2), {"learning_rate": 1e-5, "grad_accum_steps": 2}, 1.0, ())
    curve = LearningCurveView(tuple(CurvePointView(p, n, m, w, 0.4, 1.0, 100) for p, n, m, w in (("smoke", 20, 1.0, 0.9), ("tiny", 60, 30.0, 0.5), ("half", 200, 90.0, 0.4))), "improving", "STILL IMPROVING. text", "train.hyperparams of the config", 100)
    bench = BenchmarkView("whisper-tiny", 2, 4.77, 0.42, 1.93, (("baseline", 1680.0), ("search", 180.0)), 1860.0, ())
    values: dict[str, Any] = dict(
        run_info=info(), settings=settings, selection=selection, baseline_val=base, finalist_val={1: win, 2: second}, test_results=tests,
        final_histories={1: history([2.0, 1.5, 1.2], [2.2, 1.6, 1.3]), 2: history([2.0, 1.8, 1.7], [2.2, 1.9, 1.8])}, search_histories={0: history([2.0, 1.4], [2.1, 1.5])},
        search=search, learning_curve=curve, benchmark=bench,
    )
    values.update(over)
    return ReportData(**values)


def build(data: ReportData) -> str:
    return ReportBuilder().build(data).html


class SummaryTests(unittest.TestCase):
    def test_1_the_summary_shows_the_winner_the_numbers_the_interval_and_the_test_confirmation(self) -> None:
        html = build(full_data())
        top = html.split("<h2")[0]
        for part in ("WINNER: trial 1", "0.3000", "0.1000", "+0.2000", "+0.1500", "+0.2500", "90%", "CONFIRMED by the Test", "Test WER: baseline 0.3000, winner 0.1000", "Trial 1 wins on Val."):
            self.assertIn(part, top)

    def test_1b_the_summary_of_a_retained_baseline_and_of_a_winner_the_test_does_not_confirm(self) -> None:
        top = build(full_data(winner=False)).split("<h2")[0]
        for part in ("THE BASELINE STAYS", "0.3000", "best finalist, trial 1", "Test set was not evaluated", "The baseline stays."):
            self.assertIn(part, top)
        flagged = build(full_data(test="not_confirmed")).split("<h2")[0]
        self.assertIn("NOT confirmed by the Test (flagged, not dropped)", flagged)

    def test_the_test_appears_only_as_the_one_time_confirmation_never_per_file(self) -> None:
        data = full_data()
        marked = {k: replace(r, files=tuple(replace(f, sample_id="TEST_" + f.sample_id) for f in r.files)) for k, r in data.test_results.items()}  # ids that exist only in the Test results
        html = build(replace(data, test_results=marked))
        self.assertNotIn("TEST_", html)
        self.assertNotIn("Listening list of the test", html)
        self.assertEqual(html.count("Test WER:"), 1)

    def test_no_selection_says_so_and_a_page_is_still_made(self) -> None:
        html = build(full_data(selection=None))
        self.assertIn("No verdict in this run", html)
        self.assertIn("reports/selection.json", html)

    def test_no_finalists_at_all(self) -> None:
        data = full_data()
        empty = replace(data, selection=SelectionView("baseline_retained", None, "not_evaluated", "No finalist.", ()), finalist_val={})
        html = build(empty)
        self.assertIn("THE BASELINE STAYS", html)
        self.assertIn("Not available in this run: the comparison table", html)


class PageTests(unittest.TestCase):
    def test_2_the_page_refers_to_no_external_address_script_stylesheet_font_or_image(self) -> None:
        html = build(full_data())
        stripped = html.replace("http://www.w3.org/2000/svg", "")  # the SVG namespace is a name, not an address that is fetched
        self.assertNotRegex(stripped, r"https?://")
        for forbidden in ("<script", "<link", "<img", "src=", "@import", "url(", "@font-face", "<iframe"):
            self.assertNotIn(forbidden, html)
        self.assertIn('<meta charset="utf-8">', html)

    def test_5_hebrew_cells_are_right_to_left_and_markup_in_a_transcript_is_escaped(self) -> None:
        html = build(full_data())
        self.assertIn('dir="rtl" lang="he"', html)
        self.assertIn("&lt;b&gt;עולם&lt;/b&gt;", html)
        self.assertNotIn("<b>עולם</b>", html)
        title = build(replace(full_data(), settings=ReportConfig(title="<i>x</i> & y")))
        self.assertIn("&lt;i&gt;x&lt;/i&gt; &amp; y", title)

    def test_6_a_missing_input_gives_a_not_available_note_naming_the_input_and_no_error(self) -> None:
        out = ReportBuilder().build(ReportData(info(), SETTINGS))
        self.assertEqual(out.sections, ["summary", "run_info"])
        for part in ("Not available in this run: the search", "reports/search_trials.json", "reports/learning_curve.json", "reports/selection.json", "Planned time not available"):
            self.assertIn(part, out.html)
        self.assertNotIn("<svg", out.html)  # never an empty chart
        self.assertIn("Sections without data in this run", out.html)

    def test_7_the_same_inputs_give_byte_identical_output(self) -> None:
        a, b = build(full_data()), build(full_data())
        self.assertEqual(a, b)
        self.assertEqual(a.encode("utf-8"), b.encode("utf-8"))
        self.assertIn("2026-10-06T12:00:00", a)  # the generation time is the one that was passed in

    def test_10_the_print_style_is_present(self) -> None:
        html = build(full_data())
        self.assertIn("@media print", html)
        self.assertIn("page-break-inside:avoid", html)

    def test_a_full_report_has_every_section_and_the_charts(self) -> None:
        out = ReportBuilder().build(full_data())
        self.assertEqual(out.sections, ["summary", "finalists", "wer_bars", "training_curves", "distribution", "groups", "search", "learning_curve", "correlations", "hints", "listening", "run_info"])
        self.assertGreaterEqual(out.html.count("<svg"), 9)
        for part in ("Finalists compared with the baseline", "Learning curve", "A correlation is not a cause", "How often this Test set has been evaluated", "2 time(s) by registered entries", "planned time"[:7]):
            self.assertIn(part, out.html)

    def test_audio_paths_are_shown_as_text_and_as_encoded_links(self) -> None:
        review = (ReviewEntry("f001", "g01", Path("C:/Daf Yomi/שיעור 1/a b.mp3"), HEBREW, HEBREW, 0.3, "x", "worst"),)
        data = full_data()
        win = result("w", files_with([1] * 40), review=review)
        html = build(replace(data, finalist_val={1: win}))
        self.assertIn(">C:/Daf Yomi/שיעור 1/a b.mp3</a>", html)  # the plain text path
        self.assertIn('href="file:///C:/Daf%20Yomi/%D7%A9', html)  # the encoded link

    def test_a_history_without_a_val_loss_draws_the_training_loss_and_says_so(self) -> None:
        html = build(replace(full_data(), final_histories={1: history([None, None, None], [2.0, 1.5, 1.2])}))
        self.assertIn("have no Val loss", html)


class CorrelationTests(unittest.TestCase):
    def test_3_the_rank_correlation_matches_hand_computed_values_including_ties(self) -> None:
        self.assertAlmostEqual(spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) or 0.0, 0.8)  # 1 - 6 * 4 / (5 * 24)
        self.assertAlmostEqual(spearman([1, 2, 2, 3], [1, 2, 3, 4]) or 0.0, 4.5 / 22.5**0.5)  # average ranks 1, 2.5, 2.5, 4
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]) or 0.0, -1.0)
        self.assertEqual(ranks([10, 20, 20, 5]), [2.0, 3.5, 3.5, 1.0])

    def test_3b_constant_data_gives_a_note_and_no_number(self) -> None:
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))
        row_ = correlation_row("duration", [5.0] * 40, [float(i) for i in range(40)], 30)
        self.assertEqual((row_.rho, row_.n), (None, 40))
        self.assertIn("constant", row_.note)
        few = correlation_row("duration", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 30)
        self.assertEqual((few.rho, few.n, few.note), (None, 3, "too few points (3 < 30)"))
        nan = correlation_row("x", [1.0, float("nan"), 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], 2)
        self.assertEqual(nan.n, 3)  # a missing value drops the pair

    def test_the_report_table_shows_n_the_number_and_the_not_a_cause_note(self) -> None:
        html = build(full_data())
        table = html.split('id="correlations"')[1].split("</table>")[0]
        for part in ("duration", "40", "label_tokens", "extra_meta_quality_score", "not present in this run", "flag_had_digits", "constant"):
            self.assertIn(part, table)
        self.assertIn("A correlation is not a cause", html)

    def test_few_points_in_the_report_row_says_too_few_points(self) -> None:
        files = [file_score(i, i % 10, i % 4, duration=2.0 + i, extra_meta_quality_score=0.1 * i) if i < 12 else file_score(i, i % 10, i % 4, duration=2.0 + i) for i in range(40)]
        html = build(replace(full_data(), finalist_val={1: result("w", files)}))
        self.assertIn("too few points (12 &lt; 30)", html)


class ThinningTests(unittest.TestCase):
    def test_9_thinning_is_deterministic_keeps_the_cap_and_the_ends(self) -> None:
        items = list(range(1000))
        a, thinned = thin(items, 100)
        b, _ = thin(items, 100)
        self.assertEqual((a, thinned), (b, True))
        self.assertLessEqual(len(a), 100)
        self.assertEqual((a[0], a[-1]), (0, 999))
        self.assertEqual(thin(items[:50], 100), (items[:50], False))

    def test_the_page_says_when_a_chart_was_thinned(self) -> None:
        files = [file_score(i, i % 10, i % 4, duration=2.0 + i) for i in range(300)]
        out = ReportBuilder().build(replace(full_data(settings=ReportConfig(max_points_per_chart=50)), finalist_val={1: result("w", files)}, baseline_val=result("b", files)))
        self.assertTrue(any("thinned to 50 points" in w for w in out.warnings), out.warnings)


class HintTests(unittest.TestCase):
    def data(self, **over: Any) -> ReportData:
        return full_data(**over)

    def test_4_overfitting_fires_on_a_rising_val_loss_with_a_falling_training_loss_and_not_on_a_healthy_one(self) -> None:
        bad = self.data(final_histories={1: history([1.0, 0.9, 1.0, 1.1, 1.2], [1.5, 1.3, 1.1, 1.0, 0.9])})
        hint = Overfitting().evaluate(bad)
        self.assertIsNotNone(hint)
        self.assertEqual(hint.severity, "warning")  # type: ignore[union-attr]
        self.assertIn("may be starting to overfit", hint.message)  # type: ignore[union-attr]
        self.assertTrue(hint.caveat)  # type: ignore[union-attr]
        self.assertIsNone(Overfitting().evaluate(self.data()))

    def test_4_no_learning(self) -> None:
        self.assertIsNotNone(NoLearning().evaluate(self.data(final_histories={1: history([1.0, 0.999, 0.998], [1.0, 1.0, 1.0])})))
        self.assertIsNone(NoLearning().evaluate(self.data()))

    def test_4_unstable_training(self) -> None:
        spiky = self.data(final_histories={1: history([None] * 8, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0])})
        hint = UnstableTraining().evaluate(spiky)
        self.assertIsNotNone(hint)
        self.assertEqual(hint.evidence["first_spike_step"], 60)  # type: ignore[union-attr]
        self.assertIsNone(UnstableTraining().evaluate(self.data()))

    def test_4_gain_concentrated(self) -> None:
        base = result("b", files_with([3] * 40))
        errors = [0 if i % 10 == 0 else 3 for i in range(40)]  # only group 0 improves: all of the gain is in 1 of 10 groups
        concentrated = self.data(finalist_val={1: result("w", files_with(errors))}, baseline_val=base)
        hint = GainConcentrated().evaluate(concentrated)
        self.assertIsNotNone(hint)
        self.assertEqual(hint.evidence["share_of_gain"], 1.0)  # type: ignore[union-attr]
        self.assertIsNone(GainConcentrated().evaluate(self.data()))  # a gain spread over every group

    def test_4_short_clip_weakness(self) -> None:
        errors = [8 if i % 14 < 4 else 1 for i in range(56)]  # the shortest clips (duration grows with i % 14) have many more errors
        weak = self.data(finalist_val={1: result("w", files_with(errors))})
        hint = ShortClipWeakness().evaluate(weak)
        self.assertIsNotNone(hint)
        self.assertGreater(hint.evidence["ratio"], 1.5)  # type: ignore[operator,union-attr]
        self.assertIsNone(ShortClipWeakness().evaluate(self.data()))

    def test_4_length_cap_or_empty(self) -> None:
        files = [file_score(i, i % 10, 1, hit=i < 4, empty=4 <= i < 6) for i in range(40)]
        hint = LengthCapOrEmpty().evaluate(self.data(finalist_val={1: result("w", files)}))
        self.assertIsNotNone(hint)
        self.assertEqual((hint.evidence["hit_length_cap"], hint.evidence["empty_output"]), (4, 2))  # type: ignore[union-attr]
        self.assertIsNone(LengthCapOrEmpty().evaluate(self.data()))

    def test_4_edge_of_range_and_failed_or_pruned(self) -> None:
        self.assertIsNotNone(EdgeOfRange().evaluate(self.data()))  # the best learning rate is the low end of its range
        middle = replace(self.data().search, best_params={"learning_rate": 3e-5, "grad_accum_steps": 2})  # type: ignore[type-var]
        self.assertIsNone(EdgeOfRange().evaluate(self.data(search=middle)))
        trials = tuple(TrialView(i, {"learning_rate": 1e-5}, "failed" if i < 3 else "complete", None if i < 3 else 1.0, 5, 1.0) for i in range(6))
        failing = self.data(search=replace(self.data().search, trials=trials))  # type: ignore[type-var]
        hint = FailedOrPruned().evaluate(failing)
        self.assertIsNotNone(hint)
        self.assertEqual(hint.evidence["failed"], 3)  # type: ignore[union-attr]
        self.assertIsNone(FailedOrPruned().evaluate(self.data()))

    def test_every_hint_is_worded_as_a_hint_with_evidence_and_a_caveat_in_the_page(self) -> None:
        html = build(self.data(final_histories={1: history([1.0, 0.9, 1.0, 1.1, 1.2], [1.5, 1.3, 1.1, 1.0, 0.9])}))
        block = html.split('id="hints"')[1].split("<h2")[0]
        self.assertIn("Evidence:", block)
        self.assertIn("Caveat:", block)
        self.assertIn("may be", block)

    def test_rules_missing_their_input_stay_silent_and_a_healthy_run_says_no_rule_fired(self) -> None:
        for rule in default_rule_registry().names():
            self.assertIsNone(default_rule_registry().create(rule).evaluate(ReportData(info(), SETTINGS)), rule)

    def test_an_unknown_rule_name_fails_listing_the_registered_names(self) -> None:
        settings = ReportConfig(hints=HintsConfig(enabled_rules=["overfitting", "nope"]))
        with self.assertRaises(ReportConfigError) as raised:
            ReportBuilder().build(full_data(settings=settings))
        self.assertIn("Registered rules: edge_of_range, failed_or_pruned", str(raised.exception))
        only = ReportBuilder().build(full_data(settings=ReportConfig(hints=HintsConfig(enabled_rules=["edge_of_range"]))))
        self.assertIn("edge_of_range", only.html)

    def test_a_rule_that_raises_becomes_a_warning_and_the_report_is_still_made(self) -> None:
        class Broken:
            name = "broken"

            def evaluate(self, data: ReportData) -> None:
                raise RuntimeError("boom")

        registry = RuleRegistry({"broken": Broken, "no_learning": NoLearning})  # type: ignore[dict-item]
        with self.assertLogs("stt.reportbuilder", level="ERROR"):
            out = ReportBuilder(registry).build(full_data())
        self.assertTrue(any("broken failed: RuntimeError: boom" in w for w in out.warnings))
        self.assertIn("could not be evaluated", out.html)
        self.assertIn("Run information", out.html)


class ProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "live" / "progress.html"

    def records(self, n: int) -> list[StepMetrics]:
        return [StepMetrics(i + 1, (i + 1) / 20, 2.0 - 0.05 * i, 1.9 - 0.04 * i if i % 2 else None, 1e-5, 1.0, 1.5, 1.5 * (i + 1), 60.0 - i) for i in range(n)]

    def test_8_the_sink_regenerates_the_page_as_records_arrive_and_the_page_refreshes_itself(self) -> None:
        sink = HtmlProgressSink(self.path, SETTINGS, "exp_0007")
        recs = self.records(3)
        sink.write(recs[0])
        first = self.path.read_text(encoding="utf-8")
        self.assertIn('http-equiv="refresh" content="10"', first)
        self.assertIn("Step <b>1</b>", first)
        sink.write(recs[1])
        sink.write(recs[2])
        again = self.path.read_text(encoding="utf-8")
        self.assertIn("Step <b>3</b>", again)
        self.assertIn("estimated time left", again)
        self.assertIn("<svg", again)
        self.assertFalse(self.path.with_name("progress.html.tmp").exists())  # written through a temporary file and renamed

    def test_8_the_refresh_interval_and_the_write_rate_come_from_the_config(self) -> None:
        settings = ReportConfig(progress=ProgressConfig(refresh_seconds=3, every_n_records=2))
        sink = HtmlProgressSink(self.path, settings, "t")
        recs = self.records(4)
        sink.write(recs[0])  # the first record always writes
        sink.write(recs[1])
        self.assertIn("Step <b>2</b>", self.path.read_text(encoding="utf-8"))
        sink.write(recs[2])  # not a multiple of 2: the page is not rewritten
        self.assertIn("Step <b>2</b>", self.path.read_text(encoding="utf-8"))
        sink.write(recs[3])
        self.assertIn("Step <b>4</b>", self.path.read_text(encoding="utf-8"))
        self.assertIn('content="3"', self.path.read_text(encoding="utf-8"))

    def test_8_a_write_failure_is_logged_and_never_raised(self) -> None:
        blocker = Path(self._tmp.name) / "blocker"
        blocker.write_text("a file, not a folder", encoding="utf-8")
        sink = HtmlProgressSink(blocker / "progress.html", SETTINGS, "t")
        with self.assertLogs("stt.reportbuilder", level="ERROR") as logs:
            for record in self.records(3):
                sink.write(record)  # must not raise
        self.assertEqual(len(logs.records), 1)  # logged once, not for every record
        self.assertIn("training goes on", logs.output[0])

    def test_the_page_before_the_first_record_and_the_pure_render(self) -> None:
        self.assertIn("No metrics record yet", render_progress([], SETTINGS, "t"))
        self.assertEqual(render_progress(self.records(5), SETTINGS, "t"), render_progress(self.records(5), SETTINGS, "t"))

    def test_the_sink_has_the_write_method_of_the_trainer_core_protocol(self) -> None:
        sink = HtmlProgressSink(self.path, SETTINGS, "t")
        self.assertTrue(callable(sink.write))


class ConfigAndStructureTests(unittest.TestCase):
    def test_the_defaults_are_those_of_the_brief_and_the_example_yaml_matches(self) -> None:
        c = ReportConfig()
        self.assertEqual((c.min_points_for_correlation, c.max_points_per_chart, c.hints.enabled_rules, c.progress.refresh_seconds, c.progress.every_n_records, c.title), (30, 2000, None, 10, 1, None))
        self.assertEqual(c.attributes[:3], ["duration", "label_tokens", "ref_words"])
        loaded = load_config(ROOT / "configs" / "report_default.yaml", env_file=None, target_environ={}, registry=SectionRegistry({"report": ReportConfig}))
        self.assertEqual(loaded.section("report", ReportConfig), c)

    def test_svg_functions_return_one_titled_svg_and_escape_data_text(self) -> None:
        charts = [
            svg.line_chart("t <x>", [("a<", [(0, 1), (1, 2)], "#000", False)], "x", "y"),
            svg.bar_chart("t", [("a<b", 1.0, "#000"), ("c", 2.0, "#111")], "y"),
            svg.forest_plot("t", [("r<", 0.1, 0.0, 0.2, "#000"), ("s", 0.1, None, None, "#111")], "x"),
            svg.scatter("t", [(1.0, 2.0), (2.0, 3.0)], "x", "y", diagonal=True),
            svg.scatter("t", [(1e-5, 2.0), (1e-4, 3.0)], "x", "y", log_x=True),
            svg.histogram("t", [0.1, 0.2, 0.2, 0.9], 5, "x"),
        ]
        for chart in charts:
            self.assertTrue(chart.startswith("<svg") and chart.endswith("</svg>"))
            self.assertIn("<title>", chart)
            self.assertEqual(chart.count("<svg"), 1)
        self.assertIn("t &lt;x&gt;", charts[0])
        self.assertNotIn("a<b", charts[1])

    def test_a_palette_without_two_identical_colours_is_used(self) -> None:
        self.assertEqual(len(set(svg.PALETTE)), len(svg.PALETTE))

    def test_11_nothing_imports_a_plotting_library_or_touches_the_network_or_a_model_library(self) -> None:
        forbidden = {"matplotlib", "plotly", "bokeh", "seaborn", "numpy", "scipy", "requests", "socket", "http", "torch", "transformers", "urllib3"}
        for path in sorted((ROOT / "src" / "stt" / "reportbuilder").glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                for name in names:
                    self.assertNotIn(name.split(".")[0], forbidden, f"{path.name} imports {name}")
                    self.assertNotEqual(name, "urllib.request", path.name)
        code = "import sys; sys.path.insert(0, 'src'); import stt.reportbuilder; print('torch' in sys.modules, 'transformers' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False False")

    def test_the_builder_logs_nothing_at_info_level_for_a_normal_report(self) -> None:
        with self.assertNoLogs("stt.reportbuilder", level=logging.WARNING):
            ReportBuilder().build(full_data())


def top_of(html: str) -> str:
    return html.split("<h2")[0]


class VerdictLineTests(unittest.TestCase):
    """The line at the top (owner review, DECISIONS 66): Baseline retained / Proven improvement / No proven improvement."""

    def selection(self, verdict: str, test: str) -> SelectionView:
        base = full_data()
        row_ = row(1, 1, 0.1, 0.2, 0.15, 0.25, verdict)
        comparison = ComparisonResult(0.15, 0.1, 0.2, 0.9, True, 10, True, "improved", "x")
        return replace(base.selection, test_status=test, table=(row_, *base.selection.table[1:]), test_comparison=comparison if test != "not_evaluated" else None)  # type: ignore[type-var]

    def test_a_winner_that_is_improved_on_val_and_confirmed_by_the_test_is_a_proven_improvement(self) -> None:
        top = top_of(build(full_data()))
        self.assertIn('class="vline proven"', top)
        self.assertIn("<b>Proven improvement</b>", top)
        self.assertIn("WINNER: trial 1", top)

    def test_a_winner_with_no_clear_difference_is_never_called_a_winner_alone(self) -> None:
        for test in ("confirmed", "not_confirmed", "not_evaluated"):
            with self.subTest(test=test):
                data = replace(full_data(), selection=self.selection("no_clear_difference", test))
                top = top_of(build(data))
                self.assertIn('class="vline unproven"', top)
                self.assertIn("<b>No proven improvement</b>", top)
                self.assertIn("not clearly better than the baseline on Val (verdict: no_clear_difference)", top)
                self.assertNotIn("winner", top.lower())  # neither the heading nor the sentences of the builder use the word (the Selector's stored reason is not in this test data)
                self.assertIn("Best finalist: trial 1", top)
                self.assertIn("(not proven better than the baseline)", top)

    def test_the_captions_of_the_other_sections_follow_the_same_rule(self) -> None:
        html = build(replace(full_data(), selection=self.selection("no_clear_difference", "confirmed")))
        self.assertIn("Files of the best finalist on val", html)  # the listening list
        self.assertNotIn("Files of the winner", html)
        self.assertIn("Files of the winner", build(full_data()))  # a proven improvement keeps the word

    def test_improved_on_val_but_the_test_does_not_confirm_or_was_not_evaluated_is_no_proven_improvement(self) -> None:
        not_confirmed = top_of(build(replace(full_data(), selection=self.selection("improved", "not_confirmed"))))
        self.assertIn("<b>No proven improvement</b>", not_confirmed)
        self.assertIn("the one-time Test does not confirm it", not_confirmed)
        self.assertNotIn("WINNER", not_confirmed)
        unseen = top_of(build(replace(full_data(), selection=self.selection("improved", "not_evaluated"))))  # the owner's rule: Val improved but no Test is not proven
        self.assertIn("<b>No proven improvement</b>", unseen)
        self.assertIn("the Test was not evaluated, so the improvement is not confirmed", unseen)
        self.assertNotIn("WINNER", unseen)

    def test_a_retained_baseline_says_so_and_a_missing_selection_has_no_line(self) -> None:
        top = top_of(build(full_data(winner=False)))
        self.assertIn('class="vline retained"', top)
        self.assertIn("<b>Baseline retained</b>", top)
        self.assertNotIn("WINNER", top)
        self.assertNotIn("vline", top_of(build(full_data(selection=None))).split("<body>")[1])

    def test_the_line_comes_before_the_panel_and_is_derived_only_from_stored_values(self) -> None:
        top = top_of(build(full_data()))
        self.assertLess(top.index("vline"), top.index('class="panel"'))
        self.assertEqual(top_of(build(full_data())), top_of(build(full_data())))  # deterministic


class NoiseNoteTests(unittest.TestCase):
    TINY = [1.0000, 1.0004, 1.0002]

    def tiny_data(self, **settings: Any) -> ReportData:
        flat = {t: history([v, v + 0.0003, v + 0.0001], [2.0, 1.9, 1.8]) for t, v in enumerate(self.TINY)}
        trials = tuple(TrialView(i, {"learning_rate": 1e-5 * (i + 1), "grad_accum_steps": 2 + 2 * (i % 2)}, "complete", v, 100, 30.0) for i, v in enumerate(self.TINY))
        base = full_data(settings=ReportConfig(**settings))
        return replace(base, final_histories=flat, search_histories=flat, search=replace(base.search, trials=trials))  # type: ignore[type-var]

    def test_a_tiny_val_loss_range_gets_the_note_under_the_curves_the_search_curves_and_the_scatters(self) -> None:
        html = build(self.tiny_data())
        self.assertEqual(html.count("Differences this small are noise"), 2 + 2)  # two Val-loss curve charts and two parameter scatters
        self.assertIn("report.loss_noise_range = 0.01", html)
        curves = html.split('id="training_curves"')[1].split('id="distribution"')[0]
        self.assertEqual(curves.count("Differences this small are noise"), 2)
        self.assertEqual(html.split('id="search"')[1].split('id="learning_curve"')[0].count("Differences this small are noise"), 2)
        self.assertIn("<svg", html)  # the axis stays automatic: the charts are still drawn

    def test_a_range_at_or_above_the_threshold_or_a_single_value_gets_no_note(self) -> None:
        self.assertNotIn("Differences this small are noise", build(full_data()))  # the default test data varies by 0.8
        self.assertNotIn("Differences this small are noise", build(self.tiny_data(loss_noise_range=0.0)))  # a threshold of 0 never fires
        single = replace(full_data(), final_histories={1: history([1.0, None, None], [2.0, 1.9, 1.8])}, search_histories={})
        self.assertNotIn("Differences this small are noise", build(single).split('id="search"')[0])  # one Val value: nothing to compare

    def test_the_threshold_is_configurable(self) -> None:
        wide = build(replace(full_data(settings=ReportConfig(loss_noise_range=5.0))))
        self.assertIn("Differences this small are noise", wide)
        self.assertIn("report.loss_noise_range = 5", wide)

    def test_the_defaults_of_the_new_keys(self) -> None:
        c = ReportConfig()
        self.assertEqual((c.loss_noise_range, c.absolute_audio_paths), (0.01, False))


class AudioPathTests(unittest.TestCase):
    ENTRY = ReviewEntry("f001", "g01", Path("C:/Users/9090/stt/data/extracted/daf_yomi/audio/a b.mp3"), HEBREW, HEBREW, 0.3, "x", "worst")

    def page(self, folder: str | None, **settings: Any) -> str:
        data = full_data(settings=ReportConfig(**settings))
        win = result("w", files_with([1] * 40), review=(self.ENTRY, ReviewEntry("f002", "g02", Path("D:/elsewhere/b.mp3"), HEBREW, HEBREW, 0.2, "x", "random")))
        return build(replace(data, finalist_val={1: win}, run_info=replace(data.run_info, data_folder=folder)))

    def test_the_path_is_relative_to_the_data_folder_by_default_and_the_link_points_at_the_real_file(self) -> None:
        html = self.page("data/extracted/daf_yomi")  # a relative folder, an absolute path
        self.assertIn(">audio/a b.mp3</a>", html)
        self.assertIn('href="file:///C:/Users/9090/stt/data/extracted/daf_yomi/audio/a%20b.mp3"', html)
        self.assertIn("relative to the data folder", html)
        self.assertNotIn(">C:/Users/9090/stt/data/extracted/daf_yomi/audio/a b.mp3<", html)

    def test_a_path_outside_the_data_folder_stays_absolute(self) -> None:
        html = self.page("data/extracted/daf_yomi")
        self.assertIn(">D:/elsewhere/b.mp3</a>", html)

    def test_the_absolute_paths_key_shows_the_full_path(self) -> None:
        html = self.page("data/extracted/daf_yomi", absolute_audio_paths=True)
        self.assertIn(">C:/Users/9090/stt/data/extracted/daf_yomi/audio/a b.mp3</a>", html)
        self.assertIn("Audio paths are shown in full.", html)

    def test_without_a_data_folder_the_paths_are_shown_in_full(self) -> None:
        html = self.page(None)
        self.assertIn(">C:/Users/9090/stt/data/extracted/daf_yomi/audio/a b.mp3</a>", html)
        self.assertIn("Audio paths are shown in full.", html)

    def test_relative_to_folder_is_pure_string_work_for_every_combination(self) -> None:
        from stt.reportbuilder.builder import relative_to_folder

        absolute = Path("C:/Users/9090/stt/data/extracted/daf_yomi/audio/a.mp3")
        self.assertEqual(relative_to_folder(absolute, "data/extracted/daf_yomi"), "audio/a.mp3")  # a relative folder
        self.assertEqual(relative_to_folder(absolute, "C:/Users/9090/stt/data/extracted/daf_yomi"), "audio/a.mp3")  # an absolute folder
        self.assertEqual(relative_to_folder(Path("data/extracted/daf_yomi/audio/a.mp3"), "data/extracted/daf_yomi"), "audio/a.mp3")  # both relative
        self.assertEqual(relative_to_folder(absolute, "data/other"), absolute.as_posix())  # outside
        self.assertEqual(relative_to_folder(absolute, None), absolute.as_posix())
        self.assertEqual(relative_to_folder(Path("data/extracted/daf_yomi"), "data/extracted/daf_yomi"), "data/extracted/daf_yomi")  # nothing left to show


if __name__ == "__main__":
    unittest.main()
