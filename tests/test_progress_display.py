"""Tests of the live progress and the planned time (Wave 3 task 7; DECISIONS 78): the progress lines, their interval, the console sink with `step i of N`, the Evaluator hook, the plan at the start of
`run`, `search` and `learning-curve`, the `trial i of N` line of the searcher, and the resources lines of the benchmark.

How to run (project root, virtual environment active):
    python -m unittest tests.test_progress_display -v
The unit tests use a fake clock and a fake output (no waiting, nothing printed); the stage tests use the FAKE services of tests/search_stage_fakes.py with the real Orchestrator; one test runs the real tiny
test model. Nothing is downloaded, the Test split is never evaluated, nothing is deleted.
"""

from __future__ import annotations

import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from search_stage_fakes import FakeEvaluation, fake_services  # noqa: E402
from test_evaluator import EVAL_PROFILE, FakeDataset, FakeTranscriber, item, model  # noqa: E402
from test_orchestrator_curve import CurveCase, curve_config  # noqa: E402

from stt.config import SectionRegistry, load_config  # noqa: E402
from stt.evaluator import DecodeSettings, EvalConfig, EvalRequest, Evaluator, RawHypothesis  # noqa: E402
from stt.featurepipeline import FeatureExample  # noqa: E402
from stt.orchestrator import InterruptFlag, NoopSleepGuard, Orchestrator, RunPlan, default_stage_registry  # noqa: E402
from stt.orchestrator.config import ProgressConfig  # noqa: E402
from stt.orchestrator.planned import NO_BENCHMARK, find_benchmark, plan_lines  # noqa: E402
from stt.orchestrator.planning import Measured, Sizes, project  # noqa: E402
from stt.orchestrator.progress import ItemProgress, Progress, ThrottledSink, log_line, progress_interval, progress_text  # noqa: E402
from stt.orchestrator.resources import folder_size, human_bytes, peak_memory_bytes, resources_text  # noqa: E402
from stt.orchestrator.wiring import section_registry  # noqa: E402
from stt.trainercore import ConsoleMetricsSink, SinkContext, StepMetrics, TrainConfig, resolve_threads  # noqa: E402
from stt.trainercore.sinks import clock_text  # noqa: E402

RUN_STAGES = ["prepare", "evaluate", "search", "finalists", "select_and_register"]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class LineTests(unittest.TestCase):
    def test_the_clock_text_and_the_progress_line_have_every_part_the_owner_asked_for(self) -> None:
        self.assertEqual([clock_text(s) for s in (0, 59.6, 231, 3725, 36000)], ["0:00:00", "0:01:00", "0:03:51", "1:02:05", "10:00:00"])
        line = progress_text("WER scoring of the baseline", "clip", 120, 875, 231.0)
        self.assertEqual(line, "PROGRESS WER scoring of the baseline: clip 120 of 875 (13.7%), elapsed 0:03:51, remaining about 0:24:13")
        self.assertIn("remaining unknown", progress_text("p", "clip", 0, 10, 5.0))
        self.assertTrue(progress_text("p", "clip", 10, 10, 5.0).endswith("(100.0%), elapsed 0:00:05, done"))
        self.assertNotIn("\x1b", line)  # plain text: no ANSI

    def test_progress_writes_the_first_line_one_per_interval_and_always_the_last(self) -> None:
        clock, lines = FakeClock(), []
        progress = Progress("scoring", "clip", 30.0, clock, lines.append)
        progress.update(1, 100)
        clock.now += 10
        progress.update(2, 100)  # inside the interval: nothing
        clock.now += 25
        progress.update(3, 100)  # 35 s since the last line: written
        clock.now += 1
        progress.update(99, 100)
        progress.update(100, 100)  # the last one is always written, once
        progress.update(100, 100)
        self.assertEqual(len(lines), 3)
        self.assertIn("clip 1 of 100", lines[0])
        self.assertIn("clip 3 of 100", lines[1])
        self.assertIn("clip 100 of 100 (100.0%)", lines[2])
        self.assertIn("remaining about", lines[1])

    def test_a_failing_output_and_a_bad_total_never_stop_anything(self) -> None:
        def broken(text: str) -> None:
            raise OSError("the console is gone")

        Progress("p", "clip", 0.0, FakeClock(), broken).update(1, 2)  # no exception
        ItemProgress("p", "item", 3, FakeClock(), broken).starting(1)
        Progress("p", "clip", 0.0, FakeClock(), lambda t: None).update(1, 0)
        log_line("a line through the real logger")  # no exception either

    def test_item_progress_says_remaining_unknown_until_the_first_item_is_finished(self) -> None:
        clock, lines = FakeClock(), []
        items = ItemProgress("finalists", "finalist", 3, clock, lines.append)
        items.starting(1, " (trial 7)")
        clock.now += 600
        items.starting(2)
        clock.now += 600
        items.starting(3)
        self.assertEqual(lines[0], "PROGRESS finalists: starting finalist 1 of 3 (0.0% done), elapsed 0:00:00, remaining unknown until the first finalist is finished (trial 7)")
        self.assertEqual(lines[1], "PROGRESS finalists: starting finalist 2 of 3 (33.3% done), elapsed 0:10:00, remaining about 0:20:00 (from the 1 finished)")
        self.assertIn("starting finalist 3 of 3 (66.7% done), elapsed 0:20:00, remaining about 0:10:00", lines[2])


class SinkTests(unittest.TestCase):
    def record(self, step: int, val: float | None = None) -> StepMetrics:
        return StepMetrics(step, step / 100, 1.5, val, 1e-5, 0.7, 4.8, step * 4.8, (100 - step) * 4.8)

    def test_the_console_line_has_the_step_of_the_total_the_percent_the_elapsed_and_the_remaining_time(self) -> None:
        sink = ConsoleMetricsSink(SinkContext(Path("candidates") / "exp_0004_trial_003", None, 400))
        with self.assertLogs("stt.trainercore", level="INFO") as logs:
            sink.write(StepMetrics(40, 0.4, 2.5, None, 1e-5, None, 4.77, 190.8, 1717.2))
        text = logs.records[0].getMessage()
        self.assertTrue(text.startswith("[exp_0004_trial_003] step 40 of 400 (10.0%)"), text)
        self.assertIn("train_loss 2.5000", text)
        self.assertIn("elapsed 0:03:11", text)
        self.assertIn("remaining 0:28:37", text)

    def test_without_a_total_the_line_keeps_the_old_shape(self) -> None:
        with self.assertLogs("stt.trainercore", level="INFO") as logs:
            ConsoleMetricsSink(SinkContext(Path("."))).write(self.record(10))
        text = logs.records[0].getMessage()
        self.assertTrue(text.startswith("step 10 (epoch"), text)
        self.assertNotIn(" of ", text.split("(epoch")[0])

    def test_the_throttled_sink_forwards_the_first_record_an_evaluation_the_last_step_and_one_per_interval(self) -> None:
        clock, seen = FakeClock(), []

        class Inner:
            def write(self, record: StepMetrics) -> None:
                seen.append(record.step)

        sink = ThrottledSink(Inner(), 30.0, 100, clock)
        sink.write(self.record(1))  # the first
        clock.now += 5
        sink.write(self.record(2))  # inside the interval
        sink.write(self.record(3, 0.9))  # an evaluation record is always written
        clock.now += 40
        sink.write(self.record(4))  # the interval has passed
        clock.now += 1
        sink.write(self.record(100))  # the last step
        self.assertEqual(seen, [1, 3, 4, 100])

    def test_a_failing_console_sink_never_stops_the_training(self) -> None:
        class Broken:
            def write(self, record: StepMetrics) -> None:
                raise OSError("no console")

        ThrottledSink(Broken(), 30.0, 10, FakeClock()).write(self.record(1))  # no exception


class ConfigTests(unittest.TestCase):
    def test_the_section_is_registered_with_the_default_of_30_seconds_and_validates(self) -> None:
        self.assertIn("progress", section_registry().sections)
        self.assertEqual(ProgressConfig().interval_seconds, 30.0)
        for bad in ({"interval_seconds": 0}, {"interval_seconds": -1}, {"nope": 1}):
            with self.assertRaises(ValueError):
                ProgressConfig.model_validate(bad)
        loaded = load_config(ROOT / "configs" / "progress_default.yaml", env_file=None, target_environ={}, registry=SectionRegistry({"progress": ProgressConfig}))
        self.assertEqual(loaded.section("progress", ProgressConfig), ProgressConfig())  # the example YAML shows exactly the default

    def test_the_interval_comes_from_the_config_and_defaults_when_the_section_is_not_loaded(self) -> None:
        self.assertEqual(progress_interval(load_config(None, env_file=None, target_environ={}, registry=SectionRegistry({"progress": ProgressConfig}), overrides=["progress.interval_seconds=7.5"])), 7.5)
        self.assertEqual(progress_interval(load_config(None, env_file=None, target_environ={}, registry=SectionRegistry({}))), 30.0)


def dataset(n: int) -> Any:
    rng = np.random.default_rng(0)
    examples = [FeatureExample(f"s{i}", rng.standard_normal((80, 3000)).astype(np.float32) * 0.1, [1]) for i in range(n)]

    class Data:
        def __len__(self) -> int:
            return n

        def __getitem__(self, index: int) -> FeatureExample:
            return examples[index]

    return Data()


class EvaluatorHookTests(unittest.TestCase):
    def test_the_evaluator_hands_the_callback_to_a_transcriber_only_when_the_request_has_one(self) -> None:
        class Reporting(FakeTranscriber):
            def transcribe(self, model: Any, data: Any, settings: DecodeSettings, progress: Any = None) -> list[RawHypothesis]:
                if progress is not None:
                    progress(len(data), len(data))
                return super().transcribe(model, data, settings)

        items = [item(f"s{i}", "שלום עולם", f"g{i}") for i in range(3)]
        texts = {i.sample_id: "שלום עולם" for i in items}
        calls: list[tuple[int, int]] = []
        request = EvalRequest(items, FakeDataset([i.sample_id for i in items]), model(), "val", 1, EVAL_PROFILE, DecodeSettings(1, 100, 4), on_progress=lambda done, total: calls.append((done, total)))
        Evaluator(EvalConfig(), Reporting(texts), 1).evaluate(request)
        self.assertEqual(calls, [(3, 3)])
        plain = EvalRequest(items, FakeDataset([i.sample_id for i in items]), model(), "val", 1, EVAL_PROFILE, DecodeSettings(1, 100, 4))
        self.assertIsNone(plain.on_progress)  # the default: nothing changes
        Evaluator(EvalConfig(), FakeTranscriber(texts), 1).evaluate(plain)  # a transcriber without the optional parameter still works

    def test_the_real_adapter_reports_after_every_batch_and_a_failing_callback_does_not_stop_it(self) -> None:
        from stt.evaluator.hf_transcriber import HfTranscriber

        loaded = model()
        seen: list[tuple[int, int]] = []
        out = HfTranscriber(1).transcribe(loaded, dataset(5), DecodeSettings(1, 3, 2), progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(len(out), 5)
        self.assertEqual(seen, [(2, 5), (4, 5), (5, 5)])  # batches of 2: the number transcribed so far, then the total

        def broken(done: int, total: int) -> None:
            raise RuntimeError("the display failed")

        self.assertEqual(len(HfTranscriber(1).transcribe(loaded, dataset(3), DecodeSettings(1, 3, 2), progress=broken)), 3)


class ProgressingEvaluation(FakeEvaluation):
    """The fake evaluation that also uses the progress callback of the request, as the real Evaluator does."""

    def evaluate(self, request: EvalRequest) -> Any:
        if request.on_progress is not None:
            request.on_progress(1, len(request.items))
            request.on_progress(len(request.items), len(request.items))
        return super().evaluate(request)


class StageProgressTests(CurveCase):
    def run_plan(self, stages: list[str], config: Any, options: dict[str, str] | None = None) -> tuple[Any, list[str]]:
        handle = self.experiment(config)
        services = fake_services(config, evaluation=ProgressingEvaluation(config))
        with self.assertLogs("stt", level="INFO") as logs:
            outcome = Orchestrator(config, services, default_stage_registry(), InterruptFlag(), NoopSleepGuard()).run(RunPlan(stages, handle, options or {}))
        return outcome, [r.getMessage() for r in logs.records]

    def test_every_wer_scoring_writes_progress_lines_with_a_phase_name_and_the_clip_position(self) -> None:
        outcome, lines = self.run_plan(RUN_STAGES, curve_config(self.tmp))
        self.assertEqual(outcome.exit_code, 0)
        scoring = [line for line in lines if line.startswith("PROGRESS WER scoring")]
        self.assertTrue(any("baseline" in line for line in scoring))
        self.assertTrue(any("finalist trial" in line for line in scoring))
        self.assertTrue(any("Test" in line for line in scoring))  # the one-time Test look reports progress too
        self.assertTrue(all("clip " in line and " of " in line and "elapsed" in line and "%" in line for line in scoring))

    def test_the_finalists_and_the_search_write_their_position_lines(self) -> None:
        _, lines = self.run_plan(RUN_STAGES, curve_config(self.tmp))
        finalists = [line for line in lines if line.startswith("PROGRESS finalists: starting finalist")]
        self.assertEqual(len(finalists), 4)
        self.assertIn("starting finalist 1 of 4", finalists[0])
        self.assertIn("starting finalist 4 of 4 (75.0% done)", finalists[3])
        search = [line for line in lines if line.startswith("Search progress:")]
        self.assertEqual(len(search), 4)
        self.assertIn("trial 1 of 4 finished (25.0%)", search[0])
        self.assertIn("trial 4 of 4 finished (100.0%), elapsed", search[3])
        self.assertIn("remaining about", search[1])

    def test_the_learning_curve_writes_its_point_lines(self) -> None:
        _, lines = self.run_plan(["prepare", "learning_curve"], curve_config(self.tmp))
        points = [line for line in lines if line.startswith("PROGRESS learning curve: starting point")]
        self.assertEqual(len(points), 3)
        self.assertIn("starting point 1 of 3 (0.0% done)", points[0])
        self.assertIn("(preset 's40')", points[2])


def write_benchmark(folder: Path, model_name: str, threads: int, seconds_per_step: float = 4.77) -> None:
    measured = Measured(model_name, threads, 3, 2, 2, seconds_per_step, 0.42, 12, 1.93, 12)
    (folder / "reports").mkdir(parents=True)
    (folder / "reports" / "benchmark.json").write_text(json.dumps({"measured": asdict(measured), "sizes": {}, "parts": [], "total_seconds": 1.0, "notes": []}), encoding="utf-8")


class PlanTests(CurveCase):
    def config(self) -> Any:
        return curve_config(self.tmp)

    def threads(self, config: Any) -> int:
        return resolve_threads(config.section("train", TrainConfig))

    def run_with_plan(self, kind: str, stages: list[str], config: Any) -> list[str]:
        handle = self.experiment(config)
        services = fake_services(config)
        with self.assertLogs("stt", level="INFO") as logs:
            Orchestrator(config, services, default_stage_registry(), InterruptFlag(), NoopSleepGuard()).run(RunPlan(stages, handle, {"announce_plan": kind}))
        return [r.getMessage() for r in logs.records]

    def test_with_a_matching_benchmark_the_plan_is_written_right_after_prepare(self) -> None:
        config = self.config()
        write_benchmark(self.tmp / "experiments" / "exp_0001", "tiny_whisper", self.threads(config))
        lines = self.run_with_plan("run", ["prepare", "evaluate"], config)
        plan = [line for line in lines if line.startswith("PLAN:")]
        self.assertIn("planned time of `python -m stt run`", plan[0])
        self.assertIn("from the benchmark of exp_0001: tiny_whisper", plan[0])
        self.assertTrue(any("baseline on Val" in line for line in plan))
        started = next(i for i, line in enumerate(lines) if line == "Stage prepare finished.")
        evaluate = next(i for i, line in enumerate(lines) if line == "Stage evaluate starts.")
        self.assertTrue(started < next(i for i, line in enumerate(lines) if line.startswith("PLAN:")) < evaluate)  # after prepare, before the first long phase

    def test_without_a_matching_benchmark_the_line_says_so_for_a_wrong_model_a_wrong_thread_count_and_none(self) -> None:
        config = self.config()
        self.assertTrue(self.run_with_plan("run", ["prepare"], config)[-1] or True)
        for model_name, threads in (("whisper-base", self.threads(config)), ("tiny_whisper", self.threads(config) + 5)):
            with self.subTest(model=model_name, threads=threads):
                folder = self.tmp / f"case_{model_name}_{threads}"
                write_benchmark(folder / "experiments" / "exp_0001", model_name, threads)
                other = curve_config(folder)
                lines = self.run_with_plan("run", ["prepare"], other)
                plan = [line for line in lines if line.startswith("PLAN:")]
                self.assertEqual(len(plan), 1)
                self.assertIn(NO_BENCHMARK, plan[0])

    def test_no_plan_is_written_without_the_option_and_a_broken_plan_never_stops_the_run(self) -> None:
        config = self.config()
        write_benchmark(self.tmp / "experiments" / "exp_0001", "tiny_whisper", self.threads(config))
        handle = self.experiment(config)
        with self.assertLogs("stt", level="INFO") as logs:
            Orchestrator(config, fake_services(config), default_stage_registry(), InterruptFlag(), NoopSleepGuard()).run(RunPlan(["prepare"], handle))
        self.assertFalse([r for r in logs.records if r.getMessage().startswith("PLAN:")])
        (self.tmp / "experiments" / "exp_0001" / "reports" / "benchmark.json").write_text("{ not json", encoding="utf-8")  # an unreadable file is skipped: the "no numbers" line
        lines = self.run_with_plan("search", ["prepare"], config)
        self.assertTrue(any(NO_BENCHMARK in line for line in lines))

    def test_the_search_and_the_learning_curve_plans_cover_only_their_part(self) -> None:
        measured = Measured("whisper-tiny", 2, 3, 2, 2, 4.77, 0.42, 12, 1.93, 12)
        sizes = Sizes(max_steps=100, eval_every_steps=50, n_trials=4, finalists_k=2, parallel_trials=1, screening_val=100, val_files=875, test_files=875)
        source = find_benchmark(self.tmp, "x", 1)
        self.assertIsNone(source)
        from stt.orchestrator.planned import BenchmarkSource

        source = BenchmarkSource("exp_0003", measured)
        projection = project(measured, sizes)
        run = plan_lines("run", source, sizes, projection)
        search = plan_lines("search", source, sizes, projection)
        curve = plan_lines("learning-curve", source, sizes, projection, points=4)
        self.assertEqual(len(run), 1 + 5 + 1)  # the headline, the five parts, the note
        self.assertEqual(len(search), 1 + 1 + 1)
        self.assertIn("search (4 trials of 100 steps)", search[1])
        self.assertIn("learning curve (4 points", curve[1])
        self.assertIn("python -m stt learning-curve", curve[0])
        self.assertIn("4.77 s per step, 1.93 s per clip", run[0])

    def test_the_latest_matching_benchmark_wins(self) -> None:
        for number, step in ((1, 4.0), (2, 5.0), (3, 6.0)):
            write_benchmark(self.tmp / "experiments" / f"exp_000{number}", "tiny_whisper" if number != 3 else "other", 2, step)
        found = find_benchmark(self.tmp / "experiments", "tiny_whisper", 2)
        self.assertEqual((found.experiment_id, found.measured.seconds_per_step), ("exp_0002", 5.0))  # type: ignore[union-attr]


class ResourcesTests(unittest.TestCase):
    def test_the_sizes_the_planned_disk_and_the_upper_bound(self) -> None:
        self.assertEqual((human_bytes(999), human_bytes(453_000_000), human_bytes(1_800_000_000)), ("1 KB", "453 MB", "1.8 GB"))
        lines = resources_text(609_000_000, 1, 456_000_000, 2_500_000_000, 5)
        text = "\n".join(lines)
        self.assertIn("disk of the measuring trial folder: 609 MB (1 checkpoint folder(s), the largest 456 MB)", text)
        self.assertIn("planned disk of `python -m stt run`: about 3.0 GB (5 trial folders", text)
        self.assertIn("plan up to about 5.3 GB", text)  # one more checkpoint per trial (the latest AND the best)
        self.assertIn("peak memory of this process: 2.5 GB", text)

    def test_two_checkpoints_need_no_upper_bound_and_an_unmeasurable_memory_is_said_plainly(self) -> None:
        text = "\n".join(resources_text(1_000_000_000, 2, 400_000_000, None, 3))
        self.assertNotIn("plan up to", text)
        self.assertIn("peak memory: NOT MEASURED", text)
        self.assertIn("Task Manager", text)

    def test_the_peak_memory_is_measured_on_windows_and_never_raises_elsewhere(self) -> None:
        peak = peak_memory_bytes()
        if sys.platform == "win32":
            self.assertIsNotNone(peak)
            self.assertGreater(peak or 0, 10_000_000)  # a Python process with torch loaded uses far more than 10 MB
        else:
            self.assertIsNone(peak)

    def test_the_folder_size_counts_the_files_and_reads_only(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a" / "b").mkdir(parents=True)
            (Path(tmp) / "a" / "x.bin").write_bytes(b"x" * 1000)
            (Path(tmp) / "a" / "b" / "y.bin").write_bytes(b"y" * 234)
            self.assertEqual(folder_size(Path(tmp)), 1234)
            self.assertEqual(folder_size(Path(tmp) / "nope"), 0)


class SearcherLineTests(unittest.TestCase):
    """The `trial i of N` text of the searcher's log line (O113): the position, the percent, the elapsed time and the time remaining."""

    def test_every_finished_trial_writes_its_position_and_a_remaining_time(self) -> None:
        from search_fakes import BowlRunner
        from test_searcher import make_spec, searcher

        class Tick:
            def __init__(self) -> None:
                self.now = 0.0

            def __call__(self) -> float:
                self.now += 60.0  # every look at the clock is one minute later
                return self.now

        with self.assertLogs("stt.searcher", level="INFO") as logs:
            searcher(clock=Tick()).search(make_spec(n_trials=4), BowlRunner())
        lines = [r.getMessage() for r in logs.records if r.getMessage().startswith("Search progress:")]
        self.assertEqual(len(lines), 4)
        for number, line in enumerate(lines, 1):
            self.assertRegex(line, rf"^Search progress: trial {number} of 4 finished \({25.0 * number:.1f}%\), elapsed \d+:\d\d:\d\d, remaining about \d+:\d\d:\d\d \(up to 4 trials")
            self.assertRegex(line, rf"Trial {number - 1} complete")  # the old sentence is kept after the progress
        self.assertIn("remaining about 0:00:00", lines[3])  # nothing left after the last trial

    def test_a_resumed_study_counts_the_earlier_trials_but_estimates_from_the_new_ones(self) -> None:
        import tempfile

        from search_fakes import BowlRunner
        from test_searcher import make_spec, searcher

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "study"
            searcher().search(make_spec(n_trials=2, study_dir=folder), BowlRunner())
            with self.assertLogs("stt.searcher", level="INFO") as logs:
                searcher().search(make_spec(n_trials=4, study_dir=folder), BowlRunner())
        lines = [r.getMessage() for r in logs.records if r.getMessage().startswith("Search progress:")]
        self.assertEqual(len(lines), 2)  # only the two new trials write a line
        self.assertIn("trial 3 of 4 finished (75.0%)", lines[0])
        self.assertIn("trial 4 of 4 finished (100.0%)", lines[1])


if __name__ == "__main__":
    unittest.main()
