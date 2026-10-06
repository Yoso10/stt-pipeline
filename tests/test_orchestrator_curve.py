"""Unit tests of Orchestrator Part 3: the `learning_curve` and `benchmark` stages, the planning numbers, and the two commands.

How to run (project root, virtual environment active):
    python -m unittest tests.test_orchestrator_curve -v
Most tests use FAKE services (tests/search_stage_fakes.py: no model, no audio; the number of wrong words per file comes from a table by learning-curve point) with the REAL Subsampler, search
service, Comparator and Orchestrator in temporary folders (fast). `RealCurveTests` run both commands on synthetic clips and the stored tiny test model (about 30 seconds). Neither command touches the
Test split: the fake models record every split they are asked for. Nothing is downloaded, nothing is deleted by the code under test.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from search_stage_fakes import FakeEvaluation, SPLIT_SIZES, fake_services  # noqa: E402

from stt.config import AppConfig, load_config  # noqa: E402
from stt.evaluator import ComparisonResult  # noqa: E402
from stt.orchestrator import InterruptFlag, NoopSleepGuard, Orchestrator, OrchestratorError, RunOutcome, RunPlan, SmokeConfig, default_stage_registry, write_smoke_data  # noqa: E402
from stt.orchestrator import cli  # noqa: E402
from stt.orchestrator.curve import CurvePoint, read_params_from, reading  # noqa: E402
from stt.orchestrator.planning import Measured, Sizes, evaluations, format_duration, project  # noqa: E402
from stt.orchestrator.wiring import section_registry  # noqa: E402
from stt.runmanager import ExperimentHandle, ExperimentSpec, ExperimentStore, RunConfig  # noqa: E402

TINY = ROOT / "tests" / "assets" / "tiny_whisper"
SMOKE_YAML = ROOT / "configs" / "smoke.yaml"
GRID = '{"learning_rate": {"kind": "log_float", "low": 1e-5, "high": 1e-4, "points": 2}, "grad_accum_steps": {"kind": "categorical", "choices": [2, 4]}}'
PRESETS = '{"s10": {"size": {"samples": 10}}, "s20": {"size": {"samples": 20}}, "s40": {"size": "all"}}'  # three Train sizes of the 40 fake Train samples


def curve_config(tmp: Path, *overrides: str) -> AppConfig:
    base = [
        f"data.source.path={TINY.as_posix()}", f"model.path={TINY.as_posix()}", f"run.experiments_dir={(tmp / 'experiments').as_posix()}", f"registry.dir={(tmp / 'registry').as_posix()}",
        f"subsample.presets={PRESETS}", 'learning_curve.presets=["s10", "s20", "s40"]', "search.n_trials=4", f"search.parameters={GRID}", "search.screening.preset=smoke", "search.pruning.enabled=false",
        "search.finalists.k=4", "train.cpu_threads=4",
    ]
    return load_config(SMOKE_YAML, overrides=[*base, *overrides], registry=section_registry(), env_file=None, target_environ={})


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class PlanningTests(unittest.TestCase):
    M = Measured("tiny", 2, 3, 4, 4, seconds_per_step=1.0, seconds_per_val_loss_example=0.5, val_loss_probe=12, seconds_per_decode_clip=2.0, decode_probe=12)
    S = Sizes(max_steps=100, eval_every_steps=50, n_trials=6, finalists_k=3, parallel_trials=1, screening_val=100, val_files=875, test_files=875)

    def test_the_number_of_evaluations_of_one_training(self) -> None:
        self.assertEqual([evaluations(100, 50), evaluations(120, 50), evaluations(30, 50), evaluations(50, 50)], [2, 3, 1, 1])

    def test_the_projection_adds_up_from_the_measured_numbers(self) -> None:
        p = project(self.M, self.S)
        training = 100 * 1.0 + 2 * 100 * 0.5  # steps x step time + evaluations x slice x Val-loss time
        self.assertEqual(p.one_training_seconds, training)
        self.assertEqual(
            [round(seconds) for _, seconds in p.parts],
            [875 * 2, round(6 * training), round(3 * training), 3 * 875 * 2, 2 * 875 * 2],  # baseline, search, finalists, finalists on Val, Test (counted, never evaluated)
        )
        self.assertEqual(p.total_seconds, sum(seconds for _, seconds in p.parts))
        self.assertEqual(p.evaluations_per_training, 2)

    def test_the_notes_say_what_the_numbers_do_not_cover(self) -> None:
        notes = " ".join(project(self.M, self.S).notes)
        self.assertIn("this machine", notes)
        self.assertIn("early stopping", notes)
        self.assertIn("Val probe was small", " ".join(project(Measured("t", 2, 3, 4, 4, 1.0, 0.5, 4, 2.0, 4), self.S).notes))
        parallel = " ".join(project(self.M, Sizes(100, 50, 6, 3, 2, 100, 875, 875)).notes)
        self.assertIn("share the same cores", parallel)

    def test_durations_are_written_in_plain_words(self) -> None:
        self.assertEqual([format_duration(s) for s in (5, 59.6, 90, 3600, 5832, 90000)], ["5 s", "1 min", "2 min", "1 h 0 min", "1 h 37 min", "1 d 1 h"])
        self.assertEqual(format_duration(math.nan), "unknown")


class CurveCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.store = ExperimentStore(RunConfig(experiments_dir=self.tmp / "experiments"))

    def experiment(self, config: AppConfig) -> ExperimentHandle:
        return self.store.create(ExperimentSpec(config, None, "0.0.0", {}, "2026-10-06T09:00:00"))

    def execute(self, stages: list[str], config: AppConfig, handle: ExperimentHandle, evaluation: FakeEvaluation | None = None, flag: Any = None, hook: Any = None) -> tuple[RunOutcome, Any]:
        services = fake_services(config, hook, evaluation=evaluation)
        outcome = Orchestrator(config, services, default_stage_registry(), flag or InterruptFlag(), NoopSleepGuard()).run(RunPlan(stages, handle))
        return outcome, services

    @staticmethod
    def call(handle: ExperimentHandle, name: str) -> dict[str, Any]:
        return json.loads((handle.paths.metrics / name / "call.json").read_text(encoding="utf-8"))


class LearningCurveTests(CurveCase):
    STAGES = ["prepare", "learning_curve"]

    def test_the_curve_trains_the_same_settings_on_increasing_sizes_and_scores_the_same_val_set(self) -> None:
        config = curve_config(self.tmp)
        handle = self.experiment(config)
        outcome, services = self.execute(self.STAGES, config, handle, FakeEvaluation(config, curve_errors={"s10": 3, "s20": 2, "s40": 1}))
        self.assertEqual((outcome.exit_code, outcome.status), (0, "completed"))
        s = outcome.stage_results["learning_curve"].summary
        self.assertEqual((s["points"], s["trend"], s["val_files"]), (3, "improving", SPLIT_SIZES["val"]))
        rows = json.loads(str(s["table"]))
        self.assertEqual([(r["preset"], r["train_samples"], r["val_wer"]) for r in rows], [("s10", 10, 0.5), ("s20", 20, round(2 / 6, 4)), ("s40", 40, round(1 / 6, 4))])
        calls = [self.call(handle, f"curve_{p}") for p in ("s10", "s20", "s40")]
        self.assertEqual([c["n_train"] for c in calls], [10, 20, 40])
        self.assertEqual({(c["learning_rate"], c["grad_accum_steps"], c["n_val"]) for c in calls}, {(1e-5, 4, SPLIT_SIZES["val"])})  # the same hyperparameters and the same Val set
        self.assertEqual(sorted(set(services.models.requested)), ["train", "val"])  # the Test split is never built
        self.assertEqual({split for split, _ in services.evaluation.calls}, {"val"})
        for name in ("learning_curve.txt", "learning_curve.csv", "learning_curve.json", "curve/s10.val.json", "curve/s40.json"):
            self.assertTrue((handle.paths.reports / name).is_file(), name)
        for preset in ("s10", "s20", "s40"):
            self.assertEqual(len(list((handle.paths.candidates / f"exp_0001_curve_{preset}").glob(f"exp_0001_curve_{preset}_step_*"))), 1)

    def test_the_reading_says_still_improving_flat_or_got_worse(self) -> None:
        cases = {"improving": ({"s10": 3, "s20": 2, "s40": 1}, "STILL IMPROVING"), "flat": ({"s10": 3, "s20": 2, "s40": 2}, "FLAT"), "worse": ({"s10": 1, "s20": 2, "s40": 3}, "GOT WORSE")}
        for trend, (errors, words) in cases.items():
            with self.subTest(trend=trend):
                config = curve_config(self.tmp / trend)
                handle = ExperimentStore(RunConfig(experiments_dir=self.tmp / trend / "experiments")).create(ExperimentSpec(config, None, "0", {}, "x"))
                outcome, _ = self.execute(self.STAGES, config, handle, FakeEvaluation(config, curve_errors=errors))
                s = outcome.stage_results["learning_curve"].summary
                self.assertEqual(s["trend"], trend)
                self.assertIn(words, str(s["reading"]))
                text = (handle.paths.reports / "learning_curve.txt").read_text(encoding="utf-8")
                self.assertIn("READING: " + words, text)
                self.assertIn("preset     Train samples  minutes   Val WER", text)

    def test_the_csv_and_json_hold_the_curve_data(self) -> None:
        config = curve_config(self.tmp)
        handle = self.experiment(config)
        self.execute(self.STAGES, config, handle, FakeEvaluation(config, curve_errors={"s10": 3, "s20": 2, "s40": 1}))
        rows = (handle.paths.reports / "learning_curve.csv").read_text(encoding="utf-8-sig").splitlines()
        self.assertEqual(rows[0].split(",")[:4], ["preset", "train_samples", "train_minutes", "val_wer"])
        self.assertEqual([r.split(",")[0] for r in rows[1:]], ["s10", "s20", "s40"])
        data = json.loads((handle.paths.reports / "learning_curve.json").read_text(encoding="utf-8"))
        self.assertEqual((data["trend"], len(data["points"]), data["hyperparameters_from"]), ("improving", 3, "train.hyperparams of the config"))

    def test_a_rerun_skips_and_an_interrupted_point_resumes_without_training_the_finished_ones_again(self) -> None:
        config = curve_config(self.tmp)
        handle = self.experiment(config)
        flag = InterruptFlag()
        first, _ = self.execute(self.STAGES, config, handle, flag=flag, hook=lambda call, step: flag.set() if (call, step) == (2, 100) else None)  # the second point is stopped at its second evaluation
        self.assertEqual((first.exit_code, first.status), (130, "stopped"))
        self.assertTrue((handle.paths.reports / "curve" / "s10.json").is_file())
        self.assertFalse((handle.paths.reports / "curve" / "s20.json").exists())
        second, services = self.execute(self.STAGES, config, handle)
        self.assertEqual((second.exit_code, second.status), (0, "completed"))
        self.assertEqual(services.training.calls, 2)  # the second point (resumed) and the third; the first is not trained again
        self.assertTrue(self.call(handle, "curve_s20")["resume"])
        third, services3 = self.execute(self.STAGES, config, handle)
        self.assertEqual([r.status for r in third.stage_results.values()], ["skipped", "skipped"])
        self.assertEqual(services3.training.calls, 0)

    def test_a_size_larger_than_the_data_fails_in_the_subsampler_naming_the_size(self) -> None:
        config = curve_config(self.tmp, 'subsample.presets={"s10": {"size": {"samples": 10}}, "s999": {"size": {"samples": 999}}}', 'learning_curve.presets=["s10", "s999"]')
        outcome, _ = self.execute(self.STAGES, config, self.experiment(config))
        self.assertEqual(outcome.status, "failed")
        self.assertIn("999", outcome.message or "")
        self.assertEqual(outcome.exit_code, 3)

    def test_an_unknown_preset_is_a_config_error_listing_the_known_ones(self) -> None:
        config = curve_config(self.tmp, 'learning_curve.presets=["s10", "nope"]')
        outcome, _ = self.execute(self.STAGES, config, self.experiment(config))
        self.assertEqual(outcome.exit_code, 2)
        self.assertIn("Known presets", outcome.message or "")

    def test_the_reading_with_too_few_groups_or_one_point_says_so(self) -> None:
        config = curve_config(self.tmp)
        handle = self.experiment(config)
        outcome, _ = self.execute(self.STAGES, config, handle, FakeEvaluation(config, curve_errors={"s10": 3, "s20": 1, "s40": 1}))
        self.assertEqual(outcome.exit_code, 0)

        def point(preset: str, minutes: float, wer_errors: int) -> CurvePoint:
            fake = FakeEvaluation(config, curve_errors={preset: wer_errors})
            from search_stage_fakes import FakeModels, make_samples
            from stt.evaluator import EvalRequest
            from stt.orchestrator.stagehelpers import make_items
            from stt.orchestrator.types import PreparedData

            prepared = PreparedData({"val": make_samples("val", 3)}, 1, str, {}, {})
            model = FakeModels().load(Path(f"exp_0001_curve_{preset}_step_00001"))
            req = EvalRequest(make_items(prepared, "val", frozenset(), len), [], model, "val", 1, str, fake.decode_settings(model, []))  # type: ignore[arg-type]
            return CurvePoint(preset, 10, minutes, 3, 0.1, 1.0, "m", fake.evaluate(req))

        a, b = point("s10", 1.0, 3), point("s20", 2.0, 1)
        insufficient = ComparisonResult(0.33, None, None, 0.9, True, 3, False, "insufficient_groups", "only 3 groups")
        trend, text, _ = reading([a, b], lambda x, y: insufficient)
        self.assertEqual(trend, "unclear")
        self.assertIn("no confidence statement", text)
        self.assertEqual(reading([a], lambda x, y: insufficient)[0], "too_few")


class ParamsFromTests(CurveCase):
    def run_experiment(self, val_errors: dict[int, int]) -> ExperimentHandle:
        config = curve_config(self.tmp)
        handle = self.experiment(config)
        services = fake_services(config, evaluation=FakeEvaluation(config, val_errors=val_errors))
        stages = ["prepare", "evaluate", "search", "finalists", "select_and_register"]
        outcome = Orchestrator(config, services, default_stage_registry(), InterruptFlag(), NoopSleepGuard()).run(RunPlan(stages, handle))
        self.assertEqual(outcome.exit_code, 0)
        return handle

    def test_the_hyperparameters_come_from_the_winner_of_an_earlier_experiment(self) -> None:
        self.run_experiment({0: 3, 1: 3, 2: 1, 3: 3})  # trial 2 (learning rate 1e-4, accumulation 2) wins
        config = curve_config(self.tmp, "learning_curve.params_from=exp_0001")
        handle = self.experiment(config)
        outcome, _ = self.execute(["prepare", "learning_curve"], config, handle)
        s = outcome.stage_results["learning_curve"].summary
        self.assertIn("the selected winner (trial 2) of exp_0001", str(s["hyperparameters_from"]))
        for preset in ("s10", "s20", "s40"):
            call = self.call(handle, f"curve_{preset}")
            self.assertEqual((call["learning_rate"], call["grad_accum_steps"]), (1e-4, 2))

    def test_a_retained_baseline_falls_back_to_the_best_finalist_and_says_so(self) -> None:
        self.run_experiment({0: 3, 1: 3, 2: 3, 3: 3})
        params, source = read_params_from(self.tmp / "experiments", "exp_0001")
        self.assertIn("the best finalist (trial 1) of exp_0001 (its baseline was retained)", source)
        self.assertEqual(params, {"learning_rate": 1e-5, "grad_accum_steps": 4})

    def test_a_missing_experiment_or_search_result_is_a_config_error_that_lists_what_exists(self) -> None:
        self.run_experiment({0: 2, 1: 1, 2: 3, 3: 3})
        with self.assertRaises(OrchestratorError) as raised:
            read_params_from(self.tmp / "experiments", "exp_0099")
        self.assertEqual(raised.exception.exit_code, 2)
        self.assertIn("exp_0001", str(raised.exception))
        (self.tmp / "experiments" / "exp_0001" / "reports" / "search_trials.json").unlink()
        with self.assertRaises(OrchestratorError) as raised:
            read_params_from(self.tmp / "experiments", "exp_0001")
        self.assertIn("no readable search result", str(raised.exception))

    def test_the_command_line_option_sets_the_config_value(self) -> None:
        with mock.patch.object(cli, "_run_pipeline", return_value=0) as run:
            self.assertEqual(cli.main(["learning-curve", "--params-from", "exp_0003"]), 0)
        self.assertIn("learning_curve.params_from=exp_0003", run.call_args.args[1])
        with mock.patch.object(cli, "_run_pipeline", return_value=0) as run:
            cli.main(["learning-curve"])
        self.assertNotIn("learning_curve.params_from=exp_0003", run.call_args.args[1])
        self.assertEqual(cli.PLANS["learning-curve"], ["prepare", "learning_curve"])
        self.assertEqual(cli.PLANS["benchmark"], ["prepare", "benchmark"])


class BenchmarkTests(CurveCase):
    STAGES = ["prepare", "benchmark"]

    def test_the_benchmark_measures_the_three_times_and_projects_the_run_without_touching_the_test(self) -> None:
        config = curve_config(self.tmp)
        handle = self.experiment(config)
        outcome, services = self.execute(self.STAGES, config, handle)
        self.assertEqual((outcome.exit_code, outcome.status), (0, "completed"))
        s = outcome.stage_results["benchmark"].summary
        probe = 3 * 2  # benchmark.steps x eval.batch_size (smoke.yaml: 2)
        self.assertEqual((s["seconds_per_step"], s["seconds_per_val_loss_example"], s["seconds_per_decode_clip"], s["threads"]), (0.01, round(0.1 / (3 * probe), 4), round(0.1 / probe, 3), 4))
        measured = Measured("tiny_whisper", 4, 3, 4, 4, 0.01, 0.1 / (3 * probe), probe, 0.1 / probe, probe)
        sizes = Sizes(150, 3, 4, 4, 1, SPLIT_SIZES["val"], SPLIT_SIZES["val"], SPLIT_SIZES["test"])  # eval_every_steps 3 comes from smoke.yaml
        self.assertEqual(s["projected_run_seconds"], round(project(measured, sizes).total_seconds))  # the numbers of the file are the numbers of the model
        call = self.call(handle, "benchmark")
        self.assertEqual((call["n_train"], call["n_val"], call["name"]), (min(SPLIT_SIZES["train"], 3 * 4 * 4), probe, "exp_0001_benchmark"))
        self.assertEqual(sorted(set(services.models.requested)), ["train", "val"])  # only the COUNT of the Test clips is read
        self.assertEqual({split for split, _ in services.evaluation.calls}, {"val"})
        text = (handle.paths.reports / "benchmark.txt").read_text(encoding="utf-8")
        for part in ("BENCHMARK on this machine", "per training step", "per example for the Val loss", "per clip for the WER scoring", "TOTAL", "Test, winner and baseline", "this machine, these threads and this model"):
            self.assertIn(part, text)
        data = json.loads((handle.paths.reports / "benchmark.json").read_text(encoding="utf-8"))
        self.assertEqual((data["sizes"]["test_files"], len(data["parts"])), (SPLIT_SIZES["test"], 5))

    def test_the_measuring_folder_is_kept_and_never_reused(self) -> None:
        config = curve_config(self.tmp)
        handle = self.experiment(config)
        (handle.paths.candidates / "exp_0001_benchmark").mkdir(parents=True)
        (handle.paths.candidates / "exp_0001_benchmark" / "keep.txt").write_text("keep me", encoding="utf-8")
        outcome, _ = self.execute(self.STAGES, config, handle)
        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual((handle.paths.candidates / "exp_0001_benchmark" / "keep.txt").read_text(encoding="utf-8"), "keep me")
        self.assertTrue((handle.paths.candidates / "exp_0001_benchmark_2").is_dir())

    def test_a_rerun_prints_the_earlier_measurement_and_the_steps_option_changes_the_probe(self) -> None:
        config = curve_config(self.tmp)
        handle = self.experiment(config)
        self.execute(self.STAGES, config, handle)
        again, services = self.execute(self.STAGES, config, handle)
        self.assertEqual([r.status for r in again.stage_results.values()], ["skipped", "skipped"])
        self.assertEqual(services.training.calls, 0)
        config5 = curve_config(self.tmp / "five", "benchmark.steps=5")
        handle5 = ExperimentStore(RunConfig(experiments_dir=self.tmp / "five" / "experiments")).create(ExperimentSpec(config5, None, "0", {}, "x"))
        self.execute(self.STAGES, config5, handle5)
        self.assertEqual(self.call(handle5, "benchmark")["n_val"], 5 * 2)


class RealCurveTests(unittest.TestCase):
    """Both commands on 60 synthetic clips and the stored tiny test model, 2 threads."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        write_smoke_data(SmokeConfig(n_samples=60), 42, cls.tmp / "smoke")
        sets = [
            f"data.source.path={(cls.tmp / 'smoke').as_posix()}", f"model.path={TINY.as_posix()}", f"run.experiments_dir={(cls.tmp / 'experiments').as_posix()}",
            'subsample.presets={"s8": {"size": {"samples": 8}}, "s16": {"size": {"samples": 16}}, "s32": {"size": {"samples": 32}}}', 'learning_curve.presets=["s8", "s16", "s32"]',
            "subsample.min_train_samples=1", "subsample.val_samples=6", "train.hyperparams.max_steps=6", "train.eval_every_steps=3", "train.save_every_steps=3",
        ]
        cls.args = ["--config", str(SMOKE_YAML), *[a for s in sets for a in ("--set", s)]]
        cls.curve = run_cli("learning-curve", *cls.args)
        cls.bench = run_cli("benchmark", *cls.args)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_real_learning_curve_trains_three_sizes_and_prints_the_table_and_a_reading(self) -> None:
        code, out, err = self.curve
        self.assertEqual(code, 0, err)
        s = json.loads(out)["stages"]["learning_curve"]["summary"]
        self.assertEqual(s["points"], 3)
        self.assertIn(s["trend"], ("improving", "flat", "worse", "unclear"))
        rows = json.loads(str(s["table"]))
        self.assertEqual([r["train_samples"] for r in rows], [8, 16, 32])
        self.assertEqual(s["val_files"], 6)  # the same fixed Val slice for every size
        self.assertIn("LEARNING CURVE", err)
        self.assertIn("READING:", err)
        root = self.tmp / "experiments" / "exp_0001"
        self.assertEqual(sorted(p.name for p in (root / "candidates").iterdir()), ["exp_0001_curve_s16", "exp_0001_curve_s32", "exp_0001_curve_s8"])
        self.assertEqual(sorted(p.name for p in (root / "reports").glob("*test*")), [])

    def test_the_real_benchmark_prints_the_measured_times_and_the_planned_run(self) -> None:
        code, out, err = self.bench
        self.assertEqual(code, 0, err)
        s = json.loads(out)["stages"]["benchmark"]["summary"]
        self.assertGreater(s["seconds_per_step"], 0)
        self.assertGreater(s["seconds_per_decode_clip"], 0)
        self.assertGreater(s["projected_run_seconds"], 0)
        self.assertIn("BENCHMARK on this machine", err)
        self.assertIn("TOTAL", err)
        root = self.tmp / "experiments" / "exp_0002"
        self.assertTrue((root / "reports" / "benchmark.txt").is_file())
        self.assertTrue((root / "candidates" / "exp_0002_benchmark").is_dir())  # kept: nothing is deleted
        self.assertEqual(sorted(p.name for p in (root / "reports").glob("*test*")), [])


if __name__ == "__main__":
    unittest.main()
