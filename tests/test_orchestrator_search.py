"""Unit tests of the `search` stage and command (stt.orchestrator, Part 1 of the Wave 2 wiring): the stage, the trial runner, the resume, the exit codes, the reports.

How to run (project root, virtual environment active):
    python -m unittest tests.test_orchestrator_search -v
Most tests use FAKE services (tests/search_stage_fakes.py: a synthetic Val loss, no model, no audio) with the real search service (grid, screening) and the real Orchestrator, in temporary
folders (fast; one test runs real worker processes). `RealSearchTests` run the real chain on synthetic clips and the stored tiny test model (about 40 seconds). The Test split is never
evaluated; the fake models record every split they are asked for and the tests check that the Test is not among them. Nothing is downloaded.
"""

from __future__ import annotations

import contextlib
import io
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType
from typing import Any

ROOT =Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))  # the fakes are imported by name, also by worker processes

from search_stage_fakes import SPLIT_SIZES, FakeTraining, fake_services, make_samples  # noqa: E402

from stt.config import AppConfig, load_config  # noqa: E402
from stt.modelloader import ModelConfig, create_loader  # noqa: E402
from stt.orchestrator import (  # noqa: E402
    InterruptFlag,
    NoopSleepGuard,
    Orchestrator,
    RunOutcome,
    RunPlan,
    SmokeConfig,
    StageResult,
    default_stage_registry,
    load_result,
    write_smoke_data,
)
from stt.orchestrator import cli  # noqa: E402
from stt.orchestrator.trials import SearchTrialRunner  # noqa: E402
from stt.orchestrator.types import ScreeningSet  # noqa: E402
from stt.orchestrator.wiring import build_services, exit_code_for, section_registry  # noqa: E402
from stt.runmanager import ExperimentHandle, ExperimentSpec, ExperimentStore, RunConfig  # noqa: E402
from stt.searcher import SearchConfigError, SearchError, SearchInterrupted, Trial  # noqa: E402

TINY = ROOT / "tests" / "assets" / "tiny_whisper"
SMOKE_YAML = ROOT / "configs" / "smoke.yaml"
GRID = '{"learning_rate": {"kind": "log_float", "low": 1e-5, "high": 1e-4, "points": 2}, "grad_accum_steps": {"kind": "categorical", "choices": [2, 4]}}'
GRID_COMBOS = [(1e-5, 2), (1e-5, 4), (1e-4, 2), (1e-4, 4)]  # learning rate, accumulation: the order of the grid


def make_config(*overrides: str) -> AppConfig:
    base = [
        f"data.source.path={TINY.as_posix()}", f"model.path={TINY.as_posix()}", "search.screening.preset=smoke", "search.n_trials=4", f"search.parameters={GRID}", "train.cpu_threads=4",
        "search.pruning.enabled=false",  # the tests that look at pruning switch it on; the others want every trial to finish
    ]
    return load_config(SMOKE_YAML, overrides=[*base, *overrides], registry=section_registry(), env_file=None, target_environ={})


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class SearchCase(unittest.TestCase):
    """A temporary experiments folder, a config with a 2 x 2 grid of 4 trials and fake services."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.store = ExperimentStore(RunConfig(experiments_dir=self.tmp / "experiments"))

    def experiment(self, config: AppConfig) -> ExperimentHandle:
        return self.store.create(ExperimentSpec(config, None, "0.0.0", {}, "2026-10-05T12:00:00"))

    def run_search(self, config: AppConfig, handle: ExperimentHandle, flag: Any = None, hook: Any = None, fail: frozenset[str] = frozenset()) -> tuple[RunOutcome, Any]:
        services = fake_services(config, hook, fail)
        outcome = Orchestrator(config, services, default_stage_registry(), flag or InterruptFlag(), NoopSleepGuard()).run(RunPlan(["prepare", "search"], handle))
        return outcome, services

    @staticmethod
    def call(handle: ExperimentHandle, trial: int) -> dict[str, Any]:
        return json.loads((handle.paths.metrics / f"trial_{trial:03d}" / "call.json").read_text(encoding="utf-8"))

    @staticmethod
    def trials(handle: ExperimentHandle) -> list[dict[str, Any]]:
        return json.loads((handle.paths.reports / "search_trials.json").read_text(encoding="utf-8"))["trials"]


class GridRunTests(SearchCase):
    def test_a_four_trial_grid_runs_and_writes_every_artifact(self) -> None:
        config = make_config()
        handle = self.experiment(config)
        outcome, services = self.run_search(config, handle)
        self.assertEqual((outcome.exit_code, outcome.status), (0, "completed"))
        summary = outcome.stage_results["search"].summary
        self.assertEqual((summary["strategy"], summary["trials_started"], summary["trials_complete"], summary["trials_pruned"], summary["trials_failed"], summary["trials_from_earlier_run"]), ("grid", 4, 4, 0, 0, 0))
        self.assertEqual((summary["stopped_reason"], summary["screening_train_samples"], summary["screening_val_samples"]), ("n_trials", 20, SPLIT_SIZES["val"]))
        self.assertEqual(len(json.loads(str(summary["finalist_trials"]))), 3)  # search.finalists.k
        for name in ("search_report.txt", "search_trials.json", "search_screening.txt", "stage_results/search.json"):
            self.assertTrue((handle.paths.reports / name).is_file(), name)
        self.assertEqual(len(json.loads((handle.paths.search / "study.json").read_text(encoding="utf-8"))["trials"]), 4)
        for i in range(4):
            folder = handle.paths.candidates / f"exp_0001_trial_{i:03d}"
            self.assertEqual([p.name for p in folder.glob("exp_*_step_*")], [f"exp_0001_trial_{i:03d}_step_00150"])
            call = self.call(handle, i)
            self.assertEqual((call["learning_rate"], call["grad_accum_steps"]), (GRID_COMBOS[i][0], GRID_COMBOS[i][1]))  # apply_params: the searched values reach the trainer
            self.assertEqual((call["threads"], call["resume"], call["name"]), (4, False, f"exp_0001_trial_{i:03d}"))  # trial.threads = total threads / parallel trials
        self.assertEqual(sorted(set(services.models.requested)), ["train", "val"])  # the Test split is never built, let alone scored
        self.assertEqual(load_result(handle, "search").summary["trials_complete"], 4)

    def test_the_prepare_stage_carries_the_data_and_split_fingerprints(self) -> None:
        config = make_config()
        outcome, _ = self.run_search(config, self.experiment(config))
        prepare = outcome.stage_results["prepare"].summary
        self.assertEqual((prepare["data_fingerprint"], prepare["split_fingerprint"]), ("d" * 64, "s" * 64))
        self.assertNotIn("subsample_fingerprint", prepare)  # the Subsampler is off for the experiment

    def test_the_report_names_the_trials_the_finalists_and_the_screening_data(self) -> None:
        config = make_config()
        handle = self.experiment(config)
        self.run_search(config, handle)
        text = (handle.paths.reports / "search_report.txt").read_text(encoding="utf-8")
        for part in ("search (grid, seed 42)", "4 trials (4 complete", "trial  status", "finalists (best 3", "20 Train and 10 Val samples"):
            self.assertIn(part, text)
        data = json.loads((handle.paths.reports / "search_trials.json").read_text(encoding="utf-8"))
        losses = {t["trial_id"]: t["val_loss"] for t in data["trials"]}
        self.assertEqual(data["finalists"], sorted(losses, key=lambda i: (losses[i], i))[:3])  # best Val loss first
        self.assertEqual((data["screening"]["n_train"], data["screening"]["n_val"], data["strategy"], data["stopped_reason"]), (20, 10, "grid", "n_trials"))
        screening = (handle.paths.reports / "search_screening.txt").read_text(encoding="utf-8")
        self.assertIn("subsample:", screening)

    def test_the_fixed_size_val_slice_reaches_the_trials(self) -> None:
        config = make_config("subsample.val_samples=3")
        handle = self.experiment(config)
        outcome, _ = self.run_search(config, handle)
        self.assertEqual(outcome.stage_results["search"].summary["screening_val_samples"], 3)
        call = self.call(handle, 0)
        self.assertEqual((call["n_train"], call["n_val"]), (20, 3))

    def test_the_screening_preset_comes_from_search_screening_preset(self) -> None:
        config = make_config("search.screening.preset=full")
        outcome, _ = self.run_search(config, self.experiment(config))
        self.assertEqual(outcome.stage_results["search"].summary["screening_train_samples"], SPLIT_SIZES["train"])

    def test_a_trial_that_the_pruner_stops_is_recorded_as_pruned_not_as_an_interrupt(self) -> None:
        FakeTraining.EVAL_STEPS = (50, 100, 150, 200, 250)  # the median rule needs four evaluations
        self.addCleanup(setattr, FakeTraining, "EVAL_STEPS", (50, 100, 150))
        config = make_config("search.pruning.enabled=true", "search.pruning.min_evals=1")
        handle = self.experiment(config)
        outcome, _ = self.run_search(config, handle)
        self.assertEqual((outcome.exit_code, outcome.status), (0, "completed"))
        statuses = [t["status"] for t in self.trials(handle)]
        self.assertIn("pruned", statuses)
        self.assertIn("complete", statuses)
        pruned = next(t for t in self.trials(handle) if t["status"] == "pruned")
        self.assertLess(pruned["steps"], 250)


class FailureTests(SearchCase):
    def test_a_failing_trial_is_recorded_and_the_search_goes_on(self) -> None:
        config = make_config()
        handle = self.experiment(config)
        outcome, _ = self.run_search(config, handle, fail=frozenset({"exp_0001_trial_001"}))
        summary = outcome.stage_results["search"].summary
        self.assertEqual((outcome.exit_code, summary["trials_complete"], summary["trials_failed"]), (0, 3, 1))
        failed = next(t for t in self.trials(handle) if t["status"] == "failed")
        self.assertIn("fake failure of exp_0001_trial_001", failed["error"])
        self.assertIn("fake failure", (handle.paths.reports / "search_report.txt").read_text(encoding="utf-8"))

    def test_when_no_trial_completes_the_exit_code_is_4_and_the_reports_are_still_written(self) -> None:
        config = make_config()
        handle = self.experiment(config)
        outcome, _ = self.run_search(config, handle, fail=frozenset(f"exp_0001_trial_{i:03d}" for i in range(4)))
        self.assertEqual((outcome.exit_code, outcome.status), (4, "failed"))
        self.assertIn("No trial completed", outcome.message or "")
        self.assertTrue((handle.paths.reports / "search_report.txt").is_file())
        self.assertTrue((handle.paths.search / "study.json").is_file())
        self.assertIsNone(load_result(handle, "search"))  # no stored result ...
        self.assertEqual(handle.stage_mark("search").status, "failed")  # type: ignore[union-attr]  # ... and the marker says failed, so a rerun does not skip it

    def test_a_search_config_error_has_exit_code_2(self) -> None:
        for override in ("search.strategy=nope", "search.parallel_trials=64"):
            with self.subTest(override=override):
                config = make_config(override)
                outcome, _ = self.run_search(config, self.experiment(config))
                self.assertEqual((outcome.exit_code, outcome.status), (2, "failed"))
                self.assertIn("search", (outcome.message or "").lower())

    def test_the_exit_code_mapping_of_the_search_errors(self) -> None:
        self.assertEqual((exit_code_for(SearchConfigError("x")), exit_code_for(SearchError("x"))), (2, 4))


class InterruptAndResumeTests(SearchCase):
    def test_an_interrupted_search_keeps_its_finished_trials_and_a_rerun_continues_with_the_same_grid(self) -> None:
        config = make_config()
        handle = self.experiment(config)
        flag = InterruptFlag()
        first, _ = self.run_search(config, handle, flag, hook=lambda call, step: flag.set() if (call, step) == (3, 100) else None)  # the third trial is stopped at its second evaluation
        self.assertEqual((first.exit_code, first.status), (130, "stopped"))
        self.assertIsNone(handle.stage_mark("search"))
        self.assertIsNotNone(handle.stage_mark("prepare"))
        self.assertEqual([t["trial_id"] for t in json.loads((handle.paths.search / "study.json").read_text(encoding="utf-8"))["trials"]], [0, 1])  # the interrupted trial is not recorded
        second, _ = self.run_search(config, handle)  # the same command again: a new flag, no hook
        self.assertEqual((second.exit_code, second.status), (0, "completed"))
        summary = second.stage_results["search"].summary
        self.assertEqual((summary["trials_started"], summary["trials_complete"], summary["trials_from_earlier_run"]), (4, 4, 2))
        self.assertEqual(second.stage_results["prepare"].status, "skipped")
        self.assertTrue(self.call(handle, 2)["resume"])  # the interrupted trial continued from its checkpoint
        self.assertFalse(self.call(handle, 3)["resume"])
        self.assertEqual([p.name for p in (handle.paths.candidates / "exp_0001_trial_002").glob("exp_*_step_*")], ["exp_0001_trial_002_step_00150"])
        # the same four parameter sets as an uninterrupted search
        other = self.experiment(config)
        self.run_search(config, other)
        self.assertEqual([t["params"] for t in self.trials(handle)], [t["params"] for t in self.trials(other)])

    def test_a_stop_before_a_trial_starts_raises_search_interrupted(self) -> None:
        config = make_config()
        handle = self.experiment(config)
        services = fake_services(config)
        runner = SearchTrialRunner(handle, ScreeningSet("train", "val", (), (), "r", "f"), services.search.services_provider, services, lambda: True)  # type: ignore[union-attr]
        with self.assertRaises(SearchInterrupted):
            runner(Trial(0, {"learning_rate": 1e-5}), lambda metrics: False)
        self.assertFalse((handle.paths.candidates / "exp_0001_trial_000").exists())

    def test_a_folder_that_cannot_be_resumed_is_renamed_never_deleted(self) -> None:
        config = make_config()
        handle = self.experiment(config)
        leftover = handle.paths.candidates / "exp_0001_trial_000"
        leftover.mkdir(parents=True)
        (leftover / "leftover.txt").write_text("keep me", encoding="utf-8")  # no checkpoint, no parameter file
        other = handle.paths.candidates / "exp_0001_trial_001"
        (other / "checkpoints" / "checkpoint-50").mkdir(parents=True)  # a checkpoint of an attempt with OTHER parameters
        handle.paths.search.mkdir(parents=True, exist_ok=True)
        (handle.paths.search / "trial_001.params.json").write_text(json.dumps({"learning_rate": 0.5, "grad_accum_steps": 1}), encoding="utf-8")
        outcome, _ = self.run_search(config, handle)
        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual((handle.paths.candidates / "exp_0001_trial_000_stale_1" / "leftover.txt").read_text(encoding="utf-8"), "keep me")
        self.assertTrue((handle.paths.candidates / "exp_0001_trial_001_stale_1" / "checkpoints" / "checkpoint-50").is_dir())
        self.assertEqual((self.call(handle, 0)["resume"], self.call(handle, 1)["resume"]), (False, False))  # both started fresh
        text = (handle.paths.reports / "search_report.txt").read_text(encoding="utf-8")
        self.assertIn("exp_0001_trial_000_stale_1", text)
        self.assertIn("nothing deletes them by itself", text)


class ParallelTests(SearchCase):
    def test_two_parallel_trials_run_in_worker_processes_with_their_share_of_the_threads(self) -> None:
        config = make_config("search.parallel_trials=2")
        handle = self.experiment(config)
        outcome, _ = self.run_search(config, handle)
        self.assertEqual((outcome.exit_code, outcome.status), (0, "completed"))
        summary = outcome.stage_results["search"].summary
        self.assertEqual((summary["trials_started"], summary["trials_complete"]), (4, 4))
        for i in range(4):
            call = self.call(handle, i)
            self.assertEqual(call["threads"], 2)  # 4 threads in all / 2 parallel trials
            self.assertTrue((handle.paths.logs / f"trial_{i:03d}.log").is_file())  # a worker has its own log file
        self.assertEqual(sorted(t["trial_id"] for t in self.trials(handle)), [0, 1, 2, 3])
        self.assertIn("parallel trials 2", (handle.paths.reports / "search_report.txt").read_text(encoding="utf-8"))
        self.assertIn("can vary", (handle.paths.reports / "search_report.txt").read_text(encoding="utf-8"))

    def test_the_runner_can_be_sent_to_a_worker_without_its_services_and_stop_flag(self) -> None:
        config = make_config()
        handle = self.experiment(config)
        services = fake_services(config)
        train, val = tuple(make_samples("train", 3)), tuple(make_samples("val", 2))  # their `extra` is a read-only mappingproxy, as in the real data
        runner = SearchTrialRunner(handle, ScreeningSet("train", "val", train, val, "r", "f"), services.search.services_provider, services, InterruptFlag())  # type: ignore[union-attr]
        again = pickle.loads(pickle.dumps(runner))
        self.assertEqual((again._services, again._stop), (None, None))
        self.assertEqual(again._experiment.experiment_id, "exp_0001")
        self.assertEqual([dict(n.sample.extra) for n in again._screening.train], [dict(n.sample.extra) for n in train])  # the same values, plain dicts
        self.assertEqual((again._screening.val[0].sample.id, len(again._screening.train)), ("val000", 3))
        self.assertIsInstance(runner._screening.train[0].sample.extra, MappingProxyType)  # the parent's own samples are untouched


class CommandLineTests(unittest.TestCase):
    def test_the_search_command_is_in_the_parser_with_an_example_and_runs_prepare_then_search(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            cli.main(["search", "--help"])
        self.assertIn("python -m stt search", out.getvalue())
        self.assertEqual(cli.PLANS["search"], ["prepare", "search"])

    def test_the_search_section_is_registered_with_the_documented_defaults(self) -> None:
        config = make_config()
        self.assertIn("search", config.sections)
        from stt.searcher import SearchConfig

        defaults = SearchConfig()
        self.assertEqual((defaults.strategy, defaults.n_trials, defaults.finalists.k, defaults.screening.preset, defaults.parallel_trials), ("grid", 6, 3, "tiny", 1))

    def test_services_built_from_a_config_have_the_search_service_and_the_picklable_provider(self) -> None:
        services = build_services(make_config())
        self.assertIsNotNone(services.search)
        self.assertEqual(pickle.loads(pickle.dumps(services.search.services_provider)).__name__, "services_from_experiment_config")  # type: ignore[union-attr]


class RealSearchTests(unittest.TestCase):
    """The real chain on synthetic clips and the stored tiny test model: a 4-trial grid, then an interrupted search that is resumed."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        write_smoke_data(SmokeConfig(n_samples=30), 42, cls.tmp / "smoke")
        cls.overrides = [
            f"data.source.path={(cls.tmp / 'smoke').as_posix()}",
            f"model.path={TINY.as_posix()}",
            f"run.experiments_dir={(cls.tmp / 'experiments').as_posix()}",
            "search.n_trials=4",
            'search.parameters={"learning_rate": {"kind": "log_float", "low": 1e-5, "high": 1e-4, "points": 2}, "grad_accum_steps": {"kind": "categorical", "choices": [1, 2]}}',
            "search.screening.preset=smoke",
            "subsample.min_train_samples=1",
            "subsample.val_samples=3",
            "train.hyperparams.max_steps=3",
        ]
        args = ["--config", str(SMOKE_YAML), *[a for o in cls.overrides for a in ("--set", o)]]
        cls.code, cls.out, cls.err = run_cli("search", *args)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_real_four_trial_grid_runs_and_leaves_loadable_models_and_reports(self) -> None:
        self.assertEqual(self.code, 0, self.err)
        s = json.loads(self.out)
        search = s["stages"]["search"]["summary"]
        self.assertEqual((s["status"], search["trials_started"], search["trials_complete"], search["screening_val_samples"]), ("completed", 4, 4, 3))
        self.assertIn("best_val_loss", s["key_numbers"])  # the numbers a human looks at first
        root = self.tmp / "experiments" / "exp_0001"
        for name in ("search_report.txt", "search_trials.json", "search_screening.txt"):
            self.assertTrue((root / "reports" / name).is_file(), name)
        self.assertEqual(sorted(p.name for p in (root / "candidates").iterdir()), [f"exp_0001_trial_{i:03d}" for i in range(4)])
        model = next((root / "candidates" / "exp_0001_trial_000").glob("exp_0001_trial_000_step_*"))
        self.assertEqual(create_loader(ModelConfig(path=model)).load().info.name, model.name)
        self.assertTrue(all((root / "metrics" / f"trial_{i:03d}" / "metrics.csv").is_file() for i in range(4)))
        self.assertFalse(list((root / "reports").glob("*test*")))  # nothing about the Test split
        prepare = s["stages"]["prepare"]["summary"]
        self.assertEqual((len(prepare["data_fingerprint"]), len(prepare["split_fingerprint"])), (64, 64))

    def test_the_prepare_stage_adds_the_subsample_fingerprint_only_when_the_subsampler_is_on(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = [o for o in self.overrides if not o.startswith("run.experiments_dir")] + [f"run.experiments_dir={(Path(tmp.name) / 'experiments').as_posix()}"]
        args = ["--config", str(SMOKE_YAML), *[a for o in base for a in ("--set", o)]]
        code, out, err = run_cli("prepare", *args)
        self.assertEqual(code, 0, err)
        self.assertNotIn("subsample_fingerprint", json.loads(out)["stages"]["prepare"]["summary"])
        code, out, err = run_cli("prepare", *args, "--set", "subsample.enabled=true", "--set", "subsample.preset=smoke")
        self.assertEqual(code, 0, err)
        summary = json.loads(out)["stages"]["prepare"]["summary"]
        self.assertEqual(len(summary["subsample_fingerprint"]), 64)
        self.assertNotEqual(summary["subsample_fingerprint"], summary["split_fingerprint"])

    def test_a_real_search_stopped_in_its_second_trial_is_resumed_by_the_same_command(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        overrides = [o for o in self.overrides if not o.startswith("run.experiments_dir")] + [f"run.experiments_dir={(Path(tmp.name) / 'experiments').as_posix()}", "train.hyperparams.max_steps=6", "train.eval_every_steps=3", "train.save_every_steps=3"]
        config = load_config(SMOKE_YAML, overrides=overrides, registry=section_registry(), env_file=None, target_environ={})
        handle = ExperimentStore(RunConfig(experiments_dir=Path(tmp.name) / "experiments")).create(ExperimentSpec(config, None, "0", {}, "x"))
        metrics = handle.paths.metrics / "trial_001" / "metrics.csv"
        flag = lambda: metrics.is_file() and len(metrics.read_text(encoding="utf-8").splitlines()) >= 3  # noqa: E731  (the second trial wrote two records: stop)
        registry = default_stage_registry()
        first = Orchestrator(config, build_services(config), registry, flag, NoopSleepGuard()).run(RunPlan(["prepare", "search"], handle))
        self.assertEqual((first.exit_code, first.status), (130, "stopped"))
        self.assertIsNone(handle.stage_mark("search"))
        self.assertEqual(len(json.loads((handle.paths.search / "study.json").read_text(encoding="utf-8"))["trials"]), 1)
        second = Orchestrator(config, build_services(config), registry, InterruptFlag(), NoopSleepGuard()).run(RunPlan(["prepare", "search"], handle))
        self.assertEqual((second.exit_code, second.status), (0, "completed"))
        summary: StageResult = second.stage_results["search"]
        self.assertEqual((summary.summary["trials_started"], summary.summary["trials_complete"], summary.summary["trials_from_earlier_run"]), (4, 4, 1))
        steps = [int(r.split(",")[0]) for r in metrics.read_text(encoding="utf-8").splitlines()[1:]]
        self.assertEqual(steps, sorted(steps))  # the metrics of the resumed trial continue without a repeat of the stopped part
        self.assertEqual([p.name for p in (handle.paths.candidates / "exp_0001_trial_001").glob("exp_*_step_*")], ["exp_0001_trial_001_step_00006"])


if __name__ == "__main__":
    unittest.main()
