"""Unit tests of the minimal Orchestrator and command line (stt.orchestrator): the acceptance tests of the C13 brief that apply to this version, offline.

How to run (project root, virtual environment active):
    python -m unittest tests.test_orchestrator -v
Most tests use fake stages and fake services in temporary folders (fast). The smoke tests run the real chain on synthetic clips and the stored tiny
test model (tests/assets/tiny_whisper), about 15 seconds each. The Test split is never evaluated except to check that it is REFUSED. Nothing is downloaded.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import logging
import signal
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stt.config import AppConfig, ConfigError, SectionRegistry, load_config  # noqa: E402
from stt.datasource import AudioSample, DataSourceError  # noqa: E402
from stt.evaluator import EvaluatorError  # noqa: E402
from stt.modelloader import ModelConfig, create_loader  # noqa: E402
from stt.orchestrator import (  # noqa: E402
    EvaluateStage,
    InterruptFlag,
    NoopSleepGuard,
    Orchestrator,
    OrchestratorError,
    PipelineConfig,
    PreparedData,
    RunPlan,
    Services,
    SmokeConfig,
    StageContext,
    StageInterrupted,
    StageRegistry,
    StageResult,
    WindowsSleepGuard,
    code_version,
    default_sleep_guard,
    default_stage_registry,
    fingerprint,
    load_result,
    manifest_hash,
    write_smoke_data,
)
from stt.orchestrator import check as check_module  # noqa: E402
from stt.orchestrator import cli  # noqa: E402
from stt.orchestrator.sleepguard import ES_CONTINUOUS, ES_SYSTEM_REQUIRED  # noqa: E402
from stt.orchestrator.stages import _items  # noqa: E402
from stt.orchestrator.wiring import build_services, exit_code_for, section_registry  # noqa: E402
from stt.runmanager import ExperimentSpec, ExperimentStore, RunConfig, RunManagerError  # noqa: E402
from stt.samplefilter import FilterError  # noqa: E402
from stt.splitter import SplitError  # noqa: E402
from stt.textnormalizer import NormalizedSample  # noqa: E402
from stt.trainercore import TrainerConfigError, TrainerCoreError  # noqa: E402

TINY = ROOT / "tests" / "assets" / "tiny_whisper"
SMOKE_YAML = ROOT / "configs" / "smoke.yaml"


def run_cli(*argv: str) -> tuple[int, str, str]:
    """Run the command line in-process; returns (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class FakeStage:
    """A stage with a counter, a configurable config slice and a configurable behavior."""

    def __init__(self, name: str, depends_on: tuple[str, ...] = (), calls: list[str] | None = None, behavior: Any = None, slice_key: str = "train") -> None:
        self.name = name
        self.depends_on = depends_on
        self.calls = calls if calls is not None else []
        self.behavior = behavior
        self.slice_key = slice_key

    def config_slice(self, config: AppConfig) -> Mapping[str, Any]:
        return {"x": dict(config.sections[self.slice_key].model_dump(mode="json"))}

    def run(self, ctx: StageContext) -> StageResult:
        self.calls.append(self.name)
        if self.behavior is not None:
            self.behavior(ctx)
        return StageResult("ok", {"value": len(self.calls)}, [])


def fake_services() -> Services:
    unused: Any = None
    return Services(unused, unused, unused, unused, exit_code_for)


def make_config(*overrides: str) -> AppConfig:
    base = [f"data.source.path={TINY.as_posix()}", f"model.path={TINY.as_posix()}"]  # existing folders: only the config is used, no data is read
    return load_config(SMOKE_YAML, overrides=[*base, *overrides], registry=section_registry(), env_file=None, target_environ={})


class OrchestratorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.config = make_config()
        self.store = ExperimentStore(RunConfig(experiments_dir=self.tmp / "experiments"))
        self.handle = self.store.create(ExperimentSpec(self.config, None, "0.0.0", {}, "2026-10-05T12:00:00"))
        self.calls: list[str] = []

    def registry(self, *stages: FakeStage) -> StageRegistry:
        return StageRegistry({s.name: (lambda s=s: s) for s in stages})  # type: ignore[misc]

    def orchestrator(self, *stages: FakeStage, config: AppConfig | None = None, flag: Any = None) -> Orchestrator:
        return Orchestrator(config or self.config, fake_services(), self.registry(*stages), flag or InterruptFlag(), NoopSleepGuard())

    def plan(self, *names: str, handle: Any = None) -> RunPlan:
        return RunPlan(list(names), handle or self.handle)


class RunTests(OrchestratorTestCase):
    def test_1_the_stages_run_in_order_and_the_outcome_lists_every_result(self) -> None:
        stages = [FakeStage("a", calls=self.calls), FakeStage("b", ("a",), self.calls), FakeStage("c", ("b",), self.calls)]
        outcome = self.orchestrator(*stages).run(self.plan("a", "b", "c"))
        self.assertEqual((self.calls, outcome.exit_code, outcome.status, outcome.verdict), (["a", "b", "c"], 0, "completed", "n/a"))
        self.assertEqual([outcome.stage_results[n].status for n in "abc"], ["ok"] * 3)
        status = self.handle.status()
        self.assertEqual(status.status, "completed")
        self.assertEqual({n: m.status for n, m in status.stages.items()}, {"a": "ok", "b": "ok", "c": "ok"})
        self.assertEqual(load_result(self.handle, "b").summary["value"], 2)  # type: ignore[union-attr]

    def test_2_a_rerun_skips_finished_stages_and_a_changed_slice_reruns_it_and_the_later_stages(self) -> None:
        a, b, c = FakeStage("a", calls=self.calls, slice_key="split"), FakeStage("b", ("a",), self.calls, slice_key="train"), FakeStage("c", ("b",), self.calls, slice_key="eval")
        orchestrator = self.orchestrator(a, b, c)
        orchestrator.run(self.plan("a", "b", "c"))
        self.calls.clear()
        again = orchestrator.run(self.plan("a", "b", "c"))
        self.assertEqual(self.calls, [])  # nothing ran
        self.assertEqual([again.stage_results[n].status for n in "abc"], ["skipped"] * 3)
        self.assertEqual(again.stage_results["b"].summary["value"], 2)  # the stored result of the earlier run is returned
        changed = make_config("train.hyperparams.max_steps=7")
        self.orchestrator(a, b, c, config=changed).run(self.plan("a", "b", "c"))
        self.assertEqual(self.calls, ["b", "c"])  # b's slice changed: b runs again and c after it; a is skipped
        self.calls.clear()
        self.orchestrator(a, b, c, config=make_config("split.mode=hash_group")).run(self.plan("a", "b", "c"))
        self.assertEqual(self.calls, ["a", "b", "c"])  # the first stage changed: everything after it is invalidated
        self.calls.clear()
        Orchestrator(self.config, fake_services(), self.registry(a, b, c), InterruptFlag(), NoopSleepGuard(), skip_completed=False).run(self.plan("a", "b", "c"))
        self.assertEqual(self.calls, ["a", "b", "c"])

    def test_3_a_failing_stage_stops_the_run_with_its_exit_code_and_names_the_stage(self) -> None:
        def explode(ctx: StageContext) -> None:
            raise TrainerCoreError("the loss became NaN at step 3")

        stages = [FakeStage("a", calls=self.calls), FakeStage("b", ("a",), self.calls, behavior=explode), FakeStage("c", ("b",), self.calls)]
        outcome = self.orchestrator(*stages).run(self.plan("a", "b", "c"))
        self.assertEqual((outcome.exit_code, outcome.status, self.calls), (4, "failed", ["a", "b"]))
        self.assertIn("stage b failed", outcome.message or "")
        self.assertEqual((outcome.stage_results["b"].status, outcome.stage_results["c"].status), ("failed", "skipped"))
        self.assertIn("not run because stage b failed", outcome.stage_results["c"].error or "")
        status = self.handle.status()
        self.assertEqual((status.status, status.stages["b"].status), ("failed", "failed"))
        self.assertIn("NaN at step 3", status.reason or "")
        self.assertIsNone(self.handle.stage_mark("c"))

    def test_4_each_exit_code_comes_from_its_failure(self) -> None:
        cases = [
            (2, ConfigError("bad key")),
            (2, TrainerConfigError("unknown policy")),
            (2, RunManagerError("locked")),
            (2, OrchestratorError("refused", 2)),
            (3, DataSourceError("column missing")),
            (3, SplitError("gate")),
            (3, FilterError("gate")),
            (4, TrainerCoreError("nan")),
            (4, EvaluatorError("empty")),
            (1, ValueError("a bug")),
        ]
        for code, error in cases:
            with self.subTest(error=type(error).__name__):
                self.assertEqual(exit_code_for(error), code)
                handle = self.store.create(ExperimentSpec(self.config, None, "0", {}, "x"))

                def raise_it(ctx: StageContext, e: Exception = error) -> None:
                    raise e

                outcome = self.orchestrator(FakeStage("s", behavior=raise_it)).run(self.plan("s", handle=handle))
                self.assertEqual(outcome.exit_code, code)
        self.assertEqual(130, self.stopped_outcome().exit_code)  # interrupt
        # exit code 5 (export or verification) belongs to the export stage of a later wave and is not produced here.

    def stopped_outcome(self) -> Any:
        def stop(ctx: StageContext) -> None:
            raise StageInterrupted("stop")

        return self.orchestrator(FakeStage("s", behavior=stop)).run(self.plan("s", handle=self.store.create(ExperimentSpec(self.config, None, "0", {}, "x"))))

    def test_5_an_interrupt_sets_status_stopped_writes_no_marker_and_a_rerun_runs_the_stage_again(self) -> None:
        state = {"interrupt": True}

        def maybe_stop(ctx: StageContext) -> None:
            if state["interrupt"]:
                raise StageInterrupted("training stopped on request at step 4")

        stages = [FakeStage("a", calls=self.calls), FakeStage("b", ("a",), self.calls, behavior=maybe_stop)]
        outcome = self.orchestrator(*stages).run(self.plan("a", "b"))
        self.assertEqual((outcome.exit_code, outcome.status), (130, "stopped"))
        status = self.handle.status()
        self.assertEqual(status.status, "stopped")
        self.assertIn("interrupted during stage b", status.reason or "")
        self.assertIsNotNone(self.handle.stage_mark("a"))
        self.assertIsNone(self.handle.stage_mark("b"))
        state["interrupt"] = False
        self.calls.clear()
        again = self.orchestrator(*stages).run(self.plan("a", "b"))
        self.assertEqual((again.exit_code, again.status, self.calls), (0, "completed", ["b"]))  # a is skipped, b runs again

    def test_a_stop_flag_set_between_stages_stops_the_run(self) -> None:
        flag = InterruptFlag()
        a = FakeStage("a", calls=self.calls, behavior=lambda ctx: flag.set())
        b = FakeStage("b", ("a",), self.calls)
        outcome = self.orchestrator(a, b, flag=flag).run(self.plan("a", "b"))
        self.assertEqual((outcome.exit_code, self.calls, self.handle.status().status), (130, ["a"], "stopped"))
        self.assertIsNotNone(self.handle.stage_mark("a"))  # a finished and keeps its marker

    def test_an_unknown_stage_fails_before_anything_runs_and_lists_the_names(self) -> None:
        with self.assertRaises(OrchestratorError) as ctx:
            self.orchestrator(FakeStage("a", calls=self.calls)).run(self.plan("a", "export"))
        self.assertIn("export", str(ctx.exception))
        self.assertIn("a", str(ctx.exception))
        self.assertEqual((self.calls, self.handle.status().status), ([], "created"))
        with self.assertRaises(OrchestratorError) as ctx2:
            default_stage_registry().create("export")  # built in a later wave
        self.assertIn("benchmark, evaluate, finalists, learning_curve, prepare, search, select_and_register, train", str(ctx2.exception))

    def test_a_locked_experiment_fails_before_a_stage_runs(self) -> None:
        stage = FakeStage("a", calls=self.calls)
        with self.handle.lock():
            with self.assertRaises(RunManagerError):
                self.orchestrator(stage).run(self.plan("a"))
        self.assertEqual(self.calls, [])

    def test_the_prepare_slice_ignores_a_disabled_subsampler_so_older_experiments_keep_their_fingerprint(self) -> None:
        from stt.orchestrator import PrepareStage

        self.assertNotIn("subsample", PrepareStage().config_slice(self.config))
        on = make_config("subsample.enabled=true", 'subsample.size={"samples": 5}')
        self.assertIn("subsample", PrepareStage().config_slice(on))
        self.assertNotEqual(fingerprint("prepare", PrepareStage().config_slice(self.config), []), fingerprint("prepare", PrepareStage().config_slice(on), []))

    def test_fingerprints_depend_on_the_slice_and_the_dependencies(self) -> None:
        self.assertEqual(fingerprint("a", {"x": 1}, []), fingerprint("a", {"x": 1}, []))
        self.assertNotEqual(fingerprint("a", {"x": 1}, []), fingerprint("a", {"x": 2}, []))
        self.assertNotEqual(fingerprint("a", {"x": 1}, ["p"]), fingerprint("a", {"x": 1}, ["q"]))
        self.assertNotEqual(fingerprint("a", {"x": 1}, []), fingerprint("b", {"x": 1}, []))


class TestOnceTests(OrchestratorTestCase):
    def context(self, **options: str) -> StageContext:
        return StageContext(self.config, self.handle, fake_services(), InterruptFlag(), {}, options)

    def test_the_evaluate_stage_refuses_the_test_split_and_says_so(self) -> None:
        with self.assertRaises(OrchestratorError) as ctx:
            EvaluateStage().run(self.context(split="test"))
        self.assertEqual(ctx.exception.exit_code, 2)
        self.assertIn("refuses the Test split", str(ctx.exception))
        self.assertIn("ONCE", str(ctx.exception))
        custom = make_config("pipeline.test_split=holdout")
        with self.assertRaises(OrchestratorError):
            EvaluateStage().run(StageContext(custom, self.handle, fake_services(), InterruptFlag(), {}, {"split": "holdout"}))

    def test_an_unknown_split_fails_listing_the_known_ones(self) -> None:
        prepared = PreparedData({"train": [], "val": []}, 1, str, {}, {})
        services = fake_services()
        services = Services(mock.Mock(prepare=lambda out: prepared), None, None, None, exit_code_for)  # type: ignore[arg-type]
        with self.assertRaises(OrchestratorError) as ctx:
            EvaluateStage().run(StageContext(self.config, self.handle, services, InterruptFlag(), {}, {"split": "dev"}))
        self.assertIn("train, val", str(ctx.exception))

    def test_the_per_file_attributes_include_the_duration_for_the_token_cap(self) -> None:
        sample = AudioSample("a", Path("a.wav"), "raw", "g1", 2.5, None, {"meta_quality_score": 0.3, "meta_note": "text", "meta_golden": True})
        normalized = NormalizedSample(sample, "תרגום טוב", "תרגום טוב", None, ("had_digits",))
        prepared = PreparedData({"val": [normalized, NormalizedSample(AudioSample("b", Path("b.wav"), "r", "g", 1.0, None, {}), "x", "x", None, ())]}, 1, str, {}, {})
        items = _items(prepared, "val", frozenset({"b"}), lambda text: len(text.split()))
        self.assertEqual([i.sample_id for i in items], ["a"])
        self.assertEqual(dict(items[0].attributes), {"duration": 2.5, "ref_words": 2, "label_tokens": 2, "extra_meta_quality_score": 0.3, "extra_meta_golden": True, "flag_had_digits": True})
        self.assertEqual((items[0].group_id, items[0].reference, items[0].audio_path), ("g1", "תרגום טוב", Path("a.wav")))


class GuardAndFlagTests(unittest.TestCase):
    def test_6_the_windows_guard_calls_the_system_function_on_entry_and_restores_on_exit(self) -> None:
        calls: list[int] = []
        guard = WindowsSleepGuard(lambda flags: calls.append(flags) or 1)
        with guard:
            self.assertEqual(calls, [ES_CONTINUOUS | ES_SYSTEM_REQUIRED])
        self.assertEqual(calls, [ES_CONTINUOUS | ES_SYSTEM_REQUIRED, ES_CONTINUOUS])
        with self.assertRaises(ValueError):
            with guard:
                raise ValueError("the run failed")
        self.assertEqual(calls[-1], ES_CONTINUOUS)  # restored also when the run raises
        with self.assertLogs("stt.orchestrator", level="WARNING"):
            with WindowsSleepGuard(lambda flags: 0):
                pass

    def test_the_noop_guard_and_the_default_choice(self) -> None:
        with NoopSleepGuard():
            pass
        self.assertIsInstance(default_sleep_guard(False), NoopSleepGuard)
        with mock.patch("sys.platform", "linux"), self.assertLogs("stt.orchestrator", level="INFO") as logs:
            self.assertIsInstance(default_sleep_guard(True), NoopSleepGuard)
        self.assertIn("no-op guard", logs.output[0])
        with mock.patch("sys.platform", "win32"):
            self.assertIsInstance(default_sleep_guard(True), WindowsSleepGuard)

    def test_the_interrupt_flag_first_press_sets_it_and_the_second_raises(self) -> None:
        flag = InterruptFlag()
        self.assertFalse(flag())
        with self.assertLogs("stt.orchestrator", level="WARNING"):
            flag._handler(signal.SIGINT, None)
        self.assertTrue(flag())
        with self.assertRaises(KeyboardInterrupt):
            flag._handler(signal.SIGINT, None)

    def test_the_signal_handlers_are_installed_and_restored(self) -> None:
        before = signal.getsignal(signal.SIGINT)
        flag = InterruptFlag()
        with flag.installed():
            self.assertNotEqual(signal.getsignal(signal.SIGINT), before)
        self.assertEqual(signal.getsignal(signal.SIGINT), before)


class SmokeTests(unittest.TestCase):
    """The real chain on synthetic data and the tiny test model."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.experiments = cls.tmp / "experiments"
        cls.args = [
            "--config", str(SMOKE_YAML),
            "--set", f"run.experiments_dir={cls.experiments.as_posix()}",
            "--set", f"smoke.work_dir={(cls.tmp / 'smoke').as_posix()}",
            "--set", f"smoke.model_dir={TINY.as_posix()}",
        ]
        with mock.patch("socket.socket.connect", side_effect=AssertionError("network used")):
            cls.code, cls.out, cls.err = run_cli("smoke", *cls.args)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def summary(self) -> dict[str, Any]:
        return json.loads(self.out)  # the whole stdout is ONE JSON document

    def test_7_smoke_runs_the_whole_chain_and_leaves_every_artifact(self) -> None:
        self.assertEqual(self.code, 0, self.err)
        s = self.summary()
        self.assertEqual((s["experiment_id"], s["status"], s["exit_code"], s["verdict"]), ("exp_0001", "completed", 0, "n/a"))
        root = self.experiments / "exp_0001"
        for name in ("config_full.yaml", "config_diff.yaml", "run_info.json", "status.json", "logs/run.log", "metrics/trial_001/metrics.csv"):
            self.assertTrue((root / name).is_file(), name)
        for name in ("comparison.json", "evaluation_baseline.json", "evaluation_trained.json", "predictions_baseline.csv", "predictions_trained.csv", "review_baseline.txt", "review_trained.txt", "load_report.txt", "split_report.txt", "text_report.txt", "filter_report.txt"):
            self.assertTrue((root / "reports" / name).is_file(), name)
        self.assertEqual(sorted(s["stages"]), ["evaluate", "prepare", "train"])
        self.assertTrue(all(r["status"] == "ok" for r in s["stages"].values()))
        self.assertEqual(s["stages"]["prepare"]["summary"]["samples"], 12)
        self.assertEqual((s["stages"]["train"]["summary"]["stopped_reason"], s["stages"]["train"]["summary"]["steps"]), ("max_steps", 3))
        self.assertIn("trained_wer", s["key_numbers"])
        self.assertTrue(Path(s["paths"]["config"]).name == "config_full.yaml")
        self.assertEqual(json.loads((root / "status.json").read_text(encoding="utf-8"))["status"], "completed")

    def test_the_model_folder_is_named_after_the_experiment_trial_and_step_and_loads(self) -> None:
        model = self.experiments / "exp_0001" / "candidates" / "exp_0001_trial_001" / "exp_0001_trial_001_step_00003"
        loaded = create_loader(ModelConfig(path=model)).load()
        self.assertEqual((loaded.info.name, loaded.info.source_config_ref), ("exp_0001_trial_001_step_00003", "exp_0001/config_full.yaml"))
        self.assertEqual(self.summary()["stages"]["train"]["summary"]["model_dir"].replace("\\", "/").split("/")[-1], "exp_0001_trial_001_step_00003")

    def test_the_smoke_run_checks_the_test_once_rule(self) -> None:
        s = self.summary()
        self.assertEqual(s["test_once_rule"], "refused")
        self.assertIn("refuses the Test split", s["test_once_message"])
        self.assertIn("not evaluated", s["key_numbers"]["test_split"])
        self.assertIn("Test-once rule checked", self.err)

    def test_the_run_prints_where_to_look_and_the_commands(self) -> None:
        for part in ("The experiment is in:", "python -m stt.runmanager list --dir", "python -m stt.runmanager show exp_0001 --dir", "python -m stt check"):
            self.assertIn(part, self.err)
        self.assertEqual(self.out.strip()[0], "{")  # stdout holds only the JSON summary

    def test_the_log_file_has_the_stage_lines_and_the_config_is_read_only(self) -> None:
        log = (self.experiments / "exp_0001" / "logs" / "run.log").read_text(encoding="utf-8")
        for part in ("Stage prepare starts", "Stage train finished", "Evaluating the baseline model"):
            self.assertIn(part, log)
        with self.assertRaises(PermissionError):
            (self.experiments / "exp_0001" / "config_full.yaml").write_text("x", encoding="utf-8")

    def test_a_rerun_of_the_same_experiment_skips_every_stage(self) -> None:
        code, out, err = run_cli("smoke", "--experiment", "exp_0001", *self.args)
        self.assertEqual(code, 0, err)
        s = json.loads(out)
        self.assertEqual([r["status"] for r in s["stages"].values()], ["skipped"] * 3)
        self.assertEqual(s["stages"]["train"]["summary"]["steps"], 3)  # the earlier result is printed
        self.assertEqual(len(list(self.experiments.glob("exp_*"))), 1)

    def test_a_changed_config_cannot_resume_an_experiment_but_a_new_one_extends_it(self) -> None:
        code, out, err = run_cli("smoke", "--experiment", "exp_0001", "--set", "train.hyperparams.max_steps=5", *self.args)
        # the smoke settings put max_steps from smoke.steps; a user --set after them wins and differs from the stored config
        self.assertEqual(code, 2)
        self.assertIn("Cannot resume exp_0001", err)
        self.assertEqual(out, "")

    def test_the_trial_resumes_after_an_interrupt_and_the_orchestrator_sets_the_resume_itself(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        work = Path(tmp.name)
        write_smoke_data(SmokeConfig(n_samples=12), 42, work / "smoke")
        config = load_config(
            SMOKE_YAML,
            overrides=[f"data.source.path={(work / 'smoke').as_posix()}", f"model.path={TINY.as_posix()}", "train.hyperparams.max_steps=6", "train.save_every_steps=100", "train.eval_every_steps=100"],
            registry=section_registry(),
            env_file=None,
            target_environ={},
        )
        store = ExperimentStore(RunConfig(experiments_dir=work / "experiments"))
        handle = store.create(ExperimentSpec(config, None, "0", {}, "x"))
        metrics = handle.paths.metrics / "trial_001" / "metrics.csv"
        flag = lambda: metrics.is_file() and len(metrics.read_text(encoding="utf-8").splitlines()) >= 3  # noqa: E731  (two records written: stop)
        stages = default_stage_registry()
        first = Orchestrator(config, build_services(config), stages, flag, NoopSleepGuard()).run(RunPlan(["prepare", "train"], handle))
        self.assertEqual((first.exit_code, first.status), (130, "stopped"))
        self.assertIsNone(handle.stage_mark("train"))
        self.assertIsNotNone(handle.stage_mark("prepare"))
        second = Orchestrator(config, build_services(config), stages, InterruptFlag(), NoopSleepGuard()).run(RunPlan(["prepare", "train"], handle))
        self.assertEqual((second.exit_code, second.status), (0, "completed"))
        summary = second.stage_results["train"].summary
        self.assertEqual((summary["resumed"], summary["trial"], summary["steps"], summary["stopped_reason"]), (1, 1, 6, "max_steps"))
        rows = metrics.read_text(encoding="utf-8").splitlines()[1:]
        steps = [int(r.split(",")[0]) for r in rows]
        self.assertEqual(steps, sorted(steps))  # the metrics file continues without duplicates of the stopped part
        self.assertEqual(sorted(p.name for p in handle.paths.candidates.iterdir()), ["exp_0001_trial_001"])
        self.assertEqual([p.name for p in (handle.paths.candidates / "exp_0001_trial_001").glob("exp_*_step_*")], ["exp_0001_trial_001_step_00006"])


class CommandLineTests(unittest.TestCase):
    def test_13_every_command_help_shows_an_example(self) -> None:
        for command in ("prepare", "train", "evaluate", "smoke", "check"):
            with self.subTest(command=command):
                out = io.StringIO()
                with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as ctx:
                    cli.main([command, "--help"])
                self.assertEqual(ctx.exception.code, 0)
                self.assertIn(f"python -m stt {command}", out.getvalue())
                self.assertIn("Example:", out.getvalue())

    def test_an_unknown_command_is_refused_by_the_parser(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as ctx:
            cli.main(["deploy"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("invalid choice", err.getvalue())

    def test_a_missing_smoke_model_names_the_script_that_creates_it(self) -> None:
        code, out, err = run_cli("smoke", "--config", str(SMOKE_YAML), "--set", "smoke.model_dir=C:/nowhere/tiny")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("tools/make_test_model.py", err)

    def test_a_missing_parent_fails_listing_the_experiments_and_a_bad_key_is_a_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            write_smoke_data(SmokeConfig(n_samples=12), 42, work / "smoke")
            base = ["--config", str(SMOKE_YAML), "--set", f"data.source.path={(work / 'smoke').as_posix()}", "--set", f"model.path={TINY.as_posix()}", "--set", f"run.experiments_dir={(work / 'e').as_posix()}"]
            code, out, err = run_cli("prepare", "--parent", "exp_0009", *base)
            self.assertEqual((code, out), (2, ""))
            self.assertIn("exp_0009", err)
            code, out, err = run_cli("prepare", *base, "--set", "train.hyperparams.nope=1")
            self.assertEqual(code, 2)
            self.assertIn("nope", err)

    def test_prepare_alone_writes_the_reports_and_evaluate_refuses_the_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            write_smoke_data(SmokeConfig(n_samples=12), 42, work / "smoke")
            base = ["--config", str(SMOKE_YAML), "--set", f"data.source.path={(work / 'smoke').as_posix()}", "--set", f"model.path={TINY.as_posix()}", "--set", f"run.experiments_dir={(work / 'e').as_posix()}"]
            code, out, err = run_cli("prepare", *base)
            self.assertEqual(code, 0, err)
            self.assertEqual(list(json.loads(out)["stages"]), ["prepare"])
            self.assertTrue((work / "e" / "exp_0001" / "reports" / "split_report.txt").is_file())
            code, out, err = run_cli("evaluate", "--split", "test", *base)
            self.assertEqual((code, json.loads(out)["status"]), (2, "failed"))
            self.assertIn("refuses the Test split", json.loads(out)["message"])

    def test_the_subsampler_is_off_by_default_and_when_on_it_reduces_only_the_train_split_in_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            write_smoke_data(SmokeConfig(n_samples=12), 42, work / "smoke")
            base = ["--config", str(SMOKE_YAML), "--set", f"data.source.path={(work / 'smoke').as_posix()}", "--set", f"model.path={TINY.as_posix()}", "--set", f"run.experiments_dir={(work / 'e').as_posix()}"]
            code, out, err = run_cli("prepare", *base)
            self.assertEqual(code, 0, err)
            off = json.loads(out)["stages"]["prepare"]["summary"]
            self.assertEqual((off["kept_train"], off["kept_val"], off["kept_test"]), (10, 1, 1))  # nothing changed by default
            self.assertFalse((work / "e" / "exp_0001" / "reports" / "subsample_report.txt").exists())
            code, out, err = run_cli("prepare", *base, "--set", "subsample.enabled=true", "--set", 'subsample.size={"samples": 6}', "--set", "subsample.min_train_samples=3")
            self.assertEqual(code, 0, err)
            on = json.loads(out)["stages"]["prepare"]["summary"]
            self.assertEqual((on["kept_train"], on["kept_val"], on["kept_test"]), (6, 1, 1))  # Train reduced, Val and Test untouched
            report = (work / "e" / "exp_0002" / "reports" / "subsample_report.txt").read_text(encoding="utf-8")
            self.assertIn("requested 6 samples", report)
            code, out, err = run_cli("prepare", *base, "--set", "subsample.enabled=true", "--set", 'subsample.size={"samples": 99}')
            self.assertEqual((code, json.loads(out)["status"]), (3, "failed"))  # a size larger than what exists: data error, reported in the summary
            self.assertIn("99.00 samples", json.loads(out)["message"])
            code, out, err = run_cli("prepare", *base, "--set", "subsample.enabled=true", "--set", "subsample.preset=nope")
            self.assertEqual(code, 2)  # an unknown preset: config error
            self.assertIn("Known presets", err)

    def test_the_configs_load_and_the_defaults_are_the_documented_ones(self) -> None:
        smoke = load_config(SMOKE_YAML, env_file=None, target_environ={}, registry=section_registry(), overrides=[f"data.source.path={TINY.as_posix()}"])
        self.assertEqual(smoke.section("smoke", SmokeConfig), SmokeConfig())
        p = PipelineConfig()
        self.assertEqual((p.stages, p.skip_completed, p.eval_split, p.test_split), (["prepare", "evaluate", "search", "finalists", "select_and_register"], True, "val", "test"))  # the stages of `run` (DECISIONS 57)
        self.assertTrue(RunConfig().prevent_sleep)
        self.assertEqual((SmokeConfig().n_samples, SmokeConfig().duration_sec, SmokeConfig().steps), (12, 3, 3))
        loaded = load_config(ROOT / "configs" / "pipeline_default.yaml", env_file=None, target_environ={}, registry=SectionRegistry({"run": RunConfig, "pipeline": PipelineConfig}))
        self.assertEqual(loaded.section("pipeline", PipelineConfig), p)

    def test_smoke_data_is_deterministic_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            write_smoke_data(SmokeConfig(n_samples=4, duration_sec=1), 7, Path(a))
            write_smoke_data(SmokeConfig(n_samples=4, duration_sec=1), 7, Path(b))
            self.assertEqual((Path(a) / "manifest.csv").read_bytes(), (Path(b) / "manifest.csv").read_bytes())
            self.assertEqual((Path(a) / "audio" / "smoke_003.wav").read_bytes(), (Path(b) / "audio" / "smoke_003.wav").read_bytes())
            import wave

            with wave.open(str(Path(a) / "audio" / "smoke_000.wav")) as w:
                self.assertEqual((w.getframerate(), w.getnchannels(), w.getnframes()), (16000, 1, 16000))


class CheckTests(unittest.TestCase):
    def test_11_a_healthy_environment_passes_and_the_report_has_the_expected_lines(self) -> None:
        text, code = check_module.format_report(check_module.run_checks(None, ROOT, ROOT))
        self.assertEqual(code, 0, text)
        for part in ("Python 3.", "package torch", "package transformers", "free disk space", "memory:", "cores:", "code version", "Everything checked is fine."):
            self.assertIn(part, text)
        code, out, err = run_cli("check")
        self.assertEqual((code, err), (0, ""))
        self.assertIn("Everything checked is fine.", out)

    def test_check_lists_all_problems_at_once(self) -> None:
        config = make_config("model.path=C:/nowhere/model")  # the data folder is replaced by a missing one below

        def missing(name: str) -> str:
            if name in ("torch", "scipy"):
                raise check_module.metadata.PackageNotFoundError(name)
            return "1.0"

        with mock.patch.object(check_module.metadata, "version", side_effect=missing), mock.patch.object(check_module, "_data_path", return_value=Path("C:/nowhere/data")):
            text, code = check_module.format_report(check_module.run_checks(config, ROOT, ROOT))
        self.assertEqual(code, 2)
        problems = [line for line in text.splitlines() if line.startswith("PROBLEM")]
        self.assertEqual(len(problems), 4)  # torch, scipy, the model folder and the data folder: ALL of them, not only the first
        self.assertIn("4 problem(s) found", text)

    def test_check_with_an_invalid_config_exits_with_2(self) -> None:
        code, out, err = run_cli("check", "--config", str(SMOKE_YAML), "--set", "nope.x=1")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("nope", err)

    def test_the_code_version_is_the_constant_and_a_manifest_adds_its_hash(self) -> None:
        import stt

        self.assertEqual(code_version(), stt.__version__)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(code_version(Path(tmp)), stt.__version__)
            self.assertIsNone(manifest_hash(Path(tmp)))
            (Path(tmp) / "MANIFEST.sha256").write_text("abc", encoding="utf-8")
            suffix = manifest_hash(Path(tmp))
            self.assertEqual(len(suffix or ""), 8)
            self.assertEqual(code_version(Path(tmp)), f"{stt.__version__}+{suffix}")


class ImportRuleTests(unittest.TestCase):
    CONCRETE = {
        "create_data_source", "create_loader", "resolve_split", "TextNormalizer", "SampleFilter", "FeaturePipeline", "TrainerCore", "Evaluator",
        "Comparator", "HfTranscriber", "ExperimentStore", "LogSetup", "CsvMetricsSink", "ConsoleMetricsSink", "SinkRegistry",
        "HyperparamSearcher", "resolve_subsample", "Selector", "ModelRegistry", "ReportBuilder", "HtmlProgressSink", "ModelExportService",  # the search service and the screening choice are wired in wiring.py only
    }

    def test_12_only_wiring_imports_concrete_classes_and_nothing_imports_torch_or_prompts(self) -> None:
        offenders, heavy, prompts = [], [], []
        for path in sorted((ROOT / "src" / "stt" / "orchestrator").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    names = {a.name for a in node.names}
                    if names & self.CONCRETE and path.name != "wiring.py":
                        offenders.append((path.name, sorted(names & self.CONCRETE)))
                    if (node.module or "").split(".")[0] in {"torch", "transformers"}:
                        heavy.append(path.name)
                if isinstance(node, ast.Import) and any(a.name.split(".")[0] in {"torch", "transformers"} for a in node.names):
                    heavy.append(path.name)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "input":
                    prompts.append(path.name)
        self.assertEqual((offenders, heavy, prompts), ([], [], []))
        stage_source = (ROOT / "src" / "stt" / "orchestrator" / "stages.py").read_text(encoding="utf-8")
        self.assertNotIn("stt.datasource", stage_source)
        self.assertNotIn("stt.modelloader", stage_source)

    def test_importing_the_orchestrator_and_the_wiring_does_not_import_torch(self) -> None:
        code = "import sys; sys.path.insert(0, 'src'); import stt.orchestrator, stt.orchestrator.wiring, stt.orchestrator.cli; print('torch' in sys.modules, 'transformers' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False False")

    def test_the_package_runs_as_a_module(self) -> None:
        out = subprocess.run([sys.executable, "-m", "stt", "--help"], cwd=ROOT, capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": "src"})
        self.assertEqual(out.returncode, 0)
        self.assertIn("smoke", out.stdout)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
