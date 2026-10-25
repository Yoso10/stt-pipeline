"""Unit tests of Orchestrator Part 2: the `finalists` and `select_and_register` stages, the one-time Test look, the registry command and `python -m stt run`.

How to run (project root, virtual environment active):
    python -m unittest tests.test_orchestrator_run -v
Most tests use FAKE services (tests/search_stage_fakes.py: a synthetic Val loss and a scripted number of wrong words per model and split, no model, no audio) with the REAL search, selection and
registry services and the real Orchestrator, in temporary folders (fast). `RealRunTests` run the whole `run` command on synthetic clips and the stored tiny test model (about a minute).
The Test split of the REAL Daf Yomi data is never touched: the fake Test is synthetic and the real run uses synthetic clips. Nothing is downloaded, nothing is deleted by the code under test.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from search_stage_fakes import FakeEvaluation, FakeTraining, fake_services  # noqa: E402

from stt.config import AppConfig, load_config  # noqa: E402
from stt.evaluator import eval_result_from_json  # noqa: E402
from stt.orchestrator import (  # noqa: E402
    InterruptFlag,
    NoopSleepGuard,
    Orchestrator,
    OrchestratorError,
    PipelineConfig,
    RunOutcome,
    RunPlan,
    SmokeConfig,
    StageContext,
    default_stage_registry,
    load_result,
    write_smoke_data,
)
from stt.orchestrator import cli  # noqa: E402
from stt.orchestrator.finalists import load_finalists  # noqa: E402
from stt.orchestrator.selection import validate_registry_location  # noqa: E402
from stt.orchestrator.stagehelpers import evaluate_model  # noqa: E402
from stt.orchestrator.testlook import TestLookStore, once_per_experiment  # noqa: E402
from stt.orchestrator.wiring import RegistryWiring, exit_code_for, section_registry  # noqa: E402
from stt.registry import RegistryError, RegistryVerifyError  # noqa: E402
from stt.runmanager import ExperimentHandle, ExperimentSpec, ExperimentStore, RunConfig  # noqa: E402
from stt.selector import Candidate, SelectorError, TestAlreadyEvaluatedError  # noqa: E402

TINY = ROOT / "tests" / "assets" / "tiny_whisper"
SMOKE_YAML = ROOT / "configs" / "smoke.yaml"
GRID = '{"learning_rate": {"kind": "log_float", "low": 1e-5, "high": 1e-4, "points": 2}, "grad_accum_steps": {"kind": "categorical", "choices": [2, 4]}}'
WINNER_MODEL = "exp_0001_final_trial_001_step_00150"  # with the fake tables: the best finalist is trial 1


def run_config(tmp: Path, *overrides: str) -> AppConfig:
    base = [
        f"data.source.path={TINY.as_posix()}", f"model.path={TINY.as_posix()}", f"run.experiments_dir={(tmp / 'experiments').as_posix()}", f"registry.dir={(tmp / 'registry').as_posix()}",
        "search.screening.preset=smoke", "search.n_trials=4", f"search.parameters={GRID}", "train.cpu_threads=4", "search.pruning.enabled=false",
    ]
    return load_config(SMOKE_YAML, overrides=[*base, *overrides], registry=section_registry(), env_file=None, target_environ={})


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class RunCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.store = ExperimentStore(RunConfig(experiments_dir=self.tmp / "experiments"))

    def experiment(self, config: AppConfig) -> ExperimentHandle:
        return self.store.create(ExperimentSpec(config, None, "0.0.0", {"torch": "x"}, "2026-10-06T09:00:00"))

    def execute(self, config: AppConfig, handle: ExperimentHandle, evaluation: FakeEvaluation | None = None, flag: Any = None, hook: Any = None) -> tuple[RunOutcome, Any]:
        services = fake_services(config, hook, evaluation=evaluation)
        stages = list(config.section("pipeline", PipelineConfig).stages)
        outcome = Orchestrator(config, services, default_stage_registry(), flag or InterruptFlag(), NoopSleepGuard()).run(RunPlan(stages, handle))
        return outcome, services

    def registry(self, config: AppConfig) -> RegistryWiring:
        return RegistryWiring(config)


class FullRunTests(RunCase):
    def test_a_run_with_a_clear_winner_retrains_selects_looks_at_the_test_once_and_registers(self) -> None:
        config = run_config(self.tmp)
        handle = self.experiment(config)
        outcome, services = self.execute(config, handle)
        self.assertEqual((outcome.exit_code, outcome.status, outcome.verdict), (0, "completed", "winner"))
        self.assertEqual(list(outcome.stage_results), ["prepare", "evaluate", "search", "finalists", "select_and_register"])
        s = outcome.stage_results["select_and_register"].summary
        self.assertEqual((s["winner_trial"], s["test_status"], s["entry_id"], s["not_confirmed"], s["entry_kind"]), (1, "confirmed", "model_0001", 0, "model"))
        self.assertEqual((s["baseline_val_wer"], s["winner_val_wer"]), (0.5, round(1 / 6, 4)))
        calls = services.evaluation.calls
        self.assertEqual(sorted(name for split, name in calls if split == "test"), sorted(["baseline", WINNER_MODEL]))  # the Test: the winner and the baseline, once each
        self.assertEqual(sum(1 for split, _ in calls if split == "val"), 4)  # the baseline and the three finalists
        self.assertEqual(sorted(set(services.models.requested)), ["test", "train", "val"])
        self.assertEqual(sorted(json.loads((handle.paths.reports / "test_once.json").read_text(encoding="utf-8"))["looked"]), ["baseline", "trial_1"])
        reports = handle.paths.reports
        for name in ("eval_result_baseline.json", "finalists_report.txt", "selection.json", "selection_table.csv", "selection_reason.txt", "test_result_baseline.json", "test_result_trial_1.json", "finalists/final_trial_001.val.json"):
            self.assertTrue((reports / name).is_file(), name)
        self.assertEqual(eval_result_from_json((reports / "eval_result_baseline.json").read_text(encoding="utf-8")).split_name, "val")
        self.assertEqual(eval_result_from_json((reports / "test_result_trial_1.json").read_text(encoding="utf-8")).split_name, "test")

    def test_the_registry_entry_holds_the_model_the_card_the_review_list_and_the_fingerprints(self) -> None:
        config = run_config(self.tmp)
        handle = self.experiment(config)
        self.execute(config, handle)
        entry = self.tmp / "registry" / "model_0001"
        for name in ("model/model.safetensors", "MODEL_CARD.md", "model_card.json", "review_list.txt", "SHA256SUMS"):
            self.assertTrue((entry / name).is_file(), name)
        self.assertTrue((self.tmp / "registry" / "index.csv").is_file())
        card = json.loads((entry / "model_card.json").read_text(encoding="utf-8"))
        text = json.dumps(card)
        for part in ("d" * 64, "s" * 64, "2026-10-06T10:00:00", "0.0.0", "exp_0001/config_full.yaml", "torch"):
            self.assertIn(part, text)
        self.assertIn("winner", (entry / "MODEL_CARD.md").read_text(encoding="utf-8"))
        self.assertTrue(self.registry(config).verify("model_0001").ok)

    def test_nothing_is_deleted_only_the_winner_model_moves_into_the_registry(self) -> None:
        config = run_config(self.tmp)
        handle = self.experiment(config)
        self.execute(config, handle)
        candidates = handle.paths.candidates
        for trial in (0, 1, 3):  # the three finalists: folders, checkpoints and the other models are all still there
            folder = candidates / f"exp_0001_final_trial_{trial:03d}"
            self.assertTrue((folder / "checkpoints").is_dir(), trial)
        self.assertTrue((candidates / "exp_0001_final_trial_000" / "exp_0001_final_trial_000_step_00150").is_dir())
        self.assertFalse((candidates / "exp_0001_final_trial_001" / WINNER_MODEL).exists())  # moved into model_0001/model
        for trial in range(4):  # the search trials are untouched
            self.assertTrue((candidates / f"exp_0001_trial_{trial:03d}" / "checkpoints").is_dir())

    def test_when_no_finalist_beats_the_baseline_the_baseline_is_retained_and_the_test_is_never_evaluated(self) -> None:
        config = run_config(self.tmp)
        handle = self.experiment(config)
        outcome, services = self.execute(config, handle, FakeEvaluation(config, val_errors={0: 3, 1: 3, 2: 3, 3: 3}))
        self.assertEqual((outcome.exit_code, outcome.verdict), (0, "baseline_retained"))
        s = outcome.stage_results["select_and_register"].summary
        self.assertEqual((s["test_status"], s["entry_kind"], s["winner_trial"]), ("not_evaluated", "baseline_retained", "none"))
        self.assertEqual([c for c in services.evaluation.calls if c[0] == "test"], [])
        self.assertNotIn("test", services.models.requested)
        self.assertFalse((handle.paths.reports / "test_once.json").exists())
        self.assertFalse((self.tmp / "registry" / "model_0001" / "model").exists())  # a retained baseline has no model folder
        for trial in (0, 1, 3):  # nothing was deleted either (the owner's rule overrides the deleting default of the registry)
            self.assertTrue((handle.paths.candidates / f"exp_0001_final_trial_{trial:03d}" / f"exp_0001_final_trial_{trial:03d}_step_00150").is_dir())

    def test_a_winner_that_the_test_does_not_confirm_is_registered_and_flagged_not_dropped(self) -> None:
        config = run_config(self.tmp)
        handle = self.experiment(config)
        outcome, _ = self.execute(config, handle, FakeEvaluation(config, test_errors={0: 3, 1: 3, 3: 3}))
        s = outcome.stage_results["select_and_register"].summary
        self.assertEqual((outcome.exit_code, outcome.verdict, s["test_status"], s["not_confirmed"], s["entry_kind"]), (0, "winner", "not_confirmed", 1, "model"))
        card = json.loads((self.tmp / "registry" / "model_0001" / "model_card.json").read_text(encoding="utf-8"))
        self.assertIn("does not confirm", json.dumps(card))
        self.assertIn("not_confirmed", (handle.paths.reports / "selection_reason.txt").read_text(encoding="utf-8"))


class RerunAndCrashTests(RunCase):
    def test_a_rerun_skips_every_stage_and_looks_at_nothing_again(self) -> None:
        config = run_config(self.tmp)
        handle = self.experiment(config)
        self.execute(config, handle)
        second, services = self.execute(config, handle)
        self.assertEqual([r.status for r in second.stage_results.values()], ["skipped"] * 5)
        self.assertEqual((second.verdict, services.evaluation.calls, services.training.calls), ("winner", [], 0))
        self.assertEqual(len(self.registry(config).entries()), 1)

    def test_an_experiment_that_is_already_registered_is_never_registered_twice(self) -> None:
        config = run_config(self.tmp)
        handle = self.experiment(config)
        self.execute(config, handle)
        handle.mark_stage("select_and_register", "failed", None)  # as if the process died after the registration, before the marker
        second, services = self.execute(config, handle)
        s = second.stage_results["select_and_register"].summary
        self.assertEqual((second.exit_code, s["entry_id"], s["already_registered"], second.verdict), (0, "model_0001", 1, "winner"))
        self.assertEqual(len(self.registry(config).entries()), 1)
        self.assertEqual(services.evaluation.calls, [])

    def test_a_failed_registration_keeps_the_test_results_and_the_rerun_does_not_look_at_the_test_again(self) -> None:
        config = run_config(self.tmp)
        handle = self.experiment(config)
        leftover = self.tmp / "registry" / ".tmp_model_0001"
        leftover.mkdir(parents=True)  # a leftover of an interrupted registration: the registry refuses to register
        first, services = self.execute(config, handle)
        self.assertEqual((first.exit_code, first.status), (4, "failed"))
        self.assertIn("leftover temporary folder", first.message or "")
        self.assertEqual(sum(1 for split, _ in services.evaluation.calls if split == "test"), 2)  # the Test WAS looked at, and both results are stored
        self.assertTrue((handle.paths.reports / "test_result_trial_1.json").is_file())
        leftover.rmdir()
        second, services2 = self.execute(config, handle)
        self.assertEqual((second.exit_code, second.verdict), (0, "winner"))
        self.assertEqual([c for c in services2.evaluation.calls if c[0] == "test"], [])  # served from the stored results: no new look
        self.assertEqual(len(self.registry(config).entries()), 1)

    def test_an_interrupted_finalist_is_resumed_and_the_finished_one_is_not_trained_again(self) -> None:
        config = run_config(self.tmp)
        handle = self.experiment(config)
        flag = InterruptFlag()
        first, _ = self.execute(config, handle, flag=flag, hook=lambda call, step: flag.set() if (call, step) == (6, 100) else None)  # calls 1-4 are the search, 5 and 6 the first two finalists
        self.assertEqual((first.exit_code, first.status), (130, "stopped"))
        self.assertIsNone(handle.stage_mark("finalists"))
        self.assertTrue((handle.paths.reports / "finalists" / "final_trial_001.json").is_file())  # the first finalist (trial 1) was finished
        self.assertFalse((handle.paths.reports / "finalists" / "final_trial_000.json").exists())
        second, services = self.execute(config, handle)
        self.assertEqual((second.exit_code, second.verdict), (0, "winner"))
        self.assertEqual(services.training.calls, 2)  # the second finalist (resumed) and the third; the first is not trained again
        call = json.loads((handle.paths.metrics / "final_trial_000" / "call.json").read_text(encoding="utf-8"))
        self.assertTrue(call["resume"])
        self.assertEqual(len(load_finalists(handle)), 3)

    def test_finalist_repeats_above_one_are_refused_with_exit_code_2(self) -> None:
        config = run_config(self.tmp, "search.finalist_repeats=2")
        outcome, _ = self.execute(config, self.experiment(config))
        self.assertEqual((outcome.exit_code, outcome.status), (2, "failed"))
        self.assertIn("finalist_repeats", outcome.message or "")


class TestOnceTests(RunCase):
    def test_a_started_look_without_a_stored_result_fails_and_a_stored_result_is_returned(self) -> None:
        config = run_config(self.tmp)
        handle = self.experiment(config)
        store = TestLookStore(handle)
        candidate = Candidate(1, {}, Path("m"), None, 0.1, 1)  # type: ignore[arg-type]

        def boom(c: Candidate | None) -> Any:
            raise RuntimeError("the evaluation died")

        look = once_per_experiment(store, boom)
        with self.assertRaises(RuntimeError):
            look(candidate)  # marked first, then it died
        with self.assertRaises(TestAlreadyEvaluatedError):
            once_per_experiment(store, boom)(candidate)  # a second process: the look counts, even without a result
        self.assertEqual(store.looked(), ["trial_1"])

    def test_the_test_split_is_refused_everywhere_except_for_the_selection_stage(self) -> None:
        config = run_config(self.tmp)
        handle = self.experiment(config)
        services = fake_services(config)
        ctx = StageContext(config, handle, services, InterruptFlag(), {}, {})
        with self.assertRaises(OrchestratorError) as raised:
            evaluate_model(ctx, "test", None, "baseline")
        self.assertEqual(raised.exception.exit_code, 2)
        self.assertIn("ONCE", str(raised.exception))
        self.assertEqual(services.evaluation.calls, [])  # type: ignore[union-attr]
        self.assertEqual(evaluate_model(ctx, "val", None, "baseline").split_name, "val")

    def test_the_exit_codes_of_the_selection_and_the_registry(self) -> None:
        self.assertEqual([exit_code_for(e) for e in (SelectorError("x"), TestAlreadyEvaluatedError("x"), RegistryError("x"), RegistryVerifyError("x"))], [4, 4, 4, 5])


class RegistryLocationTests(RunCase):
    def test_the_registry_must_lie_outside_the_experiments_folder(self) -> None:
        experiments = self.tmp / "experiments"
        validate_registry_location(experiments, self.tmp / "registry")  # a sibling: fine
        for bad in (experiments, experiments / "registry", self.tmp):
            with self.subTest(registry=str(bad)), self.assertRaises(OrchestratorError) as raised:
                validate_registry_location(experiments, bad)
            self.assertEqual(raised.exception.exit_code, 2)
            self.assertIn("out of reach", str(raised.exception))

    def test_the_command_refuses_a_registry_inside_the_experiments_before_it_makes_an_experiment(self) -> None:
        code, out, err = run_cli("run", "--config", str(SMOKE_YAML), "--set", f"data.source.path={TINY.as_posix()}", "--set", f"model.path={TINY.as_posix()}", "--set", f"run.experiments_dir={(self.tmp / 'experiments').as_posix()}",
                                 "--set", f"registry.dir={(self.tmp / 'experiments' / 'registry').as_posix()}")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("registry.dir", err)
        self.assertFalse((self.tmp / "experiments").exists())  # nothing was created

    def test_the_stage_checks_it_too_when_it_is_run_without_the_command_line(self) -> None:
        config = run_config(self.tmp, f"registry.dir={(self.tmp / 'experiments' / 'registry').as_posix()}")
        outcome, services = self.execute(config, self.experiment(config))
        self.assertEqual((outcome.exit_code, outcome.status), (2, "failed"))
        self.assertEqual([c for c in services.evaluation.calls if c[0] == "test"], [])  # refused before any Test look


class RegistryCommandTests(RunCase):
    def setUp(self) -> None:
        super().setUp()
        self.config = run_config(self.tmp)
        self.execute(self.config, self.experiment(self.config))
        self.args = ["--set", f"registry.dir={(self.tmp / 'registry').as_posix()}"]

    def snapshot(self) -> list[tuple[str, int, int]]:
        return sorted((str(p.relative_to(self.tmp)), p.stat().st_size, p.stat().st_mtime_ns) for p in (self.tmp / "registry").rglob("*") if p.is_file())

    def test_list_and_verify_are_read_only_and_say_what_is_registered(self) -> None:
        before = self.snapshot()
        code, out, _ = run_cli("registry", "list", *self.args)
        self.assertEqual(code, 0)
        for part in ("1 entry", "model_0001", "from exp_0001", "winner", "Test status confirmed"):
            self.assertIn(part, out)
        code, out, _ = run_cli("registry", "verify", *self.args)
        self.assertEqual(code, 0)
        self.assertIn("OK      model_0001", out)
        self.assertEqual(self.snapshot(), before)  # nothing changed: not a byte, not a time stamp
        self.assertEqual(run_cli("registry", "verify", "model_0001", *self.args)[0], 0)

    def test_a_changed_file_gives_exit_code_5_and_nothing_is_repaired_or_deleted(self) -> None:
        target = self.tmp / "registry" / "model_0001" / "model" / "model.safetensors"
        target.write_bytes(b"corrupted")
        code, out, _ = run_cli("registry", "verify", *self.args)
        self.assertEqual(code, 5)
        self.assertIn("FAILED  model_0001", out)
        self.assertEqual(target.read_bytes(), b"corrupted")
        self.assertEqual(run_cli("registry", "list", *self.args)[0], 0)  # listing still works

    def test_rebuild_index_is_explicit_and_an_unknown_entry_is_an_error(self) -> None:
        index = self.tmp / "registry" / "index.csv"
        index.write_text("damaged", encoding="utf-8")
        self.assertEqual(run_cli("registry", "list", *self.args)[0], 0)  # the list reads the cards (the source of truth), not the index ...
        self.assertEqual(index.read_text(encoding="utf-8"), "damaged")  # ... and a damaged index is never rebuilt silently
        code, out, _ = run_cli("registry", "rebuild-index", *self.args)
        self.assertEqual(code, 0)
        self.assertIn("model_0001", index.read_text(encoding="utf-8-sig"))
        self.assertNotEqual(run_cli("registry", "verify", "model_9999", *self.args)[0], 0)

    def test_an_empty_or_missing_registry_lists_nothing_and_creates_nothing(self) -> None:
        none = self.tmp / "no_registry_here"
        code, out, _ = run_cli("registry", "list", "--set", f"registry.dir={none.as_posix()}")
        self.assertEqual(code, 0)
        self.assertIn("0 entries", out)
        self.assertFalse(none.exists())

    def test_the_commands_have_help_with_an_example(self) -> None:
        for argv in (["run", "--help"], ["registry", "--help"], ["registry", "list", "--help"], ["registry", "verify", "--help"]):
            out = io.StringIO()
            with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
                cli.main(argv)
            self.assertIn("python -m stt", out.getvalue(), argv)


class RunCommandTests(unittest.TestCase):
    def test_run_uses_the_stages_of_pipeline_stages_and_an_unknown_name_fails_before_any_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            write_smoke_data(SmokeConfig(n_samples=12), 42, work / "smoke")
            base = ["--config", str(SMOKE_YAML), "--set", f"data.source.path={(work / 'smoke').as_posix()}", "--set", f"model.path={TINY.as_posix()}", "--set", f"run.experiments_dir={(work / 'experiments').as_posix()}",
                    "--set", f"registry.dir={(work / 'registry').as_posix()}"]
            code, out, err = run_cli("run", *base, "--set", 'pipeline.stages=["prepare"]')
            self.assertEqual(code, 0, err)
            self.assertEqual(list(json.loads(out)["stages"]), ["prepare"])  # only what pipeline.stages lists
            code, out, err = run_cli("run", *base, "--set", 'pipeline.stages=["prepare", "nope"]')
            self.assertEqual((code, out), (2, ""))
            self.assertIn("Unknown stage 'nope'", err)
            self.assertIn("select_and_register", err)  # the registered names are listed

    def test_the_default_stage_list_is_the_run_chain(self) -> None:
        self.assertEqual(PipelineConfig().stages, ["prepare", "evaluate", "search", "finalists", "select_and_register"])
        self.assertEqual(sorted(cli.PLANS), ["benchmark", "evaluate", "final-test", "learning-curve", "prepare", "search", "smoke", "train"])  # `run` is not a fixed plan: it reads pipeline.stages



class RealRunTests(unittest.TestCase):
    """The whole `run` command on 60 synthetic clips and the stored tiny test model. `select.min_verdict=no_clear_difference` lets a finalist win (the random tiny model never improves
    clearly), so that the one-time Test look, on the SYNTHETIC Test clips, is exercised for real."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        write_smoke_data(SmokeConfig(n_samples=60), 42, cls.tmp / "smoke")
        sets = [
            f"data.source.path={(cls.tmp / 'smoke').as_posix()}", f"model.path={TINY.as_posix()}", f"run.experiments_dir={(cls.tmp / 'experiments').as_posix()}", f"registry.dir={(cls.tmp / 'registry').as_posix()}",
            "search.n_trials=4", f"search.parameters={GRID}", "search.screening.preset=smoke", "subsample.min_train_samples=1", "subsample.val_samples=6", "train.hyperparams.max_steps=6",
            "train.eval_every_steps=3", "train.save_every_steps=3", "select.min_verdict=no_clear_difference",
        ]
        cls.args = ["--config", str(SMOKE_YAML), *[a for s in sets for a in ("--set", s)]]
        cls.code, cls.out, cls.err = run_cli("run", *cls.args)

    @classmethod
    def tearDownClass(cls) -> None:
        import os
        import shutil
        import stat

        shutil.rmtree(cls.tmp, onexc=lambda func, path, exc: (os.chmod(path, stat.S_IWRITE), func(path)))  # registry entries are read-only
        cls._tmp.cleanup() if cls.tmp.exists() else None

    def test_the_real_run_registers_a_winner_after_one_test_look_and_the_registry_verifies(self) -> None:
        self.assertEqual(self.code, 0, self.err)
        s = json.loads(self.out)
        self.assertEqual((s["status"], s["verdict"], s["exit_code"]), ("completed", "winner", 0))
        self.assertEqual(list(s["stages"]), ["prepare", "evaluate", "search", "finalists", "select_and_register"])
        key = s["key_numbers"]
        self.assertEqual((key["entry_id"], key["entry_kind"], key["test_status"] in ("confirmed", "not_confirmed")), ("model_0001", "model", True))
        exp = self.tmp / "experiments" / "exp_0001"
        looked = json.loads((exp / "reports" / "test_once.json").read_text(encoding="utf-8"))["looked"]
        self.assertEqual(sorted(looked), ["baseline", f"trial_{key['winner_trial']}"])  # the Test: the winner and the baseline, once each
        self.assertEqual(sorted(p.name for p in (exp / "reports").glob("test_result_*.json")), sorted(f"test_result_{k}.json" for k in looked))
        self.assertIn("VERDICT: winner", self.err)
        self.assertTrue((self.tmp / "registry" / "model_0001" / "model" / "config.json").is_file())
        code, out, _ = run_cli("registry", "verify", "--set", f"registry.dir={(self.tmp / 'registry').as_posix()}")
        self.assertEqual(code, 0)
        self.assertIn("OK      model_0001", out)
        for trial in (1, 2, 3):
            self.assertTrue((exp / "candidates" / f"exp_0001_final_trial_{trial:03d}" / "checkpoints").is_dir())  # nothing was deleted

    def test_a_rerun_of_the_real_run_skips_everything_and_does_not_look_at_the_test_again(self) -> None:
        exp = self.tmp / "experiments" / "exp_0001"
        before = (exp / "reports" / "test_once.json").read_text(encoding="utf-8")
        code, out, err = run_cli("run", *self.args, "--experiment", "exp_0001")
        self.assertEqual(code, 0, err)
        self.assertEqual({r["status"] for r in json.loads(out)["stages"].values()}, {"skipped"})
        self.assertEqual((exp / "reports" / "test_once.json").read_text(encoding="utf-8"), before)
        self.assertEqual(len(list((self.tmp / "registry").glob("model_*"))), 1)


if __name__ == "__main__":
    unittest.main()
