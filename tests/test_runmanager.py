"""Unit tests of the RunManager component (stt.runmanager): the acceptance tests of the C12 brief, offline, in temporary folders.

How to run (project root, virtual environment active):
    python -m unittest tests.test_runmanager -v
No model, no data and no network are used. Concurrent creation is tested with threads and with separate Python processes.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stt.config import AppConfig, SectionRegistry, load_config  # noqa: E402
from stt.runmanager import (  # noqa: E402
    CleanupPolicy,
    ExperimentSpec,
    ExperimentStore,
    LogSetup,
    RunConfig,
    RunLock,
    RunManagerError,
    diff_configs,
    model_folder_name,
)
from stt.runmanager.__main__ import main as helper_main  # noqa: E402
from stt.trainercore import TrainConfig  # noqa: E402

REGISTRY = SectionRegistry({"train": TrainConfig})


def make_config(*overrides: str) -> AppConfig:
    return load_config(None, overrides=list(overrides), registry=REGISTRY, env_file=None, target_environ={})


def spec(config: AppConfig | None = None, parent: str | None = None, created: str = "2026-10-05T12:00:00") -> ExperimentSpec:
    return ExperimentSpec(config or make_config(), parent, "0.0.0", {"torch": "2.14.1"}, created)


class RunManagerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.dir = self.tmp / "experiments"
        self.run_config = RunConfig(experiments_dir=self.dir)
        self.store = ExperimentStore(self.run_config)

    def make_store(self, **overrides: Any) -> ExperimentStore:
        return ExperimentStore(RunConfig.model_validate({"experiments_dir": self.dir, **overrides}))


class CreateTests(RunManagerTestCase):
    def test_1_create_makes_the_folder_with_every_file_and_the_next_call_makes_exp_0002(self) -> None:
        handle = self.store.create(spec())
        self.assertEqual(handle.experiment_id, "exp_0001")
        p = handle.paths
        for path in (p.config_full, p.config_diff, p.run_info, p.status):
            self.assertTrue(path.is_file(), path.name)
        for folder in (p.logs, p.metrics, p.search, p.candidates, p.reports):
            self.assertTrue(folder.is_dir(), folder.name)
        info = handle.run_info()
        self.assertEqual((info.experiment_id, info.config_ref, info.parent_id, info.seed, info.created_at), ("exp_0001", "exp_0001/config_full.yaml", None, 42, "2026-10-05T12:00:00"))
        self.assertEqual((info.code_version, dict(info.library_versions)), ("0.0.0", {"torch": "2.14.1"}))
        status = handle.status()
        self.assertEqual((status.status, status.reason, dict(status.stages)), ("created", None, {}))
        self.assertEqual(handle.config_ref, "exp_0001/config_full.yaml")
        self.assertEqual(self.store.create(spec()).experiment_id, "exp_0002")
        self.assertEqual([s.experiment_id for s in self.store.list_experiments()], ["exp_0001", "exp_0002"])

    def test_2_concurrent_creation_from_threads_and_processes_gives_distinct_numbers(self) -> None:
        made: list[str] = []
        guard = threading.Lock()

        def work() -> None:
            local = ExperimentStore(RunConfig(experiments_dir=self.dir))
            for _ in range(5):
                handle = local.create(spec())
                with guard:
                    made.append(handle.experiment_id)

        threads = [threading.Thread(target=work) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(made), 30)
        self.assertEqual(len(set(made)), 30)
        code = (
            "import sys; sys.path.insert(0, 'src'); from pathlib import Path\n"
            "from stt.config import SectionRegistry, load_config\n"
            "from stt.runmanager import ExperimentSpec, ExperimentStore, RunConfig\n"
            f"s = ExperimentStore(RunConfig(experiments_dir=Path(r'{self.dir}')))\n"
            "c = load_config(None, env_file=None, target_environ={})\n"
            "for _ in range(4): print(s.create(ExperimentSpec(c, None, '0', {}, 'x')).experiment_id)\n"
        )
        processes = [subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, stdout=subprocess.PIPE, text=True) for _ in range(3)]
        ids = [line for p in processes for line in p.communicate()[0].split()]
        self.assertEqual(len(ids), 12)
        self.assertEqual(len(set(ids)), 12)
        self.assertFalse(set(ids) & set(made))
        self.assertEqual(len(self.store.list_experiments()), 42)

    def test_3_the_diff_holds_exactly_the_changed_keys_against_the_parent_or_the_defaults(self) -> None:
        base = self.store.create(spec(make_config("train.hyperparams.max_steps=7")))
        self.assertEqual(self.load_diff(base), {"train": {"hyperparams": {"max_steps": 7}}})  # no parent: against the defaults
        child = self.store.create(spec(make_config("train.hyperparams.max_steps=7", "train.hyperparams.learning_rate=3e-5", "train.eval_every_steps=10"), parent="exp_0001"))
        self.assertEqual(self.load_diff(child), {"train": {"hyperparams": {"learning_rate": 3e-5}, "eval_every_steps": 10}})
        self.assertIn("exp_0001", child.paths.config_diff.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(child.run_info().parent_id, "exp_0001")
        same = self.store.create(spec(make_config("train.hyperparams.max_steps=7"), parent="exp_0001"))
        self.assertEqual(self.load_diff(same), {})

    @staticmethod
    def load_diff(handle: Any) -> Any:
        import yaml

        return yaml.safe_load(handle.paths.config_diff.read_text(encoding="utf-8")) or {}

    def test_diff_configs_pure_function(self) -> None:
        self.assertEqual(diff_configs({"a": {"b": 1, "c": [1, 2]}, "d": 1}, {"a": {"b": 1, "c": [1, 3]}, "d": 1, "e": {"f": 2}}), {"a": {"c": [1, 3]}, "e": {"f": 2}})
        self.assertEqual(diff_configs({"a": 1}, {}), {"a": None})
        self.assertEqual(diff_configs({"a": {"b": 1}}, {"a": {"b": 1}}), {})

    def test_4_config_full_is_read_only_and_a_resume_with_a_different_config_fails(self) -> None:
        config = make_config("train.hyperparams.max_steps=7")
        handle = self.store.create(spec(config))
        before = handle.paths.config_full.read_bytes()
        with self.assertRaises(PermissionError):
            handle.paths.config_full.write_text("changed", encoding="utf-8")
        self.assertTrue(handle.paths.config_full.stat().st_mode & stat.S_IWRITE == 0)
        reopened = self.store.open("exp_0001", config)  # the same config resumes
        self.assertEqual(reopened.paths.config_full.read_bytes(), before)
        with self.assertRaises(RunManagerError) as ctx:
            self.store.open("exp_0001", make_config("train.hyperparams.max_steps=8"))
        message = str(ctx.exception)
        self.assertIn("train.hyperparams.max_steps: stored 7, now 8", message)
        self.assertIn("parent", message)
        self.assertEqual(handle.paths.config_full.read_bytes(), before)
        loaded = load_config(handle.paths.config_full, registry=REGISTRY, env_file=None, target_environ={})  # config_full.yaml is a complete config file
        self.assertEqual(loaded, config)

    def test_a_resume_ignores_new_keys_that_equal_the_schema_default_but_not_real_differences(self) -> None:
        old = make_config("train.hyperparams.max_steps=7")  # an "older" config: only the train section exists
        self.store.create(spec(old))
        newer = lambda *o: load_config(None, overrides=list(o), registry=SectionRegistry({"train": TrainConfig, "run": RunConfig}), env_file=None, target_environ={})  # noqa: E731  (a later version added the run section)
        reopened = self.store.open("exp_0001", newer("train.hyperparams.max_steps=7"))
        self.assertEqual(reopened.experiment_id, "exp_0001")
        with self.assertRaises(RunManagerError) as ctx:  # a new key with a NON-default value is a real difference
            self.store.open("exp_0001", newer("train.hyperparams.max_steps=7", "run.id_digits=6"))
        self.assertIn("run.id_digits: stored None, now 6", str(ctx.exception))
        with self.assertRaises(RunManagerError) as ctx2:  # a changed value still fails, also when new default keys are present
            self.store.open("exp_0001", newer("train.hyperparams.max_steps=8"))
        self.assertIn("train.hyperparams.max_steps: stored 7, now 8", str(ctx2.exception))
        self.assertNotIn("run.", str(ctx2.exception))  # the ignored default keys are not listed as differences
        path = self.dir / "exp_0001" / "config_full.yaml"
        import yaml

        stored = yaml.safe_load(path.read_text(encoding="utf-8"))
        stored["train"]["removed_later"] = 1  # a key that exists only in the stored config (removed from the schema) fails
        os.chmod(path, stat.S_IWRITE)
        path.write_text(yaml.safe_dump(stored), encoding="utf-8")
        with self.assertRaises(RunManagerError) as ctx3:
            self.store.open("exp_0001", newer("train.hyperparams.max_steps=7"))
        self.assertIn("train.removed_later", str(ctx3.exception))

    def test_a_writable_config_when_the_key_says_so(self) -> None:
        handle = self.make_store(config_readonly=False).create(spec())
        handle.paths.config_full.write_text("x: 1", encoding="utf-8")

    def test_numbers_are_never_reused_after_a_gap_and_the_digits_come_from_config(self) -> None:
        for _ in range(3):
            self.store.create(spec())
        shutil.rmtree(self.dir / "exp_0002", onexc=lambda f, p, e: (os.chmod(p, stat.S_IWRITE), f(p)))
        self.assertEqual(self.store.create(spec()).experiment_id, "exp_0004")
        self.assertEqual(self.make_store(id_digits=2).create(spec()).experiment_id, "exp_05")

    def test_a_missing_parent_fails_and_lists_the_existing_experiments(self) -> None:
        self.store.create(spec())
        with self.assertRaises(RunManagerError) as ctx:
            self.store.create(spec(parent="exp_0009"))
        self.assertIn("exp_0009", str(ctx.exception))
        self.assertIn("exp_0001", str(ctx.exception))
        self.assertEqual(len(self.store.list_experiments()), 1)  # nothing was created

    def test_10_paths_with_spaces_and_hebrew_work_and_a_too_long_path_warns(self) -> None:
        odd = ExperimentStore(RunConfig(experiments_dir=self.tmp / "ניסויים של דף יומי" / "my runs"))
        handle = odd.create(spec())
        handle.write_report("דוח.html", "<p>שלום</p>")
        self.assertEqual((handle.paths.reports / "דוח.html").read_text(encoding="utf-8"), "<p>שלום</p>")
        with handle.lock():
            pass
        long_dir = self.tmp / ("d" * 90) / ("e" * 90)
        with self.assertLogs("stt.runmanager", level="WARNING") as logs:
            ExperimentStore(RunConfig(experiments_dir=long_dir)).create(spec())
        self.assertIn("Windows path limit", logs.output[0])
        self.assertIn("closer to the drive root", logs.output[0])


class StatusTests(RunManagerTestCase):
    def test_5_a_failed_run_keeps_its_folder_with_the_reason_and_resume_reopens_it(self) -> None:
        handle = self.store.create(spec())
        handle.set_status("running")
        handle.mark_stage("prepare", "ok", "abc123")
        handle.set_status("failed", "stage train failed: the loss became NaN at step 3")
        again = self.store.open("exp_0001")
        status = again.status()
        self.assertEqual((status.status, status.reason), ("failed", "stage train failed: the loss became NaN at step 3"))
        self.assertEqual(status.stages["prepare"].fingerprint, "abc123")  # the markers survive a status change
        self.assertEqual(again.stage_mark("prepare").status, "ok")  # type: ignore[union-attr]
        self.assertIsNone(again.stage_mark("train"))
        again.set_status("running")
        again.mark_stage("train", "failed")
        self.assertEqual(self.store.list_experiments()[0].status, "running")
        raw = json.loads(again.paths.status.read_text(encoding="utf-8"))
        self.assertEqual(set(raw), {"status", "reason", "stages"})

    def test_unknown_status_and_stage_status_fail(self) -> None:
        handle = self.store.create(spec())
        with self.assertRaises(RunManagerError):
            handle.set_status("paused")
        with self.assertRaises(RunManagerError):
            handle.mark_stage("train", "maybe")

    def test_a_damaged_status_file_fails_naming_it_and_the_folder_is_not_touched(self) -> None:
        handle = self.store.create(spec())
        handle.paths.status.write_text("{ not json", encoding="utf-8")
        snapshot = sorted((p.name, p.read_bytes()) for p in handle.paths.root.iterdir() if p.is_file())
        with self.assertRaises(RunManagerError) as ctx:
            self.store.open("exp_0001")
        self.assertIn("status.json", str(ctx.exception))
        self.assertEqual(snapshot, sorted((p.name, p.read_bytes()) for p in handle.paths.root.iterdir() if p.is_file()))
        self.assertEqual(self.store.list_experiments()[0].status, "unreadable")

    def test_an_unknown_experiment_fails_and_lists_the_existing_ones(self) -> None:
        self.store.create(spec())
        with self.assertRaises(RunManagerError) as ctx:
            self.store.open("exp_0007")
        self.assertIn("exp_0001", str(ctx.exception))
        with self.assertRaises(RunManagerError):
            self.store.open("../outside")

    def test_a_folder_that_cannot_be_created_fails_with_the_path(self) -> None:
        blocker = self.tmp / "blocked"
        blocker.write_text("a file, not a folder", encoding="utf-8")
        with self.assertRaises(RunManagerError) as ctx:
            ExperimentStore(RunConfig(experiments_dir=blocker / "sub")).create(spec())
        self.assertIn("blocked", str(ctx.exception))


class LockTests(RunManagerTestCase):
    def test_6_the_lock_blocks_a_second_opener_the_heartbeat_keeps_it_fresh_and_it_is_released(self) -> None:
        store = self.make_store(lock_heartbeat_seconds=0.05, lock_stale_minutes=0.05)  # stale after 3 seconds
        handle = store.create(spec())
        with handle.lock():
            self.assertTrue(handle.paths.lock.exists())
            with self.assertRaises(RunManagerError) as ctx:
                with store.open("exp_0001").lock():
                    pass
            self.assertIn("locked by another run", str(ctx.exception))
            os.utime(handle.paths.lock, (time.time() - 100, time.time() - 100))  # make it look old; the heartbeat must refresh it
            time.sleep(0.4)
            self.assertLess(time.time() - handle.paths.lock.stat().st_mtime, 1.0)
            with self.assertRaises(RunManagerError) as ctx2:  # still not stale, still locked
                with handle.lock():
                    pass
            self.assertIn("locked", str(ctx2.exception))
        self.assertFalse(handle.paths.lock.exists())
        with handle.lock():  # free again
            pass

    def test_a_stale_lock_fails_with_the_removal_instruction_and_is_not_removed(self) -> None:
        handle = self.store.create(spec())
        handle.paths.lock.write_text("{}", encoding="utf-8")
        old = time.time() - 3600
        os.utime(handle.paths.lock, (old, old))
        with self.assertRaises(RunManagerError) as ctx:
            with handle.lock():
                pass
        message = str(ctx.exception)
        self.assertIn("STALE", message)
        self.assertIn(str(handle.paths.lock), message)
        self.assertIn("delete the file", message)
        self.assertTrue(handle.paths.lock.exists())

    def test_the_lock_is_released_when_the_block_raises(self) -> None:
        handle = self.store.create(spec())
        with self.assertRaises(ValueError):
            with handle.lock():
                raise ValueError("boom")
        self.assertFalse(handle.paths.lock.exists())

    def test_run_lock_with_an_injected_clock(self) -> None:
        path = self.tmp / "x.lock"
        path.write_text("{}", encoding="utf-8")
        lock = RunLock(path, 30, 10, clock=lambda: path.stat().st_mtime + 11 * 60)
        with self.assertRaises(RunManagerError) as ctx:
            lock.acquire()
        self.assertIn("STALE", str(ctx.exception))
        lock.release()  # releasing a lock that was never acquired removes nothing
        self.assertTrue(path.exists())


class ReportAndLogTests(RunManagerTestCase):
    def test_7_write_report_writes_only_inside_reports(self) -> None:
        handle = self.store.create(spec())
        written = handle.write_report("final.json", '{"a": 1}')
        self.assertEqual(written, (handle.paths.reports / "final.json").resolve())
        handle.write_report("tables/predictions.csv", b"a,b\n1,2\n")
        self.assertEqual((handle.paths.reports / "tables" / "predictions.csv").read_bytes(), b"a,b\n1,2\n")
        outside = self.tmp / "escaped.txt"
        for bad in ("../escaped.txt", "..\\escaped.txt", str(outside), "", ".", "tables/../../x.txt"):
            with self.subTest(name=bad):
                with self.assertRaises(RunManagerError):
                    handle.write_report(bad, "x")
        self.assertFalse(outside.exists())
        self.assertFalse((handle.paths.root / "x.txt").exists())

    def test_9_the_log_file_gets_the_details_and_the_console_only_the_configured_level(self) -> None:
        handle = self.store.create(spec())
        setup = LogSetup(RunConfig(experiments_dir=self.dir))
        console = io.StringIO()
        with contextlib.redirect_stderr(console):
            setup.configure(handle)
            self.addCleanup(setup.close)
            log = logging.getLogger("stt.trainercore")
            log.debug("a detailed line")
            log.info("an important line")
            log.warning("a warning line")
        setup.close()
        text = handle.paths.run_log.read_text(encoding="utf-8")
        for line in ("a detailed line", "an important line", "a warning line", "stt.trainercore"):
            self.assertIn(line, text)
        self.assertNotIn("a detailed line", console.getvalue())
        self.assertIn("an important line", console.getvalue())
        quiet = LogSetup(RunConfig.model_validate({"experiments_dir": self.dir, "log": {"console_level": "WARNING"}}))
        console2 = io.StringIO()
        with contextlib.redirect_stderr(console2):
            quiet.configure(handle)
            logging.getLogger("stt.x").info("hidden on the console")
            logging.getLogger("stt.x").warning("shown")
        quiet.close()
        self.assertNotIn("hidden", console2.getvalue())
        self.assertIn("shown", console2.getvalue())
        self.assertIn("hidden on the console", handle.paths.run_log.read_text(encoding="utf-8"))
        self.assertEqual(logging.getLogger("stt").handlers, [])  # restored


class CleanupAndNamingTests(RunManagerTestCase):
    def build(self) -> None:
        for n, (status, created) in enumerate([("completed", "2026-01-01T00:00:00"), ("failed", "2026-01-02T00:00:00"), ("completed", "2026-10-04T00:00:00")], 1):
            handle = self.store.create(spec(created=created))
            handle.set_status(status)
            (handle.candidate_dir(f"exp_000{n}_trial_001_step_00100") / "checkpoints" / "checkpoint-100").mkdir(parents=True)
            (handle.candidate_dir(f"exp_000{n}_trial_001_step_00100") / "model").mkdir()
        self.registry = self.tmp / "registry"
        (self.registry / "model_0001").mkdir(parents=True)

    def tree(self) -> list[str]:
        return sorted(str(p.relative_to(self.tmp)) for p in self.tmp.rglob("*"))

    def test_8_cleanup_lists_candidates_without_deleting_and_the_flag_deletes_only_those_listed(self) -> None:
        self.build()
        before = self.tree()
        failed = self.store.propose_cleanup(CleanupPolicy(failed_experiments=True))
        self.assertEqual(failed, [self.dir / "exp_0002"])
        finalists = self.store.propose_cleanup(CleanupPolicy(candidates_of_completed=True))
        self.assertEqual([p.name for p in finalists], ["exp_0001_trial_001_step_00100", "exp_0003_trial_001_step_00100"])
        checkpoints = self.store.propose_cleanup(CleanupPolicy(checkpoints_of_completed=True))
        self.assertEqual([p.name for p in checkpoints], ["checkpoints", "checkpoints"])
        self.assertEqual(self.store.propose_cleanup(CleanupPolicy()), [])
        old = self.store.propose_cleanup(CleanupPolicy(candidates_of_completed=True, older_than_days=30, now="2026-10-05T00:00:00"))
        self.assertEqual([p.name for p in old], ["exp_0001_trial_001_step_00100"])
        self.assertEqual(self.tree(), before)  # proposing changed nothing
        deleted = self.store.delete(checkpoints[:1])
        self.assertEqual(deleted, [checkpoints[0].resolve()])
        self.assertFalse(checkpoints[0].exists())
        self.assertTrue(checkpoints[1].exists())
        self.assertTrue((self.registry / "model_0001").exists())
        self.assertTrue((self.dir / "exp_0001" / "config_full.yaml").exists())

    def test_delete_refuses_unsafe_paths_and_deletes_nothing(self) -> None:
        self.build()
        before = self.tree()
        listed = self.store.propose_cleanup(CleanupPolicy(candidates_of_completed=True))
        for bad in (self.registry / "model_0001", self.dir, self.dir / "exp_0001" / "config_full.yaml", self.dir / "exp_0099", self.tmp):
            with self.subTest(path=str(bad)):
                with self.assertRaises(RunManagerError):
                    self.store.delete([listed[0], bad])
        self.assertEqual(self.tree(), before)  # the valid path was not deleted either: all paths are checked first
        locked = self.store.open("exp_0001")
        with locked.lock():
            with self.assertRaises(RunManagerError) as ctx:
                self.store.delete([listed[0]])
            self.assertIn("fresh lock", str(ctx.exception))
        self.assertEqual(self.tree(), before)

    def test_delete_removes_read_only_files_it_was_given(self) -> None:
        self.build()
        model = self.dir / "exp_0002" / "candidates" / "exp_0002_trial_001_step_00100" / "model"
        weights = model / "model.safetensors"
        weights.write_bytes(b"x")
        os.chmod(weights, stat.S_IREAD)
        self.store.delete([self.dir / "exp_0002"])
        self.assertFalse((self.dir / "exp_0002").exists())

    def test_a_policy_with_an_age_needs_a_readable_now(self) -> None:
        with self.assertRaises(RunManagerError):
            self.store.propose_cleanup(CleanupPolicy(failed_experiments=True, older_than_days=3))

    def test_model_folder_names(self) -> None:
        self.assertEqual(model_folder_name("exp_0004", "baseline"), "exp_0004_baseline")
        self.assertEqual(model_folder_name("exp_0004", "trial", trial=3, step=500), "exp_0004_trial_003_step_00500")
        self.assertEqual(model_folder_name("exp_0004", "final", trial=3, step=1200), "exp_0004_final_trial_003_step_01200")
        self.assertLess(model_folder_name("exp_0004", "trial", trial=3, step=500), model_folder_name("exp_0004", "trial", trial=3, step=1200))
        for bad in (("trial", None, 5), ("final", 1, None), ("weird", 1, 1)):
            with self.assertRaises(RunManagerError):
                model_folder_name("exp_0004", bad[0], trial=bad[1], step=bad[2])  # type: ignore[arg-type]
        handle = self.store.create(spec())
        self.assertEqual(handle.candidate_dir("exp_0001_baseline"), handle.paths.candidates / "exp_0001_baseline")
        with self.assertRaises(RunManagerError):
            handle.candidate_dir("../x")


class LocationsAndHelperTests(RunManagerTestCase):
    def test_one_place_to_find_config_status_log_metrics_and_reports(self) -> None:
        handle = self.store.create(spec())
        handle.set_status("completed")
        trial_metrics = handle.metrics_dir("trial_001")
        (trial_metrics / "metrics.csv").write_text("step,loss\n1,2.0\n2,1.5\n", encoding="utf-8")
        handle.paths.run_log.write_text("line one\nline two\n", encoding="utf-8")
        handle.write_report("report.html", "<html/>")
        where = handle.locations()
        self.assertEqual((where.config_full, where.status, where.log), (handle.paths.config_full, handle.paths.status, handle.paths.run_log))
        self.assertEqual([p.name for p in where.metrics_files], ["metrics.csv"])
        self.assertEqual([p.name for p in where.reports], ["report.html"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = helper_main(["show", "exp_0001", "--dir", str(self.dir)])
        text = out.getvalue()
        self.assertEqual(code, 0)
        for part in ("exp_0001", "completed", "config_full.yaml", "status.json", "run.log", "line two", "2,1.5", "report.html"):
            self.assertIn(part, text)
        listing = io.StringIO()
        with contextlib.redirect_stdout(listing):
            helper_main(["list", "--dir", str(self.dir)])
        self.assertIn("exp_0001  completed", listing.getvalue())
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(helper_main(["show", "exp_0042", "--dir", str(self.dir)]), 2)
        self.assertIn("does not exist", err.getvalue())

    def test_config_defaults_and_yaml(self) -> None:
        c = RunConfig()
        self.assertEqual((c.experiments_dir, c.parent, c.id_digits, c.config_readonly, c.lock_heartbeat_seconds, c.lock_stale_minutes), (Path("experiments"), None, 4, True, 30.0, 10.0))
        self.assertEqual((c.log.console_level, c.log.file_level), ("INFO", "DEBUG"))
        loaded = load_config(ROOT / "configs" / "run_default.yaml", env_file=None, target_environ={}, registry=SectionRegistry({"run": RunConfig}))
        self.assertEqual(loaded.section("run", RunConfig), c)
        for bad in ({"id_digits": 0}, {"lock_stale_minutes": 0}, {"log": {"console_level": "LOUD"}}, {"nope": 1}):
            with self.assertRaises(ValueError):
                RunConfig.model_validate(bad)


class ImportRuleTests(unittest.TestCase):
    def test_11_no_network_and_no_model_imports(self) -> None:
        code = "import sys; sys.path.insert(0, 'src'); import stt.runmanager; print(any(m.split('.')[0] in ('torch', 'transformers', 'safetensors', 'numpy') for m in sys.modules))"
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False")
        with tempfile.TemporaryDirectory() as tmp, mock.patch("socket.socket.connect", side_effect=AssertionError("network used")):
            handle = ExperimentStore(RunConfig(experiments_dir=Path(tmp))).create(spec())
            with handle.lock():
                handle.write_report("a.txt", "x")


if __name__ == "__main__":
    unittest.main()
