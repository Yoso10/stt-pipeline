"""Tests of the Wave 2 exit check (Orchestrator Part 4): `python -m stt smoke --full`, the full `run` chain on synthetic clips and the stored tiny test model.

How to run (project root, virtual environment active):
    python -m unittest tests.test_orchestrator_exit -v
`smoke --full` is run once (about 40 seconds): search, finalists, selection with the one-time Test look on the SYNTHETIC Test clips, registration and `registry verify`. The plain `smoke` is
not changed (its tests are in tests/test_orchestrator.py). The real Daf Yomi data is never read here. Nothing is downloaded, nothing is deleted by the code under test.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stt.orchestrator import SmokeConfig, cli  # noqa: E402
from stt.orchestrator.smoke import FULL_MIN_SAMPLES, full_overrides  # noqa: E402

TINY = ROOT / "tests" / "assets" / "tiny_whisper"


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class SmokeFullTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.args = ["--set", f"smoke.work_dir={(cls.tmp / 'smoke').as_posix()}", "--set", f"smoke.model_dir={TINY.as_posix()}", "--set", f"run.experiments_dir={(cls.tmp / 'experiments').as_posix()}"]
        cls.code, cls.out, cls.err = run_cli("smoke", "--full", *cls.args)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, onexc=lambda func, path, exc: (os.chmod(path, stat.S_IWRITE), func(path)))  # registry entries are read-only
        if cls.tmp.exists():
            cls._tmp.cleanup()

    def summary(self) -> dict[str, Any]:
        return json.loads(self.out)

    def test_smoke_full_runs_the_whole_chain_and_proves_the_one_time_test_look_and_the_registry(self) -> None:
        self.assertEqual(self.code, 0, self.err)
        s = self.summary()
        self.assertEqual((s["status"], s["exit_code"], s["verdict"]), ("completed", 0, "winner"))  # the smoke run lowers select.min_verdict so that a finalist wins
        self.assertEqual(list(s["stages"]), ["prepare", "evaluate", "search", "finalists", "select_and_register"])
        self.assertTrue(all(r["status"] == "ok" for r in s["stages"].values()))
        self.assertEqual(s["full_chain"]["entry"], "model_0001")
        winner = s["stages"]["select_and_register"]["summary"]["winner_trial"]
        self.assertEqual(sorted(s["test_looked_at"]), sorted(["baseline", f"trial_{winner}"]))  # the Test: the winner and the baseline, once each
        self.assertIn("12 files match their checksums", s["registry_verify"])
        self.assertEqual(s["test_once_rule"], "refused")  # the evaluate stage still refuses the Test split
        self.assertIn("VERDICT: winner", self.err)

    def test_the_data_is_synthetic_and_large_enough_and_the_registry_is_outside_the_experiments(self) -> None:
        manifest = (self.tmp / "smoke" / "manifest.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(manifest) - 1, FULL_MIN_SAMPLES)  # 60 clips, so that Val and Test have 6 groups each
        self.assertTrue((self.tmp / "smoke_registry" / "model_0001" / "model" / "config.json").is_file())
        self.assertFalse((self.tmp / "experiments" / "smoke_registry").exists())
        text = (self.tmp / "experiments" / "exp_0001" / "reports" / "test_result_baseline.json").read_text(encoding="utf-8")
        self.assertIn("smoke_", text)  # the Test clips are the synthetic ones (audio/smoke_NNN.wav): no real data was read

    def test_a_rerun_of_the_same_experiment_skips_every_stage_and_the_checks_still_hold(self) -> None:
        code, out, err = run_cli("smoke", "--full", "--experiment", "exp_0001", *self.args)
        self.assertEqual(code, 0, err)
        s = json.loads(out)
        self.assertEqual({r["status"] for r in s["stages"].values()}, {"skipped"})
        self.assertEqual(len(s["test_looked_at"]), 2)  # no third look

    def test_a_wrong_test_look_makes_smoke_full_fail_with_exit_code_4(self) -> None:
        looked = self.tmp / "experiments" / "exp_0001" / "reports" / "test_once.json"
        original = looked.read_text(encoding="utf-8")
        try:
            looked.write_text(json.dumps({"looked": ["baseline", "baseline", "trial_9"]}), encoding="utf-8")
            code, out, err = run_cli("smoke", "--full", "--experiment", "exp_0001", *self.args)
            self.assertEqual(code, 4)
            self.assertIn("the Test look is wrong", json.loads(out)["message"])
        finally:
            looked.write_text(original, encoding="utf-8")

    def test_the_plain_smoke_command_is_unchanged_and_the_flag_is_documented(self) -> None:
        self.assertEqual(cli.PLANS["smoke"], ["prepare", "train", "evaluate"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            cli.main(["smoke", "--help"])
        self.assertIn("--full", out.getvalue())
        self.assertIn("python -m stt smoke", out.getvalue())

    def test_the_derived_values_are_a_small_grid_a_registry_next_to_the_data_and_a_lowered_bar(self) -> None:
        values = full_overrides(SmokeConfig(work_dir=Path("data/smoke")))
        self.assertIn("registry.dir=data/smoke_registry", values)
        self.assertIn("search.n_trials=4", values)
        self.assertIn("select.min_verdict=no_clear_difference", values)
        self.assertIn("search.screening.preset=smoke", values)




if __name__ == "__main__":
    unittest.main()
