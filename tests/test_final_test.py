"""Tests of the `final-test` command (DECISIONS 90): the Test split is scored ONCE for the trained model and the baseline, refused a second time, refused without a trained model, the same
with the encoder cache on and off, and shown in the report only when `test_result.json` exists.

How to run (project root, virtual environment active):
    python -m unittest tests.test_final_test -v
Offline, on synthetic clips and the stored tiny test model (about 30 seconds per experiment).
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import tiny_flow  # noqa: E402
from stt.orchestrator.finaltest import MARKER_FILE, RESULT_FILE  # noqa: E402


def read_result(work: Path, experiment: str) -> dict:
    return json.loads((work / "e" / experiment / "reports" / RESULT_FILE).read_text(encoding="utf-8"))


class UsageTests(unittest.TestCase):
    def test_an_experiment_is_required(self) -> None:
        code, out, err = tiny_flow.run_cli("final-test", "--config", str(tiny_flow.SMOKE_YAML))
        self.assertEqual(code, 2)
        self.assertIn("--experiment", err)

    def test_without_a_trained_model_it_refuses_and_nothing_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            from stt.orchestrator.smoke import SmokeConfig, write_smoke_data

            write_smoke_data(SmokeConfig(n_samples=24), 42, work / "smoke")
            base = tiny_flow.base_args(work)
            self.assertEqual(tiny_flow.run_cli("prepare", *base)[0], 0)
            code, out, err = tiny_flow.run_cli("final-test", "--experiment", "exp_0001", *base)
            self.assertEqual(code, 2)
            self.assertIn("no trained model", err)
            self.assertFalse((work / "e" / MARKER_FILE).exists())
            self.assertFalse((work / "e" / "exp_0001" / "reports" / RESULT_FILE).exists())


class FlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._off, cls._on = tempfile.TemporaryDirectory(), tempfile.TemporaryDirectory()
        cls.off, cls.on = Path(cls._off.name), Path(cls._on.name)
        cls.base_off, cls.exp_off = tiny_flow.make_experiment(cls.off, cache=False)
        cls.base_on, cls.exp_on = tiny_flow.make_experiment(cls.on, cache=True, dtype="float32")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._off.cleanup()
        cls._on.cleanup()

    def test_1_scores_the_test_once_and_writes_the_result_with_the_status(self) -> None:
        reports = self.off / "e" / self.exp_off / "reports"
        self.assertFalse((reports / RESULT_FILE).exists())  # train and evaluate never touch the Test
        code, out, err = tiny_flow.run_cli("final-test", "--experiment", self.exp_off, *self.base_off)
        self.assertEqual(code, 0, err[-1500:])
        result = read_result(self.off, self.exp_off)
        self.assertEqual(result["split"], "test")
        self.assertIn(result["status"], ("confirmed", "not_confirmed"))
        self.assertEqual(result["status"] == "confirmed", result["verdict"] == "improved")  # the Selector's rule
        for key in ("baseline", "trained"):
            self.assertTrue({"wer", "cer", "n_files", "n_words"} <= set(result[key]))
        expected = (result["baseline"]["wer"] - result["trained"]["wer"]) / result["baseline"]["wer"]
        self.assertAlmostEqual(result["relative_reduction"], expected)
        self.assertTrue((self.off / "e" / MARKER_FILE).is_file())  # the marker is written before the scoring

    def test_2_a_second_call_is_refused_with_exit_code_3_and_changes_nothing(self) -> None:
        if not (self.off / "e" / self.exp_off / "reports" / RESULT_FILE).exists():
            tiny_flow.run_cli("final-test", "--experiment", self.exp_off, *self.base_off)
        before = (self.off / "e" / self.exp_off / "reports" / RESULT_FILE).read_text(encoding="utf-8")
        code, out, err = tiny_flow.run_cli("final-test", "--experiment", self.exp_off, *self.base_off)
        self.assertEqual(code, 3)
        self.assertIn("REFUSED", err)
        self.assertIn("already scored", err)
        self.assertEqual((self.off / "e" / self.exp_off / "reports" / RESULT_FILE).read_text(encoding="utf-8"), before)

    def test_3_the_marker_also_refuses_a_new_experiment_in_the_same_folder(self) -> None:
        if not (self.off / "e" / MARKER_FILE).exists():
            tiny_flow.run_cli("final-test", "--experiment", self.exp_off, *self.base_off)
        code, out, err = tiny_flow.run_cli("train", *self.base_off)
        self.assertEqual(code, 0, err[-800:])
        newest = sorted(p.name for p in (self.off / "e").iterdir() if p.is_dir())[-1]
        code, out, err = tiny_flow.run_cli("evaluate", "--experiment", newest, *self.base_off)
        self.assertEqual(code, 0, err[-800:])
        code, out, err = tiny_flow.run_cli("final-test", "--experiment", newest, *self.base_off)
        self.assertEqual(code, 3)
        self.assertIn(MARKER_FILE, err)
        self.assertFalse((self.off / "e" / newest / "reports" / RESULT_FILE).exists())

    def test_4_the_cache_on_gives_the_same_test_result_as_the_cache_off(self) -> None:
        code, out, err = tiny_flow.run_cli("final-test", "--experiment", self.exp_on, *self.base_on)
        self.assertEqual(code, 0, err[-1500:])
        if not (self.off / "e" / self.exp_off / "reports" / RESULT_FILE).exists():
            tiny_flow.run_cli("final-test", "--experiment", self.exp_off, *self.base_off)
        on, off = read_result(self.on, self.exp_on), read_result(self.off, self.exp_off)
        for key in ("baseline", "trained"):
            self.assertEqual((on[key]["wer"], on[key]["cer"], on[key]["n_files"], on[key]["n_words"]), (off[key]["wer"], off[key]["cer"], off[key]["n_files"], off[key]["n_words"]))
        self.assertEqual((on["verdict"], on["status"], on["relative_reduction"]), (off["verdict"], off["status"], off["relative_reduction"]))
        self.assertTrue(any((self.on / "ec").glob("enc_*")))  # the cache really was used for the Test clips too

    def test_5_the_report_shows_the_test_only_after_final_test(self) -> None:
        if not (self.off / "e" / self.exp_off / "reports" / RESULT_FILE).exists():
            tiny_flow.run_cli("final-test", "--experiment", self.exp_off, *self.base_off)
        code, out, err = tiny_flow.run_cli("report", "--experiment", self.exp_off, *self.base_off)
        self.assertEqual(code, 0, err)
        html = (self.off / "e" / self.exp_off / "reports" / "report.html").read_text(encoding="utf-8")
        self.assertIn("FINAL - Test, scored once", html)
        self.assertIn("Test: ", html)

    def test_6_a_test_look_of_the_run_command_also_blocks_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            base, experiment = tiny_flow.make_experiment(work, cache=False)
            other = work / "e" / "exp_0099" / "reports"
            other.mkdir(parents=True)
            (other / "test_once.json").write_text(json.dumps({"looked": ["baseline"]}), encoding="utf-8")
            code, out, err = tiny_flow.run_cli("final-test", "--experiment", experiment, *base)
            self.assertEqual(code, 3)
            self.assertIn("exp_0099", err)


if __name__ == "__main__":
    unittest.main()
