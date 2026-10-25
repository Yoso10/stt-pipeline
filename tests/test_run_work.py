"""Tests of the work script `tools/run_work.ps1 -Step full` (DECISIONS 90): the dry plan, a real run on the tiny model with the encoder cache OFF and ON, the resume, and `-WithTest`.

How to run (project root, virtual environment active; needs Windows PowerShell):
    python -m unittest tests.test_run_work -v
Offline, synthetic clips and the stored tiny test model; each real run starts about eight programs and takes about 1.5 minutes. The tests are skipped where PowerShell or the project's
virtual environment is missing (for example a Linux machine: `tools/run_work.sh` is checked only with `bash -n`).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from stt.orchestrator.smoke import SmokeConfig, write_smoke_data  # noqa: E402

TINY = ROOT / "tests" / "assets" / "tiny_whisper"
SCRIPT = ROOT / "tools" / "run_work.ps1"
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
VENV = ROOT / ".venv" / "Scripts" / "python.exe"
needs_windows = unittest.skipUnless(POWERSHELL and VENV.is_file(), "needs PowerShell and the project's .venv")


def write_config(work: Path, cache: bool) -> Path:
    """A config for the tiny experiment that extends the shipped work_cpu.yaml (so the real settings are tested) and fills the 4 values."""
    text = f"""extends: {(ROOT / 'configs' / 'work_cpu.yaml').as_posix()}
data:
  source: {{path: "{(work / 'smoke').as_posix()}"}}
  columns: {{audio: audio, text: text, group: group}}
model:
  path: {TINY.as_posix()}
run:
  experiments_dir: {(work / 'e').as_posix()}
registry:
  dir: {(work / 'r').as_posix()}
split:
  min_samples_per_split: 1
  min_groups_per_split: 1
  ratio_tolerance: 0.1
train:
  hyperparams: {{learning_rate: 1.0e-4}}
eval:
  review: {{n_random: 2, n_worst: 2}}
features:
  encoder_cache: {{enabled: {'true' if cache else 'false'}, dir: "{(work / 'ec').as_posix()}", dtype: float32}}
"""
    path = work / "tiny_work.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def run_script(work: Path, config: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    command = [POWERSHELL or "", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT), "-Step", "full", "-Config", str(config), "-ExperimentsDir", str(work / "e"), "-ModelPath", str(TINY), "-Threads", "2", *extra]
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1500)


@needs_windows
class DryRunTests(unittest.TestCase):
    def test_the_dry_plan_lists_the_steps_in_order_and_computes_the_numbers_without_running_anything(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            done = run_script(work, Path("configs") / "work_cpu.yaml", "-Dry", "-WithTest", "-Epochs", "3")
            self.assertEqual(done.returncode, 0, done.stderr)
            out = done.stdout
            order = ["tools\\probe_env.py", "-m stt check", "-m stt prepare", "-m stt evaluate --config configs\\work_cpu.yaml --set", "-m stt train", "--experiment <train experiment>", "tools\\compare_evals.py", "-m stt final-test", "-m stt report"]
            positions = [out.index(item) for item in order]
            self.assertEqual(positions, sorted(positions))
            self.assertIn("3 epochs = 150 steps", out)  # 800 assumed clips / 16 = 50 steps per epoch
            self.assertIn("a Val-loss point every 37 steps", out)  # max(1, 150 // 4)
            self.assertIn("ESTIMATED TIME", out)
            self.assertNotIn("final-test", run_script(work, Path("configs") / "work_cpu.yaml", "-Dry").stdout)  # not without -WithTest
            self.assertFalse((work / "e").exists())  # a dry run writes nothing


@needs_windows
class RealRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def full_run(self, name: str, cache: bool) -> tuple[Path, subprocess.CompletedProcess[str]]:
        work = self.root / name
        work.mkdir()
        write_smoke_data(SmokeConfig(n_samples=24), 42, work / "smoke")
        config = write_config(work, cache)
        return work, run_script(work, config)

    def test_1_the_cache_off_and_the_cache_on_give_the_same_trained_numbers_and_a_report(self) -> None:
        off, done_off = self.full_run("off", cache=False)
        self.assertEqual(done_off.returncode, 0, done_off.stdout[-2500:] + done_off.stderr[-1500:])
        on, done_on = self.full_run("on", cache=True)
        self.assertEqual(done_on.returncode, 0, done_on.stdout[-2500:] + done_on.stderr[-1500:])
        for work, done in ((off, done_off), (on, done_on)):
            state = json.loads((work / "e" / "full_run_state.json").read_text(encoding="utf-8"))
            self.assertEqual((state["epochs"], state["evaluated"]), (2, True))
            self.assertGreater(state["max_steps"], 0)
            report = work / "e" / state["train"] / "reports" / "report.html"
            self.assertTrue(report.is_file())
            html = report.read_text(encoding="utf-8")
            self.assertIn("Training: did it converge", html)
            self.assertNotIn("FINAL - Test", html)  # no -WithTest: the Test stays closed
            self.assertIn("OPEN THE REPORT", done.stdout)
            self.assertIn("Relative reduction of the WER", done.stdout)
            self.assertFalse((work / "e" / "test_split_scored.json").exists())
        trained = [json.loads((w / "e" / json.loads((w / "e" / "full_run_state.json").read_text(encoding="utf-8"))["train"] / "reports" / "evaluation_trained.json").read_text(encoding="utf-8")) for w in (off, on)]
        self.assertEqual((trained[0]["wer"], trained[0]["cer"]), (trained[1]["wer"], trained[1]["cer"]))  # the cache changes nothing (float32 storage)
        self.assertTrue(any((on / "ec").glob("enc_*")))
        self.assertFalse((off / "ec").exists())

    def test_2_the_same_command_again_resumes_and_does_not_start_new_experiments(self) -> None:
        work = self.root / "on"
        if not (work / "e" / "full_run_state.json").exists():
            self.skipTest("test_1 did not run")
        before = sorted(p.name for p in (work / "e").iterdir() if p.is_dir())
        done = run_script(work, work / "tiny_work.yaml")
        self.assertEqual(done.returncode, 0, done.stdout[-2000:])
        self.assertIn("CONTINUES it", done.stdout)
        self.assertEqual(sorted(p.name for p in (work / "e").iterdir() if p.is_dir()), before)

    def test_3_with_test_scores_the_test_once_and_the_report_shows_it(self) -> None:
        work = self.root / "on"
        if not (work / "e" / "full_run_state.json").exists():
            self.skipTest("test_1 did not run")
        done = run_script(work, work / "tiny_work.yaml", "-WithTest")
        self.assertEqual(done.returncode, 0, done.stdout[-2500:] + done.stderr[-1500:])
        state = json.loads((work / "e" / "full_run_state.json").read_text(encoding="utf-8"))
        self.assertTrue(state["tested"])
        self.assertTrue((work / "e" / state["train"] / "reports" / "test_result.json").is_file())
        html = (work / "e" / state["train"] / "reports" / "report.html").read_text(encoding="utf-8")
        self.assertIn("FINAL - Test, scored once", html)
        again = run_script(work, work / "tiny_work.yaml", "-WithTest")  # the state says it was done: nothing is scored again, nothing is refused
        self.assertEqual(again.returncode, 0, again.stdout[-1500:])


class BashTwinTests(unittest.TestCase):
    def test_the_linux_script_has_valid_syntax(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("no bash")
        done = subprocess.run([bash, "-n", str(ROOT / "tools" / "run_work.sh")], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == "__main__":
    unittest.main()
