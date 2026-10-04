"""Unit tests of tools/fetch_wheels.py with a fake command runner (no network, no real pip).

How to run (project root, virtual environment active):
    python -m unittest tests.test_fetch_wheels -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import fetch_wheels  # noqa: E402
from _common import FetchError  # noqa: E402
from fetch_wheels import OfflineInstallError, WheelMissingError  # noqa: E402


class FakeRunner:
    """Pretends to be pip/venv. Records every command. `fail_on` maps a keyword to (code, text)."""

    def __init__(self, fail_on: dict[str, tuple[int, str]] | None = None, version: str = "3.12") -> None:
        self.calls: list[list[str]] = []
        self.fail_on = fail_on or {}
        self.version = version

    def __call__(self, cmd: Sequence[str]) -> tuple[int, str]:
        cmd = list(cmd)
        self.calls.append(cmd)
        for key, result in self.fail_on.items():
            if key in cmd:
                return result
        if "-c" in cmd:
            return 0, self.version + "\n"
        if "download" in cmd:
            dest = Path(cmd[cmd.index("--dest") + 1])
            (dest / "pkg_a-1.0-py3-none-any.whl").write_bytes(b"w")
            (dest / "pkg_b-2.0-cp312-cp312-win_amd64.whl").write_bytes(b"w")
            return 0, ""
        if "freeze" in cmd:
            return 0, "pkg_b==2.0\npkg_a==1.0\n"
        return 0, ""


class FetchWheelsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.req = self.tmp / "requirements.txt"
        self.req.write_text("pkg_a==1.0\npkg_b==2.0\n", encoding="utf-8")
        self.out = self.tmp / "wheels"
        self.lock = self.tmp / "requirements.lock"

    def run_tool(self, runner: FakeRunner) -> fetch_wheels.WheelsResult:
        return fetch_wheels.fetch_wheels(self.req, self.out, self.lock, runner=runner)

    def test_produces_folder_and_sorted_lock_file(self) -> None:
        result = self.run_tool(FakeRunner())
        self.assertEqual(result.wheel_count, 2)
        self.assertFalse(result.reused)
        self.assertEqual(self.lock.read_text(encoding="utf-8"), "pkg_a==1.0\npkg_b==2.0\n")
        self.assertTrue((self.out / "fetch_info.json").is_file())
        self.assertFalse(self.out.with_name("wheels.partial").exists())

    def test_download_command_targets_python_and_windows_binary_only(self) -> None:
        runner = FakeRunner()
        self.run_tool(runner)
        download = next(c for c in runner.calls if "download" in c)
        for expected in ("--python-version", "3.12", "--platform", "win_amd64", "--only-binary=:all:", "cp312"):
            self.assertIn(expected, download)

    def test_offline_install_uses_no_index_and_find_links(self) -> None:
        runner = FakeRunner()
        self.run_tool(runner)
        install = next(c for c in runner.calls if "install" in c)
        self.assertIn("--no-index", install)
        self.assertIn("--find-links", install)

    def test_second_run_reuses_wheels_but_retests_install(self) -> None:
        self.run_tool(FakeRunner())
        runner = FakeRunner()
        result = self.run_tool(runner)
        self.assertTrue(result.reused)
        self.assertFalse(any("download" in c for c in runner.calls))
        self.assertTrue(any("install" in c for c in runner.calls))

    def test_changed_requirements_trigger_new_download(self) -> None:
        self.run_tool(FakeRunner())
        self.req.write_text("pkg_a==1.1\n", encoding="utf-8")
        runner = FakeRunner()
        self.assertFalse(self.run_tool(runner).reused)
        self.assertTrue(any("download" in c for c in runner.calls))

    def test_missing_wheel_fails_naming_package_and_python(self) -> None:
        text = "ERROR: No matching distribution found for sometool==3.1\n"
        with self.assertRaises(WheelMissingError) as ctx:
            self.run_tool(FakeRunner({"download": (1, text)}))
        self.assertIn("sometool==3.1", str(ctx.exception))
        self.assertIn("3.12", str(ctx.exception))
        self.assertFalse(self.out.exists())
        self.assertFalse(self.lock.exists())

    def test_offline_install_failure_is_reported(self) -> None:
        text = "ERROR: Could not find a version that satisfies the requirement dep_x\n"
        with self.assertRaises(OfflineInstallError) as ctx:
            self.run_tool(FakeRunner({"--no-index": (1, text)}))
        self.assertIn("dep_x", str(ctx.exception))
        self.assertFalse(self.lock.exists())

    def test_pip_check_problem_is_reported(self) -> None:
        with self.assertRaises(OfflineInstallError):
            self.run_tool(FakeRunner({"check": (1, "pkg_a requires zzz")}))

    def test_wrong_python_version_fails_before_any_download(self) -> None:
        runner = FakeRunner(version="3.14")
        with self.assertRaises(FetchError) as ctx:
            self.run_tool(runner)
        self.assertIn("3.14", str(ctx.exception))
        self.assertFalse(any("download" in c for c in runner.calls))

    def test_missing_requirements_file_fails(self) -> None:
        self.req.unlink()
        with self.assertRaises(FetchError):
            self.run_tool(FakeRunner())

    def test_main_exit_codes(self) -> None:
        import shutil

        args = ["--requirements", str(self.req), "--out", str(self.out), "--lock", str(self.lock)]
        self.assertEqual(fetch_wheels.main(args, FakeRunner()), 0)
        shutil.rmtree(self.out)
        self.assertEqual(fetch_wheels.main(args, FakeRunner({"download": (1, "No matching distribution found for q")})), 2)
        self.assertEqual(fetch_wheels.main(args, FakeRunner({"--no-index": (1, "boom")})), 3)


if __name__ == "__main__":
    unittest.main()
