"""Checks that the stt package is importable from src/ without installation (offline).

How to run (project root, virtual environment active):
    python -m unittest tests.test_package_layout -v
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class PackageLayoutTests(unittest.TestCase):
    def test_import_in_process(self) -> None:
        import stt

        self.assertEqual(stt.__version__, "0.0.0")

    def test_import_in_fresh_process_with_pythonpath_only(self) -> None:
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        proc = subprocess.run(
            [sys.executable, "-c", "import stt; print(stt.__version__)"],
            capture_output=True, text=True, env=env, cwd=ROOT,
        )  # fmt: skip
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "0.0.0")


if __name__ == "__main__":
    unittest.main()
