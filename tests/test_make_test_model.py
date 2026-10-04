"""Unit tests of tools/make_test_model.py (offline): the stored tiny model is rebuilt from itself in a temporary folder.

How to run (project root, virtual environment active):
    python -m unittest tests.test_make_test_model -v
The source folder is the stored test model itself (tests/assets/tiny_whisper has the real tokenizer and extractor
files), so no downloaded model is needed and nothing touches the network.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import make_test_model  # noqa: E402
from _common import FetchError  # noqa: E402

TINY = ROOT / "tests" / "assets" / "tiny_whisper"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MakeTestModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_4_creates_a_model_that_loads_through_the_model_loader_and_runs_a_forward_pass(self) -> None:
        result = make_test_model.make_test_model(TINY, self.tmp / "out", seed=0)
        self.assertTrue((self.tmp / "out" / "model.safetensors").is_file())
        self.assertLess(int(result["bytes"]), 5_000_000)  # type: ignore[call-overload]
        self.assertGreater(int(result["parameters"]), 100_000)  # type: ignore[call-overload]
        self.assertTrue(0 < float(result["loss"]) < 100)  # type: ignore[arg-type]

    def test_same_seed_gives_the_same_weights_and_another_seed_different_ones(self) -> None:
        make_test_model.make_test_model(TINY, self.tmp / "a", seed=1)
        make_test_model.make_test_model(TINY, self.tmp / "b", seed=1)
        make_test_model.make_test_model(TINY, self.tmp / "c", seed=2)
        self.assertEqual(digest(self.tmp / "a" / "model.safetensors"), digest(self.tmp / "b" / "model.safetensors"))
        self.assertNotEqual(digest(self.tmp / "a" / "model.safetensors"), digest(self.tmp / "c" / "model.safetensors"))

    def test_tokenizer_and_extractor_files_are_copied_unchanged(self) -> None:
        make_test_model.make_test_model(TINY, self.tmp / "out", seed=0)
        for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
            self.assertEqual(digest(self.tmp / "out" / name), digest(TINY / name))

    def test_incomplete_source_fails_naming_the_missing_files(self) -> None:
        source = self.tmp / "src"
        shutil.copytree(TINY, source)
        (source / "tokenizer.json").unlink()
        (source / "config.json").unlink()
        with self.assertRaises(FetchError) as ctx:
            make_test_model.make_test_model(source, self.tmp / "out")
        message = str(ctx.exception)
        self.assertIn("tokenizer.json", message)
        self.assertIn("config.json", message)
        self.assertFalse((self.tmp / "out").exists())
        with self.assertRaises(FetchError):
            make_test_model.make_test_model(self.tmp / "nothing", self.tmp / "out")

    def test_existing_output_is_never_replaced_without_force(self) -> None:
        out = self.tmp / "out"
        make_test_model.make_test_model(TINY, out, seed=0)
        marker = out / "marker.txt"
        marker.write_text("kept", encoding="utf-8")
        with self.assertRaises(FetchError) as ctx:
            make_test_model.make_test_model(TINY, out, seed=0)
        self.assertIn("--force", str(ctx.exception))
        self.assertTrue(marker.exists())
        make_test_model.make_test_model(TINY, out, seed=0, force=True)
        self.assertFalse(marker.exists())

    def test_main_exit_codes(self) -> None:
        self.assertEqual(make_test_model.main(["--tokenizer-from", str(TINY), "--out", str(self.tmp / "ok")]), 0)
        self.assertEqual(make_test_model.main(["--tokenizer-from", str(self.tmp / "nope"), "--out", str(self.tmp / "bad")]), 1)

    def test_the_stored_test_model_is_small_enough_to_commit(self) -> None:
        total = sum(p.stat().st_size for p in TINY.rglob("*") if p.is_file())
        self.assertLess(total, 5_000_000)


if __name__ == "__main__":
    unittest.main()
