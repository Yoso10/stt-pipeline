"""Tests of tools/make_manifest.py (Wave 3 task 6): the manifest verifies on an unchanged tree and fails on a changed or missing file, is deterministic, and `check` finds it.

How to run (project root, virtual environment active):
    python -m unittest tests.test_make_manifest -v
The tests build a small fake release folder in a temporary folder. Nothing is downloaded and nothing outside the temporary folder is touched.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import make_manifest  # noqa: E402

from stt.orchestrator.check import run_checks  # noqa: E402
from stt.orchestrator.codeversion import MANIFEST_NAME, code_version, manifest_hash  # noqa: E402

LINE = re.compile(r"^[0-9a-f]{64}  \S.*$")


def run_tool(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = make_manifest.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class ManifestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "release"
        self.root.mkdir()
        for name, data in {
            "src/stt/a.py": b"print('a')\n",
            "src/stt/data/inner.py": b"x = 1\n",  # a folder named data below the root is part of the code
            "tools/t.py": b"t\n",
            "configs/work.yaml": b"paths: here\n",
            "tests/assets/model.bin": bytes(range(256)),
            "readme.txt": b"line1\r\nline2\r\n",
        }.items():
            self.put(name, data)
        self.manifest = self.root / MANIFEST_NAME

    def put(self, relative: str, data: bytes) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def write(self, *extra: str) -> str:
        code, _, err = run_tool("--root", str(self.root), "--out", str(self.manifest), *extra)
        self.assertEqual(code, 0, err)
        return self.manifest.read_text(encoding="utf-8")


class WriteTests(ManifestCase):
    def test_the_manifest_lists_every_release_file_sorted_in_the_sha256sum_format(self) -> None:
        text = self.write()
        lines = text.splitlines()
        self.assertTrue(all(LINE.match(line) for line in lines), lines)
        self.assertEqual([line[66:] for line in lines], sorted(line[66:] for line in lines))
        self.assertEqual([line[66:] for line in lines], ["configs/work.yaml", "readme.txt", "src/stt/a.py", "src/stt/data/inner.py", "tests/assets/model.bin", "tools/t.py"])
        self.assertTrue(text.endswith("\n") and "\r" not in text)
        self.assertIn(hashlib.sha256(b"line1\r\nline2\r\n").hexdigest() + "  readme.txt", text)  # the RAW bytes (CRLF kept) are hashed

    def test_the_same_files_give_the_same_manifest_byte_for_byte_whatever_the_creation_order(self) -> None:
        first = self.write()
        other = Path(self._tmp.name) / "other"
        other.mkdir()
        for relative in reversed(make_manifest.collect_files(self.root)):
            target = other / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((self.root / relative).read_bytes())
        self.assertEqual(make_manifest.build_manifest(other), first)  # a copy in another folder and another order
        self.assertEqual(self.write(), first)  # a second run, with the first manifest in place (it does not list itself)

    def test_local_and_runtime_things_are_left_out_and_exclude_patterns_work(self) -> None:
        for relative in (".git/config", ".venv/lib.py", "data/raw.wav", "models/m.bin", "wheels/w.whl", "experiments/exp_0001/x", "registry/model_0001/x", "src/stt/__pycache__/a.pyc",
                         "src/x.egg-info/PKG", ".pytest_cache/v", "src/stt/b.pyc", ".env", "notes.log"):
            self.put(relative, b"x")
        names = [line[66:] for line in self.write().splitlines()]
        self.assertEqual(names, ["configs/work.yaml", "notes.log", "readme.txt", "src/stt/a.py", "src/stt/data/inner.py", "tests/assets/model.bin", "tools/t.py"])
        names = [line[66:] for line in self.write("--exclude", "configs/work.yaml", "--exclude", "*.log").splitlines()]
        self.assertNotIn("configs/work.yaml", names)
        self.assertNotIn("notes.log", names)

    def test_writing_leaves_no_temporary_file_and_replaces_an_old_manifest(self) -> None:
        self.manifest.write_text("old\n", encoding="utf-8")
        self.write()
        self.assertEqual(sorted(p.name for p in self.root.iterdir() if p.name.startswith("MANIFEST")), [MANIFEST_NAME])
        self.assertNotIn("old", self.manifest.read_text(encoding="utf-8"))

    def test_a_missing_folder_is_exit_code_2(self) -> None:
        code, _, err = run_tool("--root", str(self.root / "nope"))
        self.assertEqual(code, 2)
        self.assertIn("does not exist", err)


class VerifyTests(ManifestCase):
    def verify(self) -> tuple[int, str]:
        code, out, _ = run_tool("--root", str(self.root), "--verify", str(self.manifest))
        return code, out

    def test_it_verifies_an_unchanged_tree(self) -> None:
        self.write()
        code, out = self.verify()
        self.assertEqual(code, 0)
        self.assertIn("VERIFIED: 6 files match", out)

    def test_it_fails_on_a_changed_file_and_names_it(self) -> None:
        self.write()
        self.put("src/stt/a.py", b"print('A')\n")
        code, out = self.verify()
        self.assertEqual(code, 1)
        self.assertIn("CHANGED  src/stt/a.py", out)
        self.assertIn("VERIFY FAILED: 1 changed, 0 missing (5 files match)", out)

    def test_a_single_changed_byte_in_a_binary_asset_and_a_line_ending_change_fail(self) -> None:
        self.write()
        self.put("tests/assets/model.bin", bytes(range(255)) + b"\xfe")
        self.put("readme.txt", b"line1\nline2\n")
        code, out = self.verify()
        self.assertEqual(code, 1)
        self.assertIn("CHANGED  tests/assets/model.bin", out)
        self.assertIn("CHANGED  readme.txt", out)

    def test_it_fails_on_a_missing_file_but_an_extra_file_is_only_listed(self) -> None:
        self.write()
        self.put("src/stt/extra.py", b"new\n")
        code, out = self.verify()
        self.assertEqual(code, 0)
        self.assertIn("EXTRA    src/stt/extra.py", out)
        self.assertIn("1 extra file(s)", out)
        (self.root / "tools" / "t.py").unlink()
        code, out = self.verify()
        self.assertEqual(code, 1)
        self.assertIn("MISSING  tools/t.py", out)

    def test_the_default_manifest_is_the_one_in_the_root(self) -> None:
        self.write()
        code, out, _ = run_tool("--root", str(self.root), "--verify")
        self.assertEqual((code, "VERIFIED" in out), (0, True))

    def test_a_missing_damaged_or_unsafe_manifest_is_exit_code_2_and_nothing_is_read_outside_the_folder(self) -> None:
        code, _, err = run_tool("--root", str(self.root), "--verify", str(self.root / "none.sha256"))
        self.assertEqual(code, 2)
        self.assertIn("does not exist", err)
        good = "a" * 64
        for text, message in ((f"{good} src/stt/a.py\n", "line 1 is not"), (f"{good}  ../outside.txt\n", "unsafe path"), (f"{good}  /etc/passwd\n", "unsafe path"), (f"{good}  C:/x.txt\n", "unsafe path"),
                              (f"{good}  a.py\n{good}  a.py\n", "twice"), ("not a manifest\n", "line 1 is not")):
            self.manifest.write_text(text, encoding="utf-8")
            code, _, err = run_tool("--root", str(self.root), "--verify", str(self.manifest))
            self.assertEqual(code, 2, text)
            self.assertIn(message, err)


class CheckFindsTheManifestTests(ManifestCase):
    def test_check_and_the_code_version_find_the_manifest_and_a_changed_file_changes_the_hash(self) -> None:
        self.assertIsNone(manifest_hash(self.root))
        before = code_version(self.root)
        self.write()
        suffix = manifest_hash(self.root)
        self.assertIsNotNone(suffix)
        self.assertEqual(code_version(self.root), f"{before}+{suffix}")
        lines = [c.text for c in run_checks(None, self.root, self.root) if "code version" in c.text]
        self.assertEqual(len(lines), 1)
        self.assertIn(f"+{suffix}", lines[0])
        self.assertNotIn("not there", lines[0])
        self.put("src/stt/a.py", b"print('changed')\n")
        self.write()
        self.assertNotEqual(manifest_hash(self.root), suffix)  # a new manifest of changed code gives a new version suffix


class RuleTests(unittest.TestCase):
    def test_no_pipeline_module_imports_anything_from_tools(self) -> None:
        offenders = []
        for path in sorted((ROOT / "src").rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                if any(n == "tools" or n.startswith("tools.") or n == "make_manifest" for n in names):
                    offenders.append(path.name)
        self.assertEqual(offenders, [])

    def test_the_tool_uses_only_the_standard_library_and_starts_with_the_how_to_run_header(self) -> None:
        tree = ast.parse((ROOT / "tools" / "make_manifest.py").read_text(encoding="utf-8"))
        imported = {(n.module or "").split(".")[0] if isinstance(n, ast.ImportFrom) else a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in (n.names if isinstance(n, ast.Import) else [n])}
        self.assertLessEqual(imported, set(sys.stdlib_module_names) | {"__future__"})
        self.assertIn("How to run", ast.get_docstring(tree) or "")

    def test_the_real_project_tree_gives_a_manifest_that_verifies_and_has_no_local_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "m.sha256"
            self.assertEqual(run_tool("--root", str(ROOT), "--out", str(out))[0], 0)
            names = [line[66:] for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertIn("tools/make_manifest.py", names)
            self.assertIn("src/stt/orchestrator/cli.py", names)
            self.assertFalse([n for n in names if n.split("/")[0] in (".git", ".venv", "data", "models", "wheels", "experiments", "registry") or "__pycache__" in n])
            self.assertEqual(run_tool("--root", str(ROOT), "--verify", str(out))[0], 0)


if __name__ == "__main__":
    unittest.main()
