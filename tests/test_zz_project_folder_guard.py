"""Guard test (O108 follow-up, DECISIONS 75): no test may create, change or remove anything in the project folder (the project root, ./experiments, ./registry, docs, configs, tools, data/smoke).

How to run (project root, virtual environment active):
    python -m unittest discover -s tests          # the normal suite: this module sorts LAST (test_zz_...), so it sees what every other test did
    python -m unittest tests.test_zz_project_folder_guard -v    # alone: it only checks its own helper (nothing ran in between)
How it works: unittest imports EVERY test module before it runs any test, so the snapshot below (taken when this module is imported) is the state before the suite. The last test compares it with the state after the
suite. The snapshot only READS (names, sizes and modification times; the contents of no file), so it is cheap and writes nothing. A test that needs files must use a temporary folder (tempfile).
Left out of the snapshot: `.git`, `.venv`, `__pycache__`, `.pytest_cache`, `.mypy_cache`, the models and wheels folders' contents, and the contents of `data/` below its first level (only the names directly in `data/`
are watched, so a test that creates `data/smoke` or any other folder there is caught).
If it fails: the message lists the paths. Find the test that runs a command (or a stage) without `--set run.experiments_dir=<tempfolder>` (and `registry.dir`, `smoke.work_dir`, `--out`), and give it a temporary folder.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_NAMES = frozenset({".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"})
WATCHED_TREES = ("experiments", "registry", "docs", "configs", "tools", "data/smoke")
Snapshot = dict[str, tuple[str, int, int]]


def snapshot(root: Path) -> Snapshot:
    """What is in the project folder: every entry directly in `root` (a folder only by name), every file under the watched trees (path, size, modification time), the names directly in `data/`. Reads only."""
    found: Snapshot = {}
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.name in SKIP_NAMES:
                continue
            if entry.is_dir(follow_symlinks=False):
                found[entry.name + "/"] = ("dir", 0, 0)
            else:
                info = entry.stat(follow_symlinks=False)
                found[entry.name] = ("file", info.st_size, info.st_mtime_ns)
    data = root / "data"
    if data.is_dir():
        with os.scandir(data) as entries:
            for entry in entries:
                found[f"data/{entry.name}" + ("/" if entry.is_dir(follow_symlinks=False) else "")] = ("name", 0, 0)
    for tree in WATCHED_TREES:
        base = root / tree
        if not base.is_dir():
            continue
        for folder, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_NAMES)
            here = Path(folder)
            for name in dirs:
                found[(here / name).relative_to(root).as_posix() + "/"] = ("dir", 0, 0)
            for name in files:
                info = (here / name).stat()
                found[(here / name).relative_to(root).as_posix()] = ("file", info.st_size, info.st_mtime_ns)
    return found


def differences(before: Snapshot, after: Snapshot) -> list[str]:
    """The paths created, changed or removed between two snapshots, as readable lines."""
    lines = [f"CREATED  {name}" for name in sorted(set(after) - set(before))]
    lines += [f"CHANGED  {name}" for name in sorted(n for n in before if n in after and before[n] != after[n])]
    lines += [f"REMOVED  {name}" for name in sorted(set(before) - set(after))]
    return lines


BEFORE_THE_SUITE = snapshot(ROOT)  # taken at import: unittest imports every test module before it runs a test


class SnapshotHelperTests(unittest.TestCase):
    """The guard's own helper, on a temporary folder (so it proves that the guard can see a create, a change and a removal)."""

    def make_root(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for name in ("experiments/exp_0001/status.json", "docs/a.md", "data/smoke/m.csv", "notes.txt", ".git/config", "src/__pycache__/x.pyc"):
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text("x", encoding="utf-8")
        return root

    def test_an_unchanged_folder_has_no_differences_and_git_and_caches_are_ignored(self) -> None:
        root = self.make_root()
        first = snapshot(root)
        (root / ".git" / "config").write_text("changed", encoding="utf-8")
        (root / "src" / "__pycache__" / "y.pyc").write_text("new", encoding="utf-8")
        self.assertEqual(differences(first, snapshot(root)), [])

    def test_a_created_a_changed_and_a_removed_file_are_found_in_the_root_and_in_experiments(self) -> None:
        root = self.make_root()
        first = snapshot(root)
        (root / "cleanup_proposal_x.txt").write_text("stray", encoding="utf-8")  # the incident of Wave 3 task 6
        (root / "experiments" / "exp_0001" / "status.json").write_text("changed, longer", encoding="utf-8")
        (root / "docs" / "a.md").unlink()
        (root / "experiments" / "exp_0002").mkdir()
        (root / "data" / "part99").mkdir()
        found = differences(first, snapshot(root))
        self.assertEqual(
            found, ["CREATED  cleanup_proposal_x.txt", "CREATED  data/part99/", "CREATED  experiments/exp_0002/", "CHANGED  experiments/exp_0001/status.json", "REMOVED  docs/a.md"]
        )

    def test_it_reads_only_and_is_cheap(self) -> None:
        root = self.make_root()
        before = {p.relative_to(root).as_posix(): p.stat().st_mtime_ns for p in root.rglob("*")}
        snapshot(root)
        self.assertEqual({p.relative_to(root).as_posix(): p.stat().st_mtime_ns for p in root.rglob("*")}, before)


class ProjectFolderGuardTests(unittest.TestCase):
    def test_zz_nothing_in_the_project_folder_was_created_changed_or_removed_by_the_suite(self) -> None:
        found = differences(BEFORE_THE_SUITE, snapshot(ROOT))
        self.assertEqual(found, [], "a test wrote into the project folder (use a temporary folder):\n" + "\n".join(found[:30]))


if __name__ == "__main__":
    unittest.main()
