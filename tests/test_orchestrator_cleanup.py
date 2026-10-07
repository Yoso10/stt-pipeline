"""Tests of the `cleanup` command (Wave 3 task 6): it only PROPOSES, it deletes and changes nothing, and it never proposes the registry or an unfinished or locked experiment.

How to run (project root, virtual environment active):
    python -m unittest tests.test_orchestrator_cleanup -v
The tests build small fake experiments (real `ExperimentStore` folders with candidate and checkpoint folders of a few bytes) in a temporary folder. Nothing is downloaded or trained.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_orchestrator_curve import curve_config  # noqa: E402

from stt.orchestrator import cli  # noqa: E402
from stt.orchestrator.cleanupcmd import megabytes, propose_cleanup, render_proposal, write_proposal  # noqa: E402
from stt.orchestrator.errors import OrchestratorError  # noqa: E402
from stt.orchestrator.wiring import experiment_catalog  # noqa: E402
from stt.runmanager import ExperimentSpec, RunConfig  # noqa: E402

NOW = "2026-10-07T12:00:00"


def snapshot(folder: Path) -> dict[str, tuple[int, int, str]]:
    """Every file AND folder under `folder`: (size, modification time in ns, sha256 of the bytes; empty for a folder)."""
    result = {}
    for path in sorted(folder.rglob("*")):
        stat = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
        result[path.relative_to(folder).as_posix()] = (stat.st_size, stat.st_mtime_ns, digest)
    result["."] = (0, folder.stat().st_mtime_ns, "")
    return result


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class CleanupCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.config = curve_config(self.tmp)
        self.run_config = self.config.section("run", RunConfig)
        self.store = experiment_catalog(self.run_config)
        self.out_dir = self.tmp / "outside"  # the proposal files go here, away from the folders that are compared
        self.out_dir.mkdir()
        self.experiments = self.tmp / "experiments"
        self.registry = self.tmp / "registry"

    def experiment(self, status: str, created: str = "2026-09-01T09:00:00", trials: int = 2, lock: bool = False) -> str:
        """A fake experiment with `trials` candidate folders (a model file and a checkpoints folder each)."""
        handle = self.store.create(ExperimentSpec(self.config, None, "0.0.0", {}, created))
        for n in range(trials):
            folder = handle.candidate_dir(f"{handle.experiment_id}_trial_{n:03d}")
            (folder / "checkpoints" / "step_1").mkdir(parents=True)
            (folder / "model.safetensors").write_bytes(b"m" * 1000)
            (folder / "checkpoints" / "step_1" / "optimizer.pt").write_bytes(b"o" * 3000)
        handle.set_status(status, None)
        if lock:
            (self.experiments / handle.experiment_id / "run.lock").write_text("pid", encoding="utf-8")
        return handle.experiment_id

    def propose(self, **kwargs: object):  # type: ignore[no-untyped-def]
        return propose_cleanup(self.store, self.experiments, self.registry, kwargs.pop("now", NOW), **kwargs)  # type: ignore[arg-type]

    def cli(self, *extra: str) -> tuple[int, str, str]:
        return run_cli("cleanup", "--set", f"run.experiments_dir={self.experiments.as_posix()}", "--set", f"registry.dir={self.registry.as_posix()}", *extra)


class ProposalTests(CleanupCase):
    def test_a_completed_experiment_lists_its_checkpoints_and_candidate_folders_with_sizes(self) -> None:
        self.experiment("completed")
        proposal = self.propose()
        self.assertEqual(len(proposal.checkpoints), 2)
        self.assertEqual(len(proposal.candidates), 2)
        self.assertEqual([i.size_bytes for i in proposal.checkpoints], [3000, 3000])
        self.assertEqual([i.size_bytes for i in proposal.candidates], [4000, 4000])  # the candidate folder INCLUDES its checkpoints
        self.assertTrue(all(i.experiment_id == "exp_0001" and i.reason for i in (*proposal.checkpoints, *proposal.candidates)))
        self.assertEqual(proposal.deleted, 0)

    def test_an_experiment_that_is_not_finished_is_never_proposed_whatever_its_status(self) -> None:
        ids = {status: self.experiment(status) for status in ("created", "running", "stopped")}
        proposal = self.propose()
        self.assertEqual((proposal.checkpoints, proposal.candidates, proposal.failed), ((), (), ()))
        skipped = {s.what: s.why for s in proposal.skipped}
        for status, experiment_id in ids.items():
            self.assertIn(f"status is {status}", skipped[experiment_id])
        proposal = self.propose(include_failed=True)  # the flag adds FAILED ones only, never the unfinished
        self.assertEqual(proposal.failed, ())

    def test_an_experiment_with_a_lock_file_is_never_proposed_even_when_completed_or_failed(self) -> None:
        locked = self.experiment("completed", lock=True)
        locked_failed = self.experiment("failed", lock=True)
        free = self.experiment("completed")
        proposal = self.propose(include_failed=True)
        self.assertEqual({i.experiment_id for i in (*proposal.checkpoints, *proposal.candidates, *proposal.failed)}, {free})
        skipped = {s.what: s.why for s in proposal.skipped}
        self.assertIn("lock file", skipped[locked])
        self.assertIn("lock file", skipped[locked_failed])

    def test_failed_experiments_are_listed_only_with_the_flag(self) -> None:
        failed = self.experiment("failed")
        self.assertEqual(self.propose().failed, ())
        self.assertIn("--include-failed", {s.what: s.why for s in self.propose().skipped}[failed])
        proposal = self.propose(include_failed=True)
        self.assertEqual([(i.experiment_id, i.path.name) for i in proposal.failed], [(failed, failed)])
        self.assertEqual((proposal.checkpoints, proposal.candidates), ((), ()))  # a failed experiment gives no candidate or checkpoint proposals

    def test_nothing_inside_the_registry_or_containing_it_is_proposed(self) -> None:
        inside = self.experiment("completed")
        other = self.experiment("completed")
        self.registry = self.experiments / inside / "candidates" / f"{inside}_trial_000"  # a registry configured INSIDE a candidate folder
        proposal = self.propose()
        proposed = {str(i.path.resolve()) for i in (*proposal.checkpoints, *proposal.candidates)}
        self.assertFalse(any(str(self.registry.resolve()) in p or p in str(self.registry.resolve()) for p in proposed if p.startswith(str(self.experiments.resolve() / inside))))
        self.assertTrue(any(f"{other}" in p for p in proposed))  # the other experiment is still listed
        self.assertIn("lies inside it", {s.what: s.why for s in proposal.skipped}[inside])
        # a registry folder INSIDE the experiments folder but not in an experiment: not proposed either
        self.registry = self.experiments / "registry"
        (self.registry / "model_0001" / "model").mkdir(parents=True)
        (self.registry / "model_0001" / "model" / "model.safetensors").write_bytes(b"w")
        proposal = self.propose()
        self.assertFalse(any(self.registry.resolve() in (i.path.resolve(), *i.path.resolve().parents) for i in (*proposal.checkpoints, *proposal.candidates)))

    def test_the_age_filter_keeps_young_experiments_out_and_says_why(self) -> None:
        old = self.experiment("completed", created="2026-08-01T09:00:00")
        young = self.experiment("completed", created="2026-10-06T09:00:00")
        proposal = self.propose(older_than_days=30)
        self.assertEqual({i.experiment_id for i in proposal.candidates}, {old})
        self.assertIn("younger than 30 days", {s.what: s.why for s in proposal.skipped}[young])
        with self.assertRaises(OrchestratorError):
            self.propose(older_than_days=-1)

    def test_the_text_says_nothing_was_deleted_starts_with_how_to_run_and_does_not_add_the_nested_totals(self) -> None:
        self.experiment("completed")
        self.experiment("running")
        text = render_proposal(self.propose())
        self.assertTrue(text.startswith("HOW TO RUN: python -m stt cleanup"))
        self.assertIn("NOTHING WAS DELETED", text)
        self.assertIn("B INCLUDES A: do not add the two totals", text)
        self.assertIn("exp_0002: its status is running", text)
        self.assertTrue(text.rstrip().endswith("Deleted by this command: 0"))
        self.assertEqual((megabytes(999), megabytes(3_400_000), megabytes(1_800_000_000)), ("0 KB", "3.4 MB", "1.8 GB"))

    def test_the_source_of_the_command_has_no_call_that_deletes(self) -> None:
        source = (ROOT / "src" / "stt" / "orchestrator" / "cleanupcmd.py").read_text(encoding="utf-8")
        code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
        body = code.split('"""', 2)[2]  # without the module docstring
        for forbidden in (r"\brmtree\b", r"\.unlink\(", r"\bos\.remove\b", r"\bos\.rmdir\b", r"\.rmdir\(", r"\bshutil\.move\b", r"\.replace\(", r"\.rename\(", r"\.delete\("):
            self.assertIsNone(re.search(forbidden, body), forbidden)


class NothingChangesTests(CleanupCase):
    def build_world(self) -> None:
        self.experiment("completed")
        self.experiment("completed", lock=True)
        self.experiment("running")
        self.experiment("failed")
        (self.registry / "model_0001" / "model").mkdir(parents=True)
        (self.registry / "model_0001" / "model" / "model.safetensors").write_bytes(b"winner")
        (self.registry / "model_0001" / "SHA256SUMS").write_text("x  model/model.safetensors\n", encoding="utf-8")

    def test_the_command_changes_no_byte_and_no_timestamp_of_any_file_or_folder(self) -> None:
        self.build_world()
        before = snapshot(self.tmp / "experiments"), snapshot(self.registry)
        code, out, err = self.cli("--out", str(self.out_dir / "proposal.txt"), "--include-failed")
        self.assertEqual(code, 0, err)
        after = snapshot(self.tmp / "experiments"), snapshot(self.registry)
        self.assertEqual(before, after)  # every byte, every size, every modification time, files and folders
        body = json.loads(out)
        self.assertEqual((body["deleted"], body["exit_code"]), (0, 0))
        self.assertTrue((self.out_dir / "proposal.txt").is_file())
        text = (self.out_dir / "proposal.txt").read_text(encoding="utf-8")
        self.assertIn("exp_0001", text)
        self.assertNotIn("model_0001", text)  # the registry never appears as a candidate
        self.assertEqual(sorted(p.name for p in self.out_dir.iterdir()), ["proposal.txt"])  # the only thing written

    def test_the_default_proposal_file_is_next_to_the_experiments_folder_not_inside_it(self) -> None:
        self.build_world()
        before = snapshot(self.experiments)
        code, out, _ = self.cli()
        self.assertEqual(code, 0)
        written = Path(json.loads(out)["proposal_file"])
        self.assertEqual(written.parent.resolve(), self.tmp.resolve())
        self.assertTrue(re.fullmatch(r"cleanup_proposal_\d{8}T\d{6}\.txt", written.name), written.name)
        self.assertEqual(before, snapshot(self.experiments))  # the experiments folder is untouched

    def test_an_existing_proposal_file_is_never_overwritten(self) -> None:
        self.build_world()
        target = self.out_dir / "proposal.txt"
        target.write_text("old proposal", encoding="utf-8")
        stat = target.stat()
        code, out, err = self.cli("--out", str(target))
        self.assertEqual((code, out), (2, ""))
        self.assertIn("already exists and is not overwritten", err)
        self.assertEqual((target.read_text(encoding="utf-8"), target.stat().st_mtime_ns), ("old proposal", stat.st_mtime_ns))
        with self.assertRaises(OrchestratorError):
            write_proposal("x", self.out_dir / "no_such_folder" / "p.txt")

    def test_a_missing_experiments_folder_and_a_bad_age_are_exit_code_2(self) -> None:
        code, out, err = run_cli("cleanup", "--set", f"run.experiments_dir={(self.tmp / 'nope').as_posix()}")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("does not exist", err)
        self.build_world()
        self.assertEqual(self.cli("--older-than-days", "-3", "--out", str(self.out_dir / "p.txt"))[0], 2)
        self.assertFalse((self.out_dir / "p.txt").exists())

    def test_the_command_is_offline_has_help_and_an_example(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            cli.main(["cleanup", "--help"])
        for part in ("python -m stt cleanup", "Example:", "--older-than-days", "--include-failed", "NOTHING is deleted"):
            self.assertIn(part, out.getvalue())


if __name__ == "__main__":
    unittest.main()
