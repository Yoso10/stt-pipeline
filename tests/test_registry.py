"""Unit tests of the ModelRegistry component (stt.registry): the acceptance tests of the C10 brief, offline, in temporary folders with small fake model files.

How to run (project root, virtual environment active):
    python -m unittest tests.test_registry -v
The selections are made by the real Selector from fake evaluation results (tests/selector_fakes.py). The "model" is a folder of a few small random files. No real model,
no Test evaluation of a real model and no network are used.
"""

from __future__ import annotations

import ast
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from selector_fakes import FakeTest, candidate, make_result  # noqa: E402

from stt.config import SectionRegistry, load_config  # noqa: E402
from stt.evaluator import Comparator, EvalConfig  # noqa: E402
from stt.registry import (  # noqa: E402
    COLUMNS,
    ModelRegistry,
    RegisterRequest,
    RegistryConfig,
    RegistryError,
    RegistryVerifyError,
    card_from_json,
    card_from_selection,
    card_to_json,
    render_review,
    sha256_file,
    test_set_fingerprint,
)
from stt.selector import SelectConfig, SelectionRequest, Selector  # noqa: E402

COMPARATOR = Comparator(EvalConfig(), 42)


def tree_hashes(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): sha256_file(p) for p in sorted(folder.rglob("*")) if p.is_file()}


class RegistryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "registry"
        self.counter = 0

    def config(self, **kw: object) -> RegistryConfig:
        return RegistryConfig.model_validate({"dir": self.root, **kw})

    def registry(self, **kw: object) -> ModelRegistry:
        return ModelRegistry(self.config(**kw))

    def make_request(self, winner: bool = True, version: int = 1, test_groups: int = 10, experiment: str = "exp_0001", base: Path | None = None) -> RegisterRequest:
        """A selection (a winner confirmed by the Test, or the baseline retained) with fake finalist folders under `base`."""
        base = base or self.tmp / f"work_{self.counter}"
        self.counter += 1
        baseline = make_result("val", 4, name="baseline", version=version)
        errors = [1, 3] if winner else [4, 6]
        cands, dirs = [], []
        for i, k in enumerate(errors):
            trial = base / "candidates" / f"{experiment}_trial_{i:03d}"
            model = trial / f"{experiment}_trial_{i:03d}_step_00100"
            model.mkdir(parents=True)
            (model / "model.safetensors").write_bytes(os.urandom(4096))
            (model / "config.json").write_text('{"model_type": "whisper"}', encoding="utf-8")
            (model / "tokenizer.json").write_text("{}", encoding="utf-8")
            (model / "pipeline_model_info.json").write_text('{"source_config_ref": "%s/config_full.yaml"}' % experiment, encoding="utf-8")
            (trial / "checkpoints" / "checkpoint-100").mkdir(parents=True)
            (trial / "checkpoints" / "checkpoint-100" / "optimizer.pt").write_bytes(os.urandom(1024))
            (trial / "run_info.json").write_text("{}", encoding="utf-8")
            cands.append(replace(candidate(i, k, val_loss=0.8 + 0.05 * i, version=version), model_dir=model))
            dirs.append(trial)
        selection = Selector(SelectConfig()).select(SelectionRequest(baseline, cands, COMPARATOR, FakeTest({None: 4, 0: 2}, groups=test_groups, version=version)))
        card = card_from_selection(
            selection, baseline,
            base_model_name="whisper-tiny", base_model_path="models/whisper-tiny", data_fingerprint="d" * 16, split_fingerprint="s" * 16, subsample_fingerprint=None,
            config_ref=f"{experiment}/config_full.yaml", library_versions={"torch": "2.14.1"}, seed=42, code_version="0.0.0", generated_at="2026-10-05T12:00:00",
        )
        return RegisterRequest(selection, card, experiment, dirs, render_review(selection.review_list, "Review list"))


class RegisterTests(RegistryTestCase):
    def test_1_a_registration_creates_the_entry_with_every_file_and_an_index_row(self) -> None:
        request = self.make_request()
        winner_dir = request.selection.winner.model_dir  # type: ignore[union-attr]
        entry = self.registry().register(request)
        folder = self.root / "model_0001"
        self.assertEqual((entry.entry_id, entry.experiment_id, entry.kind, entry.test_evaluations), ("model_0001", "exp_0001", "model", 1))
        self.assertEqual(entry.model_dir, folder / "model")
        for name in ("model/model.safetensors", "model/config.json", "model/pipeline_model_info.json", "model_card.json", "MODEL_CARD.md", "SHA256SUMS", "review_list.txt"):
            self.assertTrue((folder / name).is_file(), name)
        self.assertFalse(winner_dir.exists())  # moved, not copied
        sums = (folder / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        self.assertTrue(all(len(line.split("  ")[0]) == 64 for line in sums))
        self.assertNotIn("SHA256SUMS", " ".join(sums))
        self.assertIn("model/model.safetensors", " ".join(sums))
        rows = (self.root / "index.csv").read_text(encoding="utf-8-sig").splitlines()
        self.assertEqual(len(rows), 2)  # header and one row
        self.assertEqual(rows[0].split(",")[:3], list(COLUMNS[:3]))
        self.assertTrue(rows[1].startswith("model_0001,exp_0001,model,winner,1,"))
        self.assertEqual(self.registry().verify("model_0001").ok, True)
        entry_id, experiment_id, card = card_from_json((folder / "model_card.json").read_text(encoding="utf-8"))
        self.assertEqual((entry_id, experiment_id, card.kind, card.test_status, card.trial_id), ("model_0001", "exp_0001", "model", "confirmed", 0))
        self.assertEqual(card_to_json(card, entry_id, experiment_id), (folder / "model_card.json").read_text(encoding="utf-8"))  # a lossless round trip
        self.assertEqual(self.registry().get("model_0001"), entry)
        self.assertEqual([e.entry_id for e in self.registry().list_entries()], ["model_0001"])

    def test_2_a_second_registration_creates_model_0002_and_leaves_the_first_byte_for_byte(self) -> None:
        reg = self.registry()
        reg.register(self.make_request())
        before = tree_hashes(self.root / "model_0001")
        second = reg.register(self.make_request(experiment="exp_0002"))
        self.assertEqual((second.entry_id, second.experiment_id), ("model_0002", "exp_0002"))
        self.assertEqual(tree_hashes(self.root / "model_0001"), before)
        self.assertEqual(second.test_evaluations, 2)  # the same Test set: looked at by two registered entries
        rows = (self.root / "index.csv").read_text(encoding="utf-8-sig").splitlines()
        self.assertEqual([r.split(",")[0] for r in rows[1:]], ["model_0001", "model_0002"])
        self.assertEqual(reg.card("model_0001").test_evaluations, 1)  # an entry is immutable: its own count stays as it was

    def test_numbers_are_never_reused_and_an_existing_entry_is_never_overwritten(self) -> None:
        reg = self.registry()
        for n in range(3):
            reg.register(self.make_request(experiment=f"exp_000{n + 1}"))
        import shutil

        shutil.rmtree(self.root / "model_0002")
        reg.rebuild_index()
        self.assertEqual(reg.register(self.make_request(experiment="exp_0004")).entry_id, "model_0004")  # the highest plus one, a gap stays a gap
        self.assertEqual(sorted(p.name for p in self.root.glob("model_*")), ["model_0001", "model_0003", "model_0004"])

    def test_3_a_crash_before_the_final_rename_leaves_no_entry_no_stray_files_and_no_cleanup(self) -> None:
        reg = self.registry()
        request = self.make_request()
        other, winner_trial = request.finalist_dirs[1], request.finalist_dirs[0]
        winner_before = tree_hashes(winner_trial)
        with mock.patch("stt.registry.registry.hashing.write_sums", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                reg.register(request)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), [])  # no entry, no temporary folder, no index
        self.assertEqual(tree_hashes(winner_trial), winner_before)  # the model folder is back where it was, with its checkpoints
        self.assertTrue(other.is_dir())  # the cleanup did not run
        self.assertEqual(reg.register(request).entry_id, "model_0001")  # and the same request registers fine afterwards

    def test_a_crash_after_the_rename_and_a_failing_index_write_also_roll_back(self) -> None:
        reg = self.registry()
        request = self.make_request()
        winner_before = tree_hashes(request.finalist_dirs[0])
        with mock.patch("stt.registry.index.IndexStore.write", side_effect=OSError("index not writable")):
            with self.assertRaises(OSError):
                reg.register(request)
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(tree_hashes(request.finalist_dirs[0]), winner_before)
        self.assertTrue(request.finalist_dirs[1].is_dir())

    def test_4_after_success_the_other_finalists_and_the_checkpoints_are_removed_and_keep_finalists_keeps_them(self) -> None:
        request = self.make_request()
        winner_trial, other = request.finalist_dirs
        self.registry().register(request)
        self.assertFalse(other.exists())  # the other finalist is gone
        self.assertFalse((winner_trial / "checkpoints").exists())  # the winner's checkpoints are gone
        self.assertTrue((winner_trial / "run_info.json").is_file())  # nothing else of the winner's folder is touched
        kept = self.make_request()
        self.registry(keep_finalists=True).register(kept)
        self.assertTrue(kept.finalist_dirs[1].is_dir())
        self.assertTrue((kept.finalist_dirs[0] / "checkpoints").is_dir())
        no_cleanup = self.make_request()
        self.registry(cleanup=False).register(no_cleanup)
        self.assertTrue(no_cleanup.finalist_dirs[1].is_dir())

    def test_5_baseline_retained_creates_an_entry_without_a_model_folder_and_an_index_row(self) -> None:
        request = self.make_request(winner=False)
        entry = self.registry().register(request)
        folder = self.root / "model_0001"
        self.assertEqual((entry.kind, entry.model_dir, entry.test_evaluations), ("baseline_retained", None, 0))
        self.assertFalse((folder / "model").exists())
        self.assertTrue((folder / "model_card.json").is_file() and (folder / "MODEL_CARD.md").is_file() and (folder / "SHA256SUMS").is_file())
        card = self.registry().card("model_0001")
        self.assertEqual((card.kind, card.verdict, card.test_status, card.test_fingerprint, card.test), ("baseline_retained", "baseline_retained", "not_evaluated", None, None))
        self.assertTrue(any("test_not_evaluated" in w for w in card.warnings))
        self.assertEqual(len(card.finalists), 2)  # the history keeps every finalist
        self.assertIn("THE BASELINE STAYS", (folder / "MODEL_CARD.md").read_text(encoding="utf-8"))
        row = (self.root / "index.csv").read_text(encoding="utf-8-sig").splitlines()[1].split(",")
        self.assertEqual(row[:4], ["model_0001", "exp_0001", "baseline_retained", "baseline_retained"])
        for finalist in request.finalist_dirs:  # a retained baseline has no winner: the finalists are cleaned up as the brief says
            self.assertFalse(finalist.exists())

    def test_the_card_and_the_markdown_hold_the_facts(self) -> None:
        self.registry().register(self.make_request())
        card = self.registry().card("model_0001")
        self.assertEqual((card.base_model_name, card.eval_version, card.seed, card.code_version, card.generated_at), ("whisper-tiny", 1, 42, "0.0.0", "2026-10-05T12:00:00"))
        self.assertEqual((card.val.model, card.val.split, card.test.split), ("trial_0", "val", "test"))  # type: ignore[union-attr]
        self.assertAlmostEqual(card.val.wer, 0.1)  # type: ignore[union-attr]
        self.assertAlmostEqual(card.val.baseline_wer or 0, 0.4)  # type: ignore[union-attr]
        self.assertEqual(card.val.verdict, "improved")  # type: ignore[union-attr]
        self.assertEqual(card.test.verdict, "improved")  # type: ignore[union-attr]
        self.assertEqual(card.decode_settings["num_beams"], 1)
        text = (self.root / "model_0001" / "MODEL_CARD.md").read_text(encoding="utf-8")
        for part in ("# Model card: model_0001 (experiment exp_0001)", "WINNER; confirmed by the Test", "## In plain words", "## Numbers", "## All finalists on Val", "## How it was made", "looked at 1 time(s)", "lower bound", "SHA256SUMS", "set `model.path`"):
            self.assertIn(part, text)

    def test_the_test_set_fingerprint_is_stable_and_follows_the_files_and_the_references(self) -> None:
        a = make_result("test", 1, id_prefix="t")
        self.assertEqual(test_set_fingerprint(a), test_set_fingerprint(make_result("test", 3, id_prefix="t", name="other")))  # the model and the errors do not matter
        self.assertNotEqual(test_set_fingerprint(a), test_set_fingerprint(make_result("test", 1, id_prefix="t", groups=11)))  # other files
        self.assertNotEqual(test_set_fingerprint(a), test_set_fingerprint(make_result("test", 1, id_prefix="u")))  # other ids
        edited = replace(a, files=tuple(replace(f, reference=f.reference + " x") if i == 0 else f for i, f in enumerate(a.files)))
        self.assertNotEqual(test_set_fingerprint(a), test_set_fingerprint(edited))  # another reference text (another profile)
        self.assertEqual(len(test_set_fingerprint(a)), 64)


class VerifyAndIndexTests(RegistryTestCase):
    def test_6_verify_passes_on_an_intact_entry_and_fails_on_a_changed_or_missing_file_naming_it(self) -> None:
        reg = self.registry()
        reg.register(self.make_request())
        report = reg.verify("model_0001")
        self.assertEqual((report.ok, report.missing, report.mismatching), (True, (), ()))
        self.assertGreaterEqual(report.files_checked, 6)
        weights = self.root / "model_0001" / "model" / "model.safetensors"
        data = bytearray(weights.read_bytes())
        data[10] ^= 0xFF  # one flipped byte, as after a bad copy
        weights.write_bytes(bytes(data))
        with self.assertRaises(RegistryVerifyError) as ctx:
            reg.verify("model_0001")
        self.assertIn("model/model.safetensors", str(ctx.exception))
        self.assertEqual(ctx.exception.report.mismatching, ("model/model.safetensors",))  # type: ignore[attr-defined]
        weights.unlink()
        with self.assertRaises(RegistryVerifyError) as ctx2:
            reg.verify("model_0001")
        self.assertEqual(ctx2.exception.report.missing, ("model/model.safetensors",))  # type: ignore[attr-defined]
        (self.root / "model_0001" / "stray.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(RegistryVerifyError) as ctx3:
            reg.verify("model_0001")
        self.assertIn("stray.txt", str(ctx3.exception))

    def test_verify_edge_cases(self) -> None:
        reg = self.registry(checksums=False)
        reg.register(self.make_request())
        self.assertFalse((self.root / "model_0001" / "SHA256SUMS").exists())
        with self.assertRaises(RegistryError) as ctx:
            reg.verify("model_0001")
        self.assertIn("registry.checksums false", str(ctx.exception))
        with self.assertRaises(RegistryError) as ctx2:
            reg.verify("model_0042")
        self.assertIn("does not exist", str(ctx2.exception))
        with self.assertRaises(RegistryError) as ctx3:
            reg.get("model_0042")
        self.assertIn("model_0001", str(ctx3.exception))

    def test_7_best_ignores_entries_with_another_eval_version_or_test_fingerprint(self) -> None:
        reg = self.registry()
        reg.register(self.make_request(experiment="exp_0001"))  # model_0001: Val WER 0.10, version 1, test set A
        reg.register(self.make_request(experiment="exp_0002", version=2))  # model_0002: version 2
        reg.register(self.make_request(experiment="exp_0003", test_groups=11))  # model_0003: another Test set
        reg.register(self.make_request(winner=False, experiment="exp_0004"))  # model_0004: a retained baseline
        fingerprint_a = reg.card("model_0001").test_fingerprint or ""
        self.assertEqual(reg.best(1, fingerprint_a).entry_id, "model_0001")  # type: ignore[union-attr]
        self.assertNotEqual(reg.card("model_0003").test_fingerprint, fingerprint_a)
        self.assertEqual(reg.best(2, reg.card("model_0002").test_fingerprint or "").entry_id, "model_0002")  # type: ignore[union-attr]
        self.assertIsNone(reg.best(3, fingerprint_a))  # no entry of that profile
        self.assertIsNone(reg.best(1, "0" * 64))  # no entry on that Test set
        self.assertEqual(reg.test_set_count(fingerprint_a), 1 + (reg.card("model_0002").test_fingerprint == fingerprint_a))
        self.assertEqual(reg.card("model_0003").test_evaluations, 1)

    def test_best_takes_the_lowest_val_wer_among_the_same_kind_of_entries(self) -> None:
        reg = self.registry()
        weaker = self.make_request(experiment="exp_0001")
        reg.register(weaker)
        better = self.make_request(experiment="exp_0002")
        better = replace(better, card=replace(better.card, val=replace(better.card.val, wer=0.05)))  # type: ignore[type-var]
        reg.register(better)
        fp = reg.card("model_0001").test_fingerprint or ""
        self.assertEqual(reg.best(1, fp).entry_id, "model_0002")  # type: ignore[union-attr]

    def test_8_rebuild_index_reproduces_it_from_the_cards_and_a_damaged_index_fails_without_being_rebuilt(self) -> None:
        reg = self.registry()
        reg.register(self.make_request(experiment="exp_0001"))
        reg.register(self.make_request(winner=False, experiment="exp_0002"))
        original = (self.root / "index.csv").read_bytes()
        (self.root / "index.csv").unlink()
        with self.assertRaises(RegistryError) as ctx:
            reg.best(1, "x")
        self.assertIn("missing", str(ctx.exception))
        self.assertIn("rebuild_index", str(ctx.exception))
        self.assertFalse((self.root / "index.csv").exists())  # not rebuilt silently
        reg.rebuild_index()
        self.assertEqual((self.root / "index.csv").read_bytes(), original)  # the same bytes, derived from the cards
        damaged = original.replace(b"entry_id", b"entry__id", 1)
        (self.root / "index.csv").write_bytes(damaged)
        with self.assertRaises(RegistryError) as ctx2:
            reg.best(1, "x")
        self.assertIn("unexpected header", str(ctx2.exception))
        self.assertEqual((self.root / "index.csv").read_bytes(), damaged)  # untouched
        with self.assertRaises(RegistryError):
            reg.register(self.make_request(experiment="exp_0003"))  # no registration on a damaged index
        self.assertEqual(sorted(p.name for p in self.root.glob("model_*")), ["model_0001", "model_0002"])
        (self.root / "index.csv").write_bytes(original.replace(b"model_0002", b"", 1)[: -10])
        with self.assertRaises(RegistryError):
            reg.best(1, "x")
        reg.rebuild_index()
        self.assertEqual(reg.best(1, reg.card("model_0001").test_fingerprint or "").entry_id, "model_0001")  # type: ignore[union-attr]

    def test_an_index_that_lists_other_entries_than_the_folder_holds_fails(self) -> None:
        reg = self.registry()
        reg.register(self.make_request())
        rows = (self.root / "index.csv").read_text(encoding="utf-8-sig").splitlines()
        (self.root / "index.csv").write_text("﻿" + "\r\n".join(rows[:1]) + "\r\n", encoding="utf-8")  # header only
        with self.assertRaises(RegistryError) as ctx:
            reg.best(1, "x")
        self.assertIn("does not match the entries", str(ctx.exception))

    def test_a_damaged_card_names_the_file(self) -> None:
        reg = self.registry()
        reg.register(self.make_request())
        (self.root / "model_0001" / "model_card.json").write_text("{ nope", encoding="utf-8")
        with self.assertRaises(RegistryError) as ctx:
            reg.list_entries()
        self.assertIn("model_card.json", str(ctx.exception))


class TransferAndPathTests(RegistryTestCase):
    def test_9_move_and_copy_modes_work_and_paths_with_spaces_and_hebrew_characters_work(self) -> None:
        base = self.tmp / "ניסוי ראשון" / "my work"
        moved = self.make_request(base=base)
        self.root = self.tmp / "המאגר שלי" / "my registry"
        entry = self.registry().register(moved)
        self.assertTrue((entry.model_dir / "model.safetensors").is_file())  # type: ignore[operator]
        self.assertFalse(moved.selection.winner.model_dir.exists())  # type: ignore[union-attr]
        copied = self.make_request(base=self.tmp / "copy work")
        win = copied.selection.winner.model_dir  # type: ignore[union-attr]
        entry2 = self.registry(transfer="copy", cleanup=False).register(copied)
        self.assertTrue(win.is_dir())  # copy mode keeps the original
        self.assertEqual(tree_hashes(win), {k.removeprefix("model/"): v for k, v in tree_hashes(entry2.model_dir.parent).items() if k.startswith("model/")})  # type: ignore[union-attr]
        self.assertEqual(self.registry().verify(entry2.entry_id).ok, True)

    def test_a_move_across_drives_is_a_verified_copy_and_the_source_is_deleted_only_after_success(self) -> None:
        request = self.make_request()
        win = request.selection.winner.model_dir  # type: ignore[union-attr]
        with mock.patch("stt.registry.registry._same_drive", return_value=False):
            self.registry().register(request)
        self.assertFalse(win.exists())  # deleted after the whole registration succeeded
        self.assertTrue((self.root / "model_0001" / "model" / "model.safetensors").is_file())
        failing = self.make_request()
        fwin = failing.selection.winner.model_dir  # type: ignore[union-attr]
        with mock.patch("stt.registry.registry._same_drive", return_value=False), mock.patch("stt.registry.index.IndexStore.write", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                self.registry().register(failing)
        self.assertTrue((fwin / "model.safetensors").is_file())  # a failed registration leaves the source untouched
        self.assertEqual(sorted(p.name for p in self.root.iterdir() if p.name.startswith(".")), [])

    def test_a_copy_that_differs_from_the_original_is_detected(self) -> None:
        request = self.make_request()
        win = request.selection.winner.model_dir  # type: ignore[union-attr]
        real_copytree = __import__("shutil").copytree

        def bad_copy(src, dst, *a, **k):  # type: ignore[no-untyped-def]
            result = real_copytree(src, dst, *a, **k)
            (Path(dst) / "model.safetensors").write_bytes(b"corrupted")
            return result

        with mock.patch("stt.registry.registry._same_drive", return_value=False), mock.patch("shutil.copytree", side_effect=bad_copy):
            with self.assertRaises(RegistryError) as ctx:
                self.registry().register(request)
        self.assertIn("not identical", str(ctx.exception))
        self.assertTrue((win / "model.safetensors").is_file())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_read_only_model_files_do_not_block_the_cleanup(self) -> None:
        request = self.make_request()
        other_file = request.finalist_dirs[1] / "checkpoints" / "checkpoint-100" / "optimizer.pt"
        os.chmod(other_file, stat.S_IREAD)
        self.registry().register(request)
        self.assertFalse(request.finalist_dirs[1].exists())


class FailureTests(RegistryTestCase):
    def test_a_missing_or_empty_winner_folder_a_leftover_temp_folder_and_a_card_that_does_not_match_fail_and_change_nothing(self) -> None:
        reg = self.registry()
        request = self.make_request()
        win = request.selection.winner.model_dir  # type: ignore[union-attr]
        for f in list(win.iterdir()):
            f.unlink()
        with self.assertRaises(RegistryError) as ctx:
            reg.register(request)
        self.assertIn("missing or empty", str(ctx.exception))
        self.assertIn(str(win), str(ctx.exception))
        self.assertFalse(self.root.exists() and any(self.root.iterdir()))
        good = self.make_request()
        (self.root).mkdir(exist_ok=True)
        (self.root / ".tmp_model_0001").mkdir()
        with self.assertRaises(RegistryError) as ctx2:
            reg.register(good)
        self.assertIn("leftover temporary folder", str(ctx2.exception))
        self.assertTrue((good.selection.winner.model_dir / "model.safetensors").is_file())  # type: ignore[union-attr]
        (self.root / ".tmp_model_0001").rmdir()
        wrong = replace(good, card=replace(good.card, kind="baseline_retained", verdict="baseline_retained"))
        with self.assertRaises(RegistryError) as ctx3:
            reg.register(wrong)
        self.assertIn("does not match the selection", str(ctx3.exception))
        other_version = replace(good, card=replace(good.card, eval_version=9))
        with self.assertRaises(RegistryError):
            reg.register(other_version)

    def test_finalist_folders_that_contain_or_lie_inside_the_registry_are_refused_before_anything_happens(self) -> None:
        request = self.make_request()
        inside = replace(request, finalist_dirs=[*request.finalist_dirs, self.root / "model_0009"])
        with self.assertRaises(RegistryError) as ctx:
            self.registry().register(inside)
        self.assertIn("outside the registry", str(ctx.exception))
        containing = replace(request, finalist_dirs=[*request.finalist_dirs, self.tmp])  # a folder that contains the registry
        with self.assertRaises(RegistryError):
            self.registry().register(containing)
        self.assertTrue(request.selection.winner.model_dir.is_dir())  # type: ignore[union-attr]
        self.assertTrue(all(d.is_dir() for d in request.finalist_dirs))

    def test_config_defaults_and_the_yaml_example(self) -> None:
        c = RegistryConfig()
        self.assertEqual((c.dir, c.keep_finalists, c.cleanup, c.transfer, c.checksums), (Path("registry"), False, True, "move", True))
        loaded = load_config(ROOT / "configs" / "registry_default.yaml", env_file=None, target_environ={}, registry=SectionRegistry({"registry": RegistryConfig}))
        self.assertEqual(loaded.section("registry", RegistryConfig), c)
        for bad in ({"transfer": "link"}, {"nope": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                RegistryConfig.model_validate(bad)

    def test_10_no_network_and_no_model_library_imports(self) -> None:
        forbidden = {"torch", "transformers", "socket", "urllib", "requests", "accelerate", "safetensors"}
        offenders = []
        for path in sorted((ROOT / "src" / "stt" / "registry").glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                if any(m.split(".")[0] in forbidden for m in modules):
                    offenders.append((path.name, modules))
        self.assertEqual(offenders, [])
        code = "import sys; sys.path.insert(0, 'src'); import stt.registry; print('torch' in sys.modules, 'transformers' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False False")
        with mock.patch("socket.socket.connect", side_effect=AssertionError("network used")):
            self.registry().register(self.make_request())
            self.registry().verify("model_0001")


if __name__ == "__main__":
    unittest.main()
