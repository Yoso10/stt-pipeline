"""Unit tests of the ModelExporter component (stt.modelexporter): the acceptance tests of the C11 brief, offline, in temporary folders.

How to run (project root, virtual environment active):
    python -m unittest tests.test_modelexporter -v
The registry entries are made by the real ModelRegistry from fake evaluation results (tests/test_registry.py builds the selections); the "model" of a real-export test is a copy of the stored tiny
test model (tests/assets/tiny_whisper, random weights) so that the ModelLoader really loads it; the fake exporters copy or corrupt small files. No network, no Test evaluation, no download.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
import time
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_registry import RegistryTestCase, tree_hashes  # noqa: E402

from stt.config import SectionRegistry, load_config  # noqa: E402
from stt.modelexporter import (  # noqa: E402
    ExportConfig,
    ExportConfigError,
    ExportDependencyError,
    ExporterRegistry,
    ExportError,
    ExportRecord,
    ExportResult,
    ModelExportService,
    VerifyInput,
    VerifyResult,
    append_record,
    default_exporter_registry,
    missing_packages,
    read_exports,
    wer_difference,
)
from stt.modelexporter.hf_safetensors import HfSafetensorsExporter, safetensors_parameter_count  # noqa: E402
from stt.registry import RegistryVerifyError, sha256_file  # noqa: E402

TINY = ROOT / "tests" / "assets" / "tiny_whisper"


class CopyExporter:
    """A fake exporter that COPIES the model files into the target folder and verifies that the copy is identical."""

    name = "fake_copy"
    requires: tuple[str, ...] = ()
    corrupt = False

    def export(self, source: Path, target_dir: Path) -> ExportResult:
        target_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            dest = target_dir / path.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            files.append(dest)
        if self.corrupt:
            files[0].write_bytes(b"deliberately wrong")
        return ExportResult(self.name, target_dir, files, 0.01)

    def verify(self, result: ExportResult, check: VerifyInput) -> VerifyResult:
        present = all(p.is_file() for p in result.files)
        return VerifyResult("transcribe", present, "every copied file is there" if present else "a copied file is missing", 0.0 if present else 0.5)


class WrongOutputExporter(CopyExporter):
    name = "fake_wrong"
    corrupt = True

    def verify(self, result: ExportResult, check: VerifyInput) -> VerifyResult:
        wrong = [p for p in result.files if p.read_bytes() == b"deliberately wrong"]
        return VerifyResult("transcribe", not wrong, "an exported file is wrong" if wrong else "fine", 0.4 if wrong else 0.0)


class MissingPackageExporter:
    name = "fake_missing"
    requires: tuple[str, ...] = ("a_package_that_is_not_installed_xyz",)
    called = False

    def export(self, source: Path, target_dir: Path) -> ExportResult:
        MissingPackageExporter.called = True
        return ExportResult(self.name, target_dir, [], 0.0)

    def verify(self, result: ExportResult, check: VerifyInput) -> VerifyResult:
        return VerifyResult("load", True, "")


def fake_registry() -> ExporterRegistry:
    return ExporterRegistry({"hf_safetensors": HfSafetensorsExporter, "fake_copy": CopyExporter, "fake_wrong": WrongOutputExporter, "fake_missing": MissingPackageExporter})  # type: ignore[dict-item]


class ExportCase(RegistryTestCase):
    """A registry with one entry of a REAL loadable tiny model (model_0001) and, when asked, a retained-baseline entry."""

    def entry(self, real: bool = True, winner: bool = True, base_model_path: str = "models/not_on_this_machine") -> str:
        request = self.make_request(winner=winner)
        if winner and real:
            shutil.copytree(TINY, request.selection.winner.model_dir, dirs_exist_ok=True)  # type: ignore[union-attr]
        request = replace(request, card=replace(request.card, base_model_path=base_model_path))
        return self.registry().register(request).entry_id

    def service(self, **config: Any) -> ModelExportService:
        return ModelExportService(ExportConfig(**config), self.registry(), fake_registry())

    def entry_dir(self, entry_id: str) -> Path:
        return self.root / entry_id


class RealExportTests(ExportCase):
    def test_1_the_hf_exporter_points_at_the_entrys_own_folder_and_the_load_verification_passes(self) -> None:
        entry_id = self.entry()
        records = self.service().export(entry_id, ["hf_safetensors"], exported_at="2026-10-07T12:00:00")
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual((r.exporter, r.status, r.target_dir, r.exported_at), ("hf_safetensors", "ok", "model", "2026-10-07T12:00:00"))
        self.assertIn("model/model.safetensors", r.files)
        assert r.verify is not None
        self.assertEqual((r.verify.level, r.verify.ok, r.verify.wer_diff), ("load", True, None))
        for part in ("complete Whisper folder", "216704 parameters", "equals the 216704 the weights file declares", "mel bins match config.json", "base model folder is not on this machine"):
            self.assertIn(part, r.verify.detail)
        self.assertFalse((self.entry_dir(entry_id) / "exports" / "hf_safetensors").exists())  # nothing was copied (the DEFAULT of the brief)

    def test_5_the_export_is_recorded_and_the_card_and_the_model_files_are_unchanged_and_the_registry_still_verifies(self) -> None:
        entry_id = self.entry()
        folder = self.entry_dir(entry_id)
        before = tree_hashes(folder)
        index_before = (self.root / "index.csv").read_bytes()
        self.service().export(entry_id, ["hf_safetensors", "fake_copy"], exported_at="t")
        after = tree_hashes(folder)
        for name, digest in before.items():
            self.assertEqual(after[name], digest, name)  # the card, the checksums and every model file: byte for byte
        self.assertEqual(sorted(set(after) - set(before)), sorted(["exports.json"] + [n for n in after if n.startswith("exports/")]))
        self.assertTrue(self.registry().verify(entry_id).ok)  # the exports are outside the entry's checksums
        self.assertEqual((self.root / "index.csv").read_bytes(), index_before)
        data = json.loads((folder / "exports.json").read_text(encoding="utf-8"))
        self.assertEqual((data["schema_version"], [e["exporter"] for e in data["exports"]], [e["status"] for e in data["exports"]]), (1, ["hf_safetensors", "fake_copy"], ["ok", "ok"]))
        self.assertFalse((folder / "exports.json.tmp").exists())
        self.assertEqual([r.exporter for r in self.service().records(entry_id)], ["hf_safetensors", "fake_copy"])

    def test_a_file_that_is_not_an_export_is_still_flagged_by_the_registry_verify(self) -> None:
        entry_id = self.entry()
        self.service().export(entry_id, ["hf_safetensors"])
        (self.entry_dir(entry_id) / "stray.txt").write_text("not listed", encoding="utf-8")
        with self.assertRaises(RegistryVerifyError):
            self.registry().verify(entry_id)

    def test_a_base_model_with_the_same_architecture_is_compared_and_a_different_one_fails_the_verification(self) -> None:
        same = self.entry(base_model_path=str(TINY))
        record = self.service().export(same, ["hf_safetensors"])[0]
        assert record.verify is not None
        self.assertIn("the architecture equals the base model's (tiny_whisper)", record.verify.detail)
        other = self.root.parent / "other_base"
        other.mkdir()
        config = json.loads((TINY / "config.json").read_text(encoding="utf-8"))
        config["d_model"] = config["d_model"] * 2
        (other / "config.json").write_text(json.dumps(config), encoding="utf-8")
        self.root = self.tmp / "registry2"  # a second registry with its own entry
        different = self.entry(base_model_path=str(other))
        with self.assertRaises(ExportError) as raised:
            self.service().export(different, ["hf_safetensors"])
        self.assertIn("the architecture differs from the base model", str(raised.exception))
        failed = self.service().records(different)[0]
        self.assertEqual((failed.status, failed.verify.ok if failed.verify else None), ("failed", False))  # kept and marked, as the brief says

    def test_a_parameter_count_that_differs_from_the_weights_header_fails_the_verification(self) -> None:
        entry_id = self.entry()
        self.assertEqual(safetensors_parameter_count(TINY), 216704)
        with mock.patch("stt.modelexporter.hf_safetensors.safetensors_parameter_count", return_value=216705):
            with self.assertRaises(ExportError) as raised:
                self.service().export(entry_id, ["hf_safetensors"])
        self.assertIn("declares 216705", str(raised.exception))

    def test_a_folder_that_is_not_a_model_fails_the_load_verification_with_the_reason_and_is_recorded(self) -> None:
        entry_id = self.entry(real=False)  # the files of the registry tests are not a loadable model
        with self.assertRaises(ExportError) as raised:
            self.service().export(entry_id, ["hf_safetensors"])
        self.assertIn("not a complete model folder", str(raised.exception))
        self.assertEqual(self.service().records(entry_id)[0].status, "failed")

    def test_the_hf_exporter_changes_no_file_of_the_entry_even_when_its_verification_fails(self) -> None:
        entry_id = self.entry(real=False)
        before = tree_hashes(self.entry_dir(entry_id))
        with self.assertRaises(ExportError):
            self.service().export(entry_id, ["hf_safetensors"])
        after = tree_hashes(self.entry_dir(entry_id))
        self.assertEqual({k: v for k, v in after.items() if k != "exports.json"}, before)


class FailureRuleTests(ExportCase):
    def test_2_an_unknown_exporter_fails_listing_the_registered_names_before_anything_is_touched(self) -> None:
        entry_id = self.entry()
        with self.assertRaises(ExportConfigError) as raised:
            self.service().export(entry_id, ["onnx"])
        self.assertIn("Registered exporters: fake_copy, fake_missing, fake_wrong, hf_safetensors", str(raised.exception))
        self.assertFalse((self.entry_dir(entry_id) / "exports.json").exists())

    def test_3_an_exporter_whose_package_is_missing_fails_naming_the_package_and_says_the_machine_is_offline(self) -> None:
        entry_id = self.entry()
        MissingPackageExporter.called = False
        with self.assertRaises(ExportDependencyError) as raised:
            self.service().export(entry_id, ["hf_safetensors", "fake_missing"])
        for part in ("a_package_that_is_not_installed_xyz", "offline", "do not enable online mode"):
            self.assertIn(part, str(raised.exception))
        self.assertFalse(MissingPackageExporter.called)
        self.assertFalse((self.entry_dir(entry_id) / "exports.json").exists())  # nothing was exported, not even the first format
        self.assertEqual(missing_packages(["json", "a_package_that_is_not_installed_xyz"]), ["a_package_that_is_not_installed_xyz"])

    def test_4_a_deliberately_wrong_output_fails_the_verification_and_is_recorded_as_failed_but_kept(self) -> None:
        entry_id = self.entry()
        with self.assertRaises(ExportError) as raised:
            self.service().export(entry_id, ["fake_wrong"])
        self.assertIn("marked failed", str(raised.exception))
        record = self.service().records(entry_id)[0]
        self.assertEqual((record.status, record.verify.ok, record.verify.wer_diff), ("failed", False, 0.4))  # type: ignore[union-attr]
        self.assertTrue((self.entry_dir(entry_id) / "exports" / "fake_wrong").is_dir())  # the export is kept

    def test_6_an_existing_non_empty_target_fails_and_nothing_is_overwritten(self) -> None:
        entry_id = self.entry()
        self.service().export(entry_id, ["fake_copy"])
        marker = self.entry_dir(entry_id) / "exports" / "fake_copy" / "config.json"
        before = marker.read_bytes()
        with self.assertRaises(ExportError) as raised:
            self.service().export(entry_id, ["fake_copy"])
        self.assertIn("already exists and is not empty", str(raised.exception))
        self.assertEqual(marker.read_bytes(), before)
        self.assertEqual(len(self.service().records(entry_id)), 1)

    def test_7_the_automatic_export_uses_the_configured_formats_and_skips_a_retained_baseline(self) -> None:
        winner = self.entry()
        retained = self.registry().register(self.make_request(winner=False, experiment="exp_0002")).entry_id
        service = self.service(formats=["hf_safetensors", "fake_copy"])
        self.assertEqual([r.exporter for r in service.export(winner, explicit=False)], ["hf_safetensors", "fake_copy"])  # the formats of the config
        with self.assertLogs("stt.modelexporter", level="INFO") as logs:
            self.assertEqual(service.export(retained, explicit=False), [])  # skipped and logged
        self.assertIn("nothing to export", logs.output[0])
        self.assertFalse((self.entry_dir(retained) / "exports.json").exists())
        with self.assertRaises(ExportError) as raised:  # asked explicitly: a plain explanation
            service.export(retained)
        self.assertIn("kept the baseline", str(raised.exception))

    def test_a_corrupted_entry_fails_before_anything_is_exported(self) -> None:
        entry_id = self.entry()
        weights = self.entry_dir(entry_id) / "model" / "model.safetensors"
        weights.write_bytes(weights.read_bytes()[:-1] + b"X")
        with self.assertRaises(ExportError) as raised:
            self.service().export(entry_id, ["hf_safetensors"])
        self.assertIn("do not match its checksums", str(raised.exception))
        self.assertFalse((self.entry_dir(entry_id) / "exports.json").exists())

    def test_an_entry_without_checksums_is_exported_with_a_warning(self) -> None:
        request = self.make_request()
        shutil.copytree(TINY, request.selection.winner.model_dir, dirs_exist_ok=True)  # type: ignore[union-attr]
        entry_id = self.registry(checksums=False).register(replace(request, card=replace(request.card, base_model_path="models/nope"))).entry_id
        with self.assertLogs("stt.modelexporter", level="WARNING") as logs:
            records = self.service().export(entry_id, ["hf_safetensors"])
        self.assertEqual(records[0].status, "ok")
        self.assertIn("has no checksums", logs.output[0])

    def test_an_entry_that_does_not_exist_fails(self) -> None:
        self.entry()
        with self.assertRaises(ExportError) as raised:
            self.service().export("model_0042", ["hf_safetensors"])
        self.assertIn("does not exist", str(raised.exception))

    def test_with_export_verify_false_the_export_is_recorded_as_unverified(self) -> None:
        entry_id = self.entry(real=False)  # not even a loadable model: nothing is verified
        records = self.service(verify=False).export(entry_id, ["hf_safetensors"])
        self.assertEqual((records[0].status, records[0].verify), ("unverified", None))


class RecordAndHelperTests(ExportCase):
    def test_records_round_trip_append_in_order_and_a_damaged_file_is_never_repaired(self) -> None:
        folder = self.tmp / "entry"
        folder.mkdir()
        a = ExportRecord("model_0001", "x", "exports/x", ("exports/x/a",), VerifyResult("transcribe", True, "fine", 0.01), "ok", 1.5, "t1")
        b = ExportRecord("model_0001", "y", "model", ("model/a",), None, "unverified", 0.0, "t2")
        append_record(folder, a)
        append_record(folder, b)
        self.assertEqual(read_exports(folder), [a, b])
        (folder / "exports.json").write_text("{damaged", encoding="utf-8")
        with self.assertRaises(ExportError) as raised:
            read_exports(folder)
        self.assertIn("never repaired", str(raised.exception))
        self.assertEqual((folder / "exports.json").read_text(encoding="utf-8"), "{damaged")

    def test_the_verify_input_takes_the_configured_number_of_files_and_the_allowed_difference(self) -> None:
        entry_id = self.entry()
        service = self.service(verify_samples=2, max_wer_diff=0.05)
        check = service.verify_input([Path(f"{i}.wav") for i in range(5)], [str(i) for i in range(5)], Path("base"))
        self.assertEqual((len(check.audio_paths), len(check.expected_texts), check.max_wer_diff, check.base_model_dir), (2, 2, 0.05, Path("base")))
        self.assertTrue(entry_id)

    def test_the_wer_difference_is_the_corpus_wer_and_needs_one_text_per_file(self) -> None:
        self.assertEqual(wer_difference(["a b c d"], ["a b c x"]), 0.25)
        self.assertEqual(wer_difference(["a b", "c d"], ["a b", "c d"]), 0.0)
        self.assertEqual(wer_difference(["a b", "c d e f"], ["a x", "c d e f"]), 1 / 6)  # all words together, not the mean of the files
        self.assertEqual(wer_difference([""], [""]), 0.0)
        with self.assertRaises(ValueError):
            wer_difference(["a"], [])

    def test_the_time_of_a_record_is_passed_in_and_the_component_reads_no_clock(self) -> None:
        entry_id = self.entry()
        with mock.patch.object(time, "time", side_effect=AssertionError("no wall clock")), mock.patch("datetime.datetime", side_effect=AssertionError("no clock")):
            record = self.service().export(entry_id, ["hf_safetensors"], exported_at="2026-10-07T13:00:00")[0]
        self.assertEqual(record.exported_at, "2026-10-07T13:00:00")


class ConfigAndStructureTests(unittest.TestCase):
    def test_the_defaults_are_those_of_the_brief_and_the_example_yaml_matches(self) -> None:
        c = ExportConfig()
        self.assertEqual((c.formats, c.verify, c.verify_samples, c.max_wer_diff), (["hf_safetensors"], True, 5, 0.02))
        loaded = load_config(ROOT / "configs" / "export_default.yaml", env_file=None, target_environ={}, registry=SectionRegistry({"export": ExportConfig}))
        self.assertEqual(loaded.section("export", ExportConfig), c)

    def test_8_format_libraries_are_imported_only_inside_their_adapter_and_nothing_uses_the_network(self) -> None:
        forbidden = {"torch", "transformers", "safetensors", "requests", "socket", "http", "urllib3", "numpy", "scipy"}
        for path in sorted((ROOT / "src" / "stt" / "modelexporter").glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                for name in names:
                    self.assertNotIn(name.split(".")[0], forbidden, f"{path.name} imports {name}")
                    self.assertNotEqual(name, "urllib.request", path.name)
                    if name.split(".")[0:2] == ["stt", "modelloader"]:
                        self.assertEqual(path.name, "hf_safetensors.py", f"{path.name} imports the model loader")
        code = "import sys; sys.path.insert(0, 'src'); import stt.modelexporter as m; m.default_exporter_registry(); print('torch' in sys.modules, 'transformers' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False False")

    def test_the_registry_checksum_change_is_additive(self) -> None:
        from stt.registry.hashing import list_files

        folder = ROOT / "tests" / "assets" / "tiny_whisper"
        self.assertNotIn("exports.json", list_files(folder))
        self.assertEqual(list_files(folder, exclude=(), skip_exports=False), sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()))


if __name__ == "__main__":
    unittest.main()
