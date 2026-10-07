"""Tests of the Orchestrator side of the export (Wave 3 task 5): the `export` section, the automatic export at the end of the select_and_register stage, `python -m stt export`, and the export column of
`registry list`.

How to run (project root, virtual environment active):
    python -m unittest tests.test_orchestrator_export -v
The stage tests use FAKE services (tests/search_stage_fakes.py) with the REAL selector, registry and Orchestrator and a RECORDING exporter (so that no model has to load); the command tests use a REAL
registry with a REAL loadable tiny model (a copy of tests/assets/tiny_whisper) and the real hf_safetensors exporter. Nothing is downloaded, the Test split is never evaluated, nothing is deleted.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from search_stage_fakes import FakeEvaluation, fake_services  # noqa: E402
from test_modelexporter import ExportCase  # noqa: E402
from test_orchestrator_curve import curve_config  # noqa: E402
from test_registry import tree_hashes  # noqa: E402

from stt.modelexporter import ExportConfig, ExportConfigError, ExportDependencyError, ExporterRegistry, ExportError, ExportResult, VerifyInput, VerifyResult  # noqa: E402
from stt.orchestrator import InterruptFlag, NoopSleepGuard, Orchestrator, RunPlan, cli, default_stage_registry  # noqa: E402
from stt.orchestrator.selection import SelectAndRegisterStage  # noqa: E402
from stt.orchestrator.wiring import ExportWiring, RegistryWiring, exit_code_for, section_registry  # noqa: E402
from stt.registry import RegistryConfig  # noqa: E402
from stt.runmanager import ExperimentSpec, ExperimentStore, RunConfig  # noqa: E402

RUN_STAGES = ["prepare", "evaluate", "search", "finalists", "select_and_register"]


class RecordingExporter:
    """Named hf_safetensors (the default format); records its calls and verifies as told. Nothing is copied."""

    name = "hf_safetensors"
    requires: tuple[str, ...] = ()
    calls: list[str] = []
    ok = True

    def export(self, source: Path, target_dir: Path) -> ExportResult:
        RecordingExporter.calls.append(str(source))
        return ExportResult(self.name, source, sorted(p for p in source.rglob("*") if p.is_file()), 0.0)

    def verify(self, result: ExportResult, check: VerifyInput) -> VerifyResult:
        return VerifyResult("load", RecordingExporter.ok, "fine" if RecordingExporter.ok else "the model does not load (a test)")


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class StageCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.store = ExperimentStore(RunConfig(experiments_dir=self.tmp / "experiments"))
        RecordingExporter.calls = []
        RecordingExporter.ok = True

    def run_chain(self, config: Any, handle: Any = None, errors: dict[int, int] | None = None, with_export: bool = True) -> tuple[Any, Any]:
        handle = handle or self.store.create(ExperimentSpec(config, None, "0.0.0", {}, "2026-10-07T09:00:00"))
        evaluation = FakeEvaluation(config, val_errors=errors) if errors is not None else None
        export = ExportWiring(ExportConfig(), RegistryWiring(config), ExporterRegistry({"hf_safetensors": RecordingExporter})) if with_export else None  # type: ignore[dict-item]
        services = fake_services(config, evaluation=evaluation, export=export)
        outcome = Orchestrator(config, services, default_stage_registry(), InterruptFlag(), NoopSleepGuard()).run(RunPlan(RUN_STAGES, handle))
        return outcome, handle


class AutomaticExportTests(StageCase):
    def test_a_registered_winner_is_exported_automatically_and_recorded_with_the_time_passed_in(self) -> None:
        config = curve_config(self.tmp)
        outcome, _ = self.run_chain(config)
        self.assertEqual((outcome.exit_code, outcome.verdict), (0, "winner"))
        summary = outcome.stage_results["select_and_register"].summary
        self.assertEqual(summary["export"], "hf_safetensors ok")
        self.assertEqual(len(RecordingExporter.calls), 1)
        self.assertTrue(RecordingExporter.calls[0].replace("\\", "/").endswith("registry/model_0001/model"))  # the model that was just registered
        records = RegistryWiring(config).exports("model_0001")
        self.assertEqual((len(records), records[0].exporter, records[0].status, records[0].exported_at), (1, "hf_safetensors", "ok", "2026-10-06T10:00:00"))  # Services.now
        self.assertTrue(RegistryWiring(config).verify("model_0001").ok)  # the exports are outside the checksums
        artifacts = [str(p) for p in outcome.stage_results["select_and_register"].artifacts]
        self.assertTrue(any(a.endswith("exports.json") for a in artifacts))

    def test_a_retained_baseline_is_skipped_and_logged(self) -> None:
        config = curve_config(self.tmp)
        with self.assertLogs("stt.orchestrator", level="INFO") as logs:
            outcome, _ = self.run_chain(config, errors={0: 3, 1: 3, 2: 3, 3: 3})
        self.assertEqual((outcome.exit_code, outcome.verdict), (0, "baseline_retained"))
        self.assertIn("skipped", str(outcome.stage_results["select_and_register"].summary["export"]))
        self.assertEqual(RecordingExporter.calls, [])
        self.assertFalse((self.tmp / "registry" / "model_0001" / "exports.json").exists())
        self.assertTrue(any("nothing to export" in line for line in logs.output))

    def test_without_an_export_service_the_step_says_not_wired_and_the_run_is_unchanged(self) -> None:
        config = curve_config(self.tmp)
        outcome, _ = self.run_chain(config, with_export=False)
        self.assertEqual((outcome.exit_code, outcome.stage_results["select_and_register"].summary["export"]), (0, "not wired"))

    def test_a_failed_verification_fails_the_stage_with_exit_code_5_but_the_registration_stays(self) -> None:
        config = curve_config(self.tmp)
        RecordingExporter.ok = False
        outcome, handle = self.run_chain(config)
        self.assertEqual((outcome.exit_code, outcome.status), (5, "failed"))
        self.assertIn("verification of the export of model_0001 failed", outcome.message or "")
        self.assertEqual([e.entry_id for e in RegistryWiring(config).entries()], ["model_0001"])  # the registration is done
        self.assertEqual(RegistryWiring(config).exports("model_0001")[0].status, "failed")  # kept and marked
        # the same command again, the failure is not hidden while the latest export of the format is failed ...
        again, _ = self.run_chain(config, handle)
        self.assertEqual(again.exit_code, 5)
        self.assertIn("python -m stt export model_0001", again.message or "")
        self.assertEqual(len(RegistryWiring(config).entries()), 1)  # never registered twice
        # ... and a later successful export (the command) lets the stage complete
        RecordingExporter.ok = True
        RegistryWiring(config)  # the registry
        ExportWiring(ExportConfig(), RegistryWiring(config), ExporterRegistry({"hf_safetensors": RecordingExporter})).export("model_0001", exported_at="later")  # type: ignore[dict-item]
        third, _ = self.run_chain(config, handle)
        self.assertEqual((third.exit_code, third.status), (0, "completed"))
        self.assertIn("done earlier", str(third.stage_results["select_and_register"].summary["export"]))

    def test_a_rerun_of_a_completed_experiment_exports_nothing_again(self) -> None:
        config = curve_config(self.tmp)
        _, handle = self.run_chain(config)
        calls = len(RecordingExporter.calls)
        again, _ = self.run_chain(config, handle)
        self.assertEqual(again.stage_results["select_and_register"].status, "skipped")
        self.assertEqual(len(RecordingExporter.calls), calls)

    def test_a_registered_experiment_without_an_export_is_exported_on_the_rerun(self) -> None:
        config = curve_config(self.tmp)
        _, handle = self.run_chain(config, with_export=False)  # registered by a version before the export existed
        handle.mark_stage("select_and_register", "failed", None)
        outcome, _ = self.run_chain(config, handle)
        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(outcome.stage_results["select_and_register"].summary["export"], "hf_safetensors ok")
        self.assertEqual(len(RegistryWiring(config).exports("model_0001")), 1)

    def test_the_export_section_is_part_of_the_stage_fingerprint_only_when_it_differs_from_the_defaults(self) -> None:
        config = curve_config(self.tmp)
        self.assertNotIn("export", SelectAndRegisterStage().config_slice(config))  # the default: experiments that finished before keep their completed stage
        changed = curve_config(self.tmp, 'export.formats=["hf_safetensors", "other"]')
        self.assertIn("export", SelectAndRegisterStage().config_slice(changed))

    def test_the_test_split_is_not_touched_by_the_export(self) -> None:
        config = curve_config(self.tmp)
        _, _ = self.run_chain(config)
        # the Test was looked at exactly as before (the winner and the baseline, once each): the export added no look
        looked = json.loads((self.tmp / "experiments" / "exp_0001" / "reports" / "test_once.json").read_text(encoding="utf-8"))["looked"]
        self.assertEqual(sorted(looked), ["baseline", "trial_1"])


class CommandTests(ExportCase):
    def args(self, *extra: str) -> list[str]:
        return ["--set", f"registry.dir={self.root.as_posix()}", *extra]

    def test_the_command_exports_the_chosen_entry_and_says_where_the_records_are(self) -> None:
        entry_id = self.entry()
        code, out, err = run_cli("export", entry_id, *self.args())
        self.assertEqual(code, 0, err)
        body = json.loads(out)
        self.assertEqual((body["entry_id"], body["error"], body["exit_code"], len(body["recorded"])), (entry_id, None, 0, 1))
        self.assertEqual((body["recorded"][0]["exporter"], body["recorded"][0]["status"], body["recorded"][0]["verify"]["level"]), ("hf_safetensors", "ok", "load"))
        self.assertTrue(Path(body["exports_file"]).is_file())
        self.assertIn("1 export(s) recorded in", err)

    def test_a_repeated_export_is_allowed_and_adds_a_record(self) -> None:
        entry_id = self.entry()
        run_cli("export", entry_id, *self.args())
        code, out, err = run_cli("export", entry_id, *self.args())
        self.assertEqual(code, 0, err)
        self.assertEqual(len(RegistryWiring(RegistryConfig(dir=self.root)).exports(entry_id)), 2)
        self.assertEqual(len(json.loads(out)["recorded"]), 1)  # this call's record only

    def test_without_an_entry_the_latest_one_is_exported(self) -> None:
        self.entry()
        second = self.registry().register(self.make_request(experiment="exp_0002")).entry_id
        self.assertEqual(second, "model_0002")
        code, out, _ = run_cli("export", *self.args("--set", "export.verify=false"))
        self.assertEqual((code, json.loads(out)["entry_id"], json.loads(out)["recorded"][0]["status"]), (0, "model_0002", "unverified"))

    def test_only_exports_json_and_exports_of_the_entry_are_written_and_registry_verify_still_passes(self) -> None:
        entry_id = self.entry()
        before = tree_hashes(self.root)
        code, _, err = run_cli("export", entry_id, *self.args())
        self.assertEqual(code, 0, err)
        after = tree_hashes(self.root)
        for name, digest in before.items():
            self.assertEqual(after[name], digest, name)  # every other byte of the registry, the index and the entry included
        self.assertEqual(sorted(set(after) - set(before)), [f"{entry_id}/exports.json"])
        self.assertEqual(run_cli("registry", "verify", *self.args())[0], 0)

    def test_an_unknown_format_an_unknown_entry_and_an_empty_registry_are_exit_code_2_that_list_what_exists(self) -> None:
        entry_id = self.entry()
        code, _, err = run_cli("export", entry_id, "--format", "onnx", *self.args())
        self.assertEqual(code, 2)
        self.assertIn("Registered exporters: hf_safetensors", err)
        self.assertFalse((self.root / entry_id / "exports.json").exists())
        code, out, err = run_cli("export", "model_0042", *self.args())
        self.assertEqual((code, out), (2, ""))
        self.assertIn("Existing entries: model_0001", err)
        with tempfile.TemporaryDirectory() as empty:
            code, out, err = run_cli("export", "--set", f"registry.dir={Path(empty).as_posix()}")
            self.assertEqual((code, out), (2, ""))
            self.assertIn("has no entries to export", err)

    def test_a_retained_baseline_a_corrupted_entry_and_a_failed_verification_are_exit_code_5(self) -> None:
        retained = self.registry().register(self.make_request(winner=False, experiment="exp_0003")).entry_id
        code, _, err = run_cli("export", retained, *self.args())
        self.assertEqual(code, 5)
        self.assertIn("kept the baseline", err)
        fake = self.entry(real=False)  # not a loadable model: the load verification fails
        code, out, err = run_cli("export", fake, *self.args())
        body = json.loads(out)
        self.assertEqual((code, body["exit_code"], body["recorded"][0]["status"], body["recorded"][0]["verify"]["ok"]), (5, 5, "failed", False))
        self.assertIn("verification of the export", body["error"])
        self.assertIn("ERROR:", err)
        weights = self.root / fake / "model" / "model.safetensors"
        weights.write_bytes(weights.read_bytes()[:-1] + b"X")
        code, _, err = run_cli("export", fake, *self.args())
        self.assertEqual(code, 5)
        self.assertIn("do not match its checksums", err)

    def test_registry_list_shows_the_number_of_exports_and_marks_a_failed_one(self) -> None:
        entry_id = self.entry()
        code, out, _ = run_cli("registry", "list", *self.args())
        self.assertIn("exports 0", out)
        run_cli("export", entry_id, *self.args())
        run_cli("export", entry_id, *self.args())
        self.assertIn("exports 2", run_cli("registry", "list", *self.args())[1])
        fake = self.entry(real=False)
        run_cli("export", fake, *self.args())
        self.assertIn("exports 1 (1 FAILED)", run_cli("registry", "list", *self.args())[1])
        (self.root / entry_id / "exports.json").write_text("{damaged", encoding="utf-8")
        self.assertIn("exports unreadable", run_cli("registry", "list", *self.args())[1])
        self.assertEqual(run_cli("registry", "list", *self.args())[0], 0)  # the list itself still works


class StructureTests(unittest.TestCase):
    def test_the_section_is_registered_the_exit_codes_are_mapped_and_the_command_has_help(self) -> None:
        self.assertIn("export", section_registry().sections)
        self.assertEqual((exit_code_for(ExportConfigError("x")), exit_code_for(ExportError("x")), exit_code_for(ExportDependencyError("x"))), (2, 5, 5))
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            cli.main(["export", "--help"])
        for part in ("python -m stt export", "--format"):
            self.assertIn(part, out.getvalue())

    def test_no_pipeline_stages_default_changed(self) -> None:
        from stt.orchestrator import PipelineConfig

        self.assertEqual(PipelineConfig().stages, ["prepare", "evaluate", "search", "finalists", "select_and_register"])  # the export is NOT a stage of the default list (owner decision, O105)


if __name__ == "__main__":
    unittest.main()
