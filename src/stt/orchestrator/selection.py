"""The `select_and_register` stage: choose the winner (or keep the baseline), look at the Test ONCE, register the result (Orchestrator Part 2, DECISIONS 57).

How to run: this module is not a script; `python -m stt run` runs the stage last. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.orchestrator.selection import validate_registry_location; print(validate_registry_location.__name__)"
Example output:
    validate_registry_location
What it does, in this order: (1) checks that `registry.dir` lies OUTSIDE `run.experiments_dir` (so no cleanup of the experiments can reach the registry); an entry of this experiment
already in the registry means the stage is done (nothing is registered twice); (2) reads the baseline Val result stored by the evaluate stage (`reports/eval_result_baseline.json`) and the
retrained finalists stored by the finalists stage; (3) makes ONE `TestOnceGuard` for the experiment around the Test evaluation and gives it to the Selector: the Selector asks for the Test only for the
winner and the baseline (never when the baseline stays), the Test split is scored only through this guard (`pipeline.test_split`; every other path REFUSES it), a result already stored in the
experiment is returned instead of a second look (`testlook.py`); (4) writes `reports/selection.json`, `selection_table.csv` and `selection_reason.txt`; (5) fills the model card
(`card_from_selection`: the fingerprints of the prepare stage, the base model, the config reference, the library versions, the seed, the code version, the time) and registers it with the
review list of the winner. NOTHING IS DELETED: the request names no finalist folder, so the registry removes no finalist and no checkpoint (owner rule, DECISIONS 56); only the winner's model
folder is moved into the registry (`registry.transfer`, see C10). After a registration the winner's model is at `registry/model_NNNN/model`. (6) AUTOMATIC EXPORT (C11, Wave 3 task 5, DECISIONS 71): when a winner was
registered, the model is exported for every format of `export.formats` (default `hf_safetensors`: no copy, verified at the load level) and recorded in the entry's `exports.json`; for a retained baseline
the step is skipped and logged. A failed export fails the stage with exit code 5 (the registration stays: it is already done); a rerun does not register or export again unless the entry has no export yet,
and it fails again while the LATEST export of a format is `failed` (`python -m stt export` repeats an export). The `export` section is part of the stage's fingerprint only when it differs from the defaults,
so experiments that finished before the export existed keep their completed stage.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from stt.config import AppConfig
from stt.evaluator import EvalResult, eval_result_from_json
from stt.orchestrator import reporting
from stt.modelexporter import ExportConfig, ExportError, ExportRecord
from stt.orchestrator.errors import OrchestratorError
from stt.orchestrator.finalists import load_finalists
from stt.orchestrator.stageio import load_result
from stt.orchestrator.stagehelpers import BASELINE_RESULT_FILE, evaluate_model, pipeline_value, section
from stt.orchestrator.testlook import TestLookStore, once_per_experiment
from stt.orchestrator.types import StageContext, StageResult
from stt.registry import RegisterRequest, card_from_selection, render_review
from stt.selector import Candidate, SelectionRequest, TestOnceGuard

logger = logging.getLogger("stt.orchestrator")


def validate_registry_location(experiments_dir: Path, registry_dir: Path) -> None:
    """The registry must lie OUTSIDE the experiments folder (and not contain it): no cleanup of the experiments may reach a registered model.

    Errors: OrchestratorError (exit code 2) naming both folders and the fix.
    """
    experiments, registry = experiments_dir.resolve(), registry_dir.resolve()
    if registry == experiments or experiments in registry.parents or registry in experiments.parents:
        raise OrchestratorError(
            f"registry.dir ({registry_dir}) is the experiments folder or lies inside it (run.experiments_dir is {experiments_dir}), or contains it. A registered model must be out of reach of "
            "any cleanup of the experiments: set registry.dir to a folder next to it, for example `registry`.",
            exit_code=2,
        )


def check_config_registry_location(config: AppConfig) -> None:
    """`validate_registry_location` for the folders named in a config (used by the command line before a long run starts, and by the stage)."""
    experiments = section(config, "run").get("experiments_dir", "experiments")
    registry = section(config, "registry").get("dir", "registry")
    validate_registry_location(Path(str(experiments)), Path(str(registry)))


class SelectAndRegisterStage:
    """select_and_register: the selection, the one Test look and the registration."""

    name = "select_and_register"
    depends_on: tuple[str, ...] = ("prepare", "evaluate", "finalists")

    def config_slice(self, config: AppConfig) -> Mapping[str, Any]:
        """select, registry, eval, model, text, the data sections, subsample, the pipeline split names and the seed."""
        names = ("select", "registry", "eval", "model", "text", "subsample", "data", "split", "filter", "pipeline")
        slice_: dict[str, Any] = {"seed": config.runtime.seed, **{n: section(config, n) for n in names}}
        export = section(config, "export")
        if export and export != ExportConfig().model_dump(mode="json"):  # only a CHANGED export section counts, so experiments that finished before the export existed keep their completed stage
            slice_["export"] = export
        return slice_

    def run(self, ctx: StageContext) -> StageResult:
        """Select and register. Errors: OrchestratorError (exit code 2) for a registry inside the experiments, a missing baseline or finalist result; SelectorError (exit code 4, also when
        the Test was already looked at); RegistryError (exit code 4); RegistryVerifyError (exit code 5)."""
        services = ctx.services
        if services.selection is None or services.registry is None:
            raise OrchestratorError("The select_and_register stage needs the selection and registry services, but this run has none wired. Use `python -m stt run`.", exit_code=2)
        experiment = ctx.experiment
        validate_registry_location(Path(str(section(ctx.config, "run").get("experiments_dir", "experiments"))), services.registry.location())
        existing = [e for e in services.registry.entries() if e.experiment_id == experiment.experiment_id]
        if existing:
            entry = existing[-1]
            logger.info("This experiment is already registered as %s: nothing is selected or registered again.", entry.entry_id)
            exported, export_files = self._automatic_export(ctx, entry.entry_id, entry.kind, rerun=True)
            return StageResult("ok", {"verdict": "winner" if entry.kind == "model" else "baseline_retained", "entry_id": entry.entry_id, "already_registered": 1, "registry_dir": str(services.registry.location()), "export": exported}, [entry.card_path, *export_files])
        baseline = self._baseline(ctx)
        candidates = load_finalists(experiment)
        test_split = pipeline_value(ctx.config, "test_split", "test")

        def evaluate_test(candidate: Candidate | None) -> EvalResult:
            label = "Test, baseline" if candidate is None else f"Test, finalist trial {candidate.trial_id}"
            return evaluate_model(ctx, test_split, None if candidate is None else candidate.model_dir, label, allow_test=True)

        guard = TestOnceGuard(once_per_experiment(TestLookStore(experiment), evaluate_test))  # ONE guard for the experiment; the Test is reached only through it
        selection = services.selection.select(SelectionRequest(baseline, candidates, services.evaluation, guard, test_split))
        info = experiment.run_info()
        model_section = section(ctx.config, "model")
        prepared_fp = self._fingerprints(ctx)
        card = card_from_selection(
            selection,
            baseline,
            base_model_name=Path(str(model_section.get("path", ""))).name,
            base_model_path=str(model_section.get("path", "")),
            data_fingerprint=prepared_fp.get("data"),
            split_fingerprint=prepared_fp.get("split"),
            subsample_fingerprint=prepared_fp.get("subsample"),
            config_ref=experiment.config_ref,
            library_versions=dict(info.library_versions),
            seed=info.seed,
            code_version=info.code_version,
            generated_at=(services.now or (lambda: info.created_at))(),
        )
        who = f"the winner (trial {selection.winner.trial_id})" if selection.winner is not None else "the baseline"
        review = render_review(selection.review_list, f"Review list of {who} on the {pipeline_value(ctx.config, 'eval_split', 'val')} split ({experiment.experiment_id})")
        # no finalist folder is named: the registry deletes nothing (owner rule, DECISIONS 56)
        entry = services.registry.register(RegisterRequest(selection, card, experiment.experiment_id, [], review))
        registry_dir = str(services.registry.location())
        artifacts = [
            experiment.write_report("selection.json", reporting.selection_json(selection, entry.entry_id, registry_dir)),
            experiment.write_report("selection_table.csv", reporting.selection_table_csv(selection)),
            experiment.write_report("selection_reason.txt", reporting.selection_text(selection, baseline, entry.entry_id, registry_dir)),
        ]
        summary: dict[str, str | float | int] = {
            "verdict": selection.verdict,
            "winner_trial": selection.winner.trial_id if selection.winner is not None else "none",
            "baseline_val_wer": round(baseline.wer, 4),
            "winner_val_wer": round(selection.winner.val.wer, 4) if selection.winner is not None else "none",
            "test_status": selection.test_status,
            "not_confirmed": int(selection.test_status == "not_confirmed"),
            "entry_id": entry.entry_id,
            "entry_kind": entry.kind,
            "registry_dir": registry_dir,
            "reason": selection.reason,
        }
        rows = list(selection.comparison_table)  # the best model for the improvement line: the winner's row, else the best finalist (the first row), as the report shows it
        best = next((r for r in rows if selection.winner is not None and r.trial_id == selection.winner.trial_id), rows[0] if rows else None)
        if best is not None:
            summary.update({"best_trial": best.trial_id, "best_val_wer": round(best.val_wer, 4), "best_val_verdict": str(best.verdict)})
        if selection.test_baseline is not None and selection.test_winner is not None:
            summary.update({"test_baseline_wer": round(selection.test_baseline.wer, 4), "test_winner_wer": round(selection.test_winner.wer, 4)})
        if selection.test_error:
            summary["test_error"] = selection.test_error
        summary["export"], export_files = self._automatic_export(ctx, entry.entry_id, entry.kind, rerun=False)  # an ExportError here is exit code 5; the registration above stays
        return StageResult("ok", summary, [*artifacts, entry.card_path, *export_files])

    @staticmethod
    def _automatic_export(ctx: StageContext, entry_id: str, kind: str, rerun: bool) -> tuple[str, list[Path]]:
        """The automatic export after a registration: (a short text for the summary, the files it made). Skipped and logged for a retained baseline or when no export service is wired.

        On a rerun of an already registered experiment: an entry with no export yet is exported; an entry whose LATEST export of some format is `failed` fails again (the failure is not hidden);
        otherwise nothing is exported again. Errors: ExportError (exit code 5), ExportConfigError (exit code 2)."""
        service = ctx.services.export
        if service is None:
            return "not wired", []
        if kind != "model":
            logger.info("The registry entry %s kept the baseline: nothing to export, the automatic export is skipped.", entry_id)
            return "skipped (the baseline was retained: nothing to export)", []
        now = (ctx.services.now or (lambda: ""))()
        if rerun:
            latest: dict[str, ExportRecord] = {}
            for record in service.records(entry_id):
                latest[record.exporter] = record
            failed = [name for name, record in latest.items() if record.status == "failed"]
            if failed:
                raise ExportError(f"The latest export of {entry_id} with {', '.join(failed)} failed (see its exports.json). Look at the detail there, fix the cause, and repeat it with `python -m stt export {entry_id}`.")
            if latest:
                return "done earlier (" + ", ".join(f"{n} {r.status}" for n, r in sorted(latest.items())) + ")", []
        records = service.export(entry_id, None, explicit=False, exported_at=now)
        entry_dir = ctx.services.registry.location() / entry_id if ctx.services.registry is not None else None
        files = [entry_dir / "exports.json"] if entry_dir is not None and records else []
        return ", ".join(f"{r.exporter} {r.status}" for r in records) or "nothing to export", files

    @staticmethod
    def _baseline(ctx: StageContext) -> EvalResult:
        path = ctx.experiment.paths.reports / BASELINE_RESULT_FILE
        try:
            return eval_result_from_json(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise OrchestratorError(f"The baseline Val result {path} is missing ({exc}). The evaluate stage writes it: run `python -m stt run` (or the evaluate stage) for this experiment first.", exit_code=2) from exc

    @staticmethod
    def _fingerprints(ctx: StageContext) -> dict[str, str | None]:
        prepare = ctx.previous.get("prepare")
        summary: Mapping[str, Any] = prepare.summary if prepare is not None else {}
        if not summary.get("data_fingerprint"):
            stored = load_result(ctx.experiment, "prepare")
            summary = stored.summary if stored is not None else {}
        return {kind: (str(summary[f"{kind}_fingerprint"]) if summary.get(f"{kind}_fingerprint") else None) for kind in ("data", "split", "subsample")}

