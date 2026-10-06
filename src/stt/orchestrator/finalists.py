"""The `finalists` stage: retrain each finalist of the search on the FULL data and score it on the Val split (Orchestrator Part 2, DECISIONS 57).

How to run: this module is not a script; `python -m stt run` runs the stage after `search`. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.orchestrator import default_stage_registry; print(default_stage_registry().names())"
Example output:
    ['evaluate', 'finalists', 'prepare', 'search', 'select_and_register', 'train']
What it does: reads the finalists (the best `search.finalists.k` trials by Val loss) from `reports/search_trials.json`; for each one trains a fresh model with the hyperparameters of
the trial (`apply_params(train.hyperparams, params)`) on the FULL Train split of the prepared data, with the screening Val slice (`subsample.val_samples`) as the Val loss that picks the
best checkpoint; then scores the resulting model on the FULL evaluation split (`pipeline.eval_split`, Val) by WER. Files: `candidates/exp_NNNN_final_trial_TTT/` (model folder
`exp_NNNN_final_trial_TTT_step_SSSSS`), `metrics/final_trial_TTT/`, `reports/finalists/final_trial_TTT.json` (what was trained), `final_trial_TTT.val.json` (the lossless Val result,
the codec of C7) and `reports/finalists_report.txt`. Resume: a finished finalist (both files and the model folder) is not trained again; an interrupted one continues from its
checkpoint when its parameters are the same, otherwise its folder is renamed `..._stale_N` (never deleted). `search.finalist_repeats` above 1 is not supported yet (it needs a naming
scheme for the repeats): the stage refuses it with exit code 2. The Test split is never touched. Nothing is deleted.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from stt.config import AppConfig
from stt.evaluator import eval_result_from_json, eval_result_to_json
from stt.orchestrator import reporting
from stt.orchestrator.errors import OrchestratorError, StageInterrupted
from stt.orchestrator.stagehelpers import evaluate_model, pipeline_value, prepared_data, section
from stt.orchestrator.trials import prepare_trial_folder
from stt.orchestrator.types import StageContext, StageResult
from stt.runmanager import ExperimentHandle, model_name_prefix
from stt.searcher import apply_params
from stt.selector import Candidate
from stt.trainercore import TrainRequest

logger = logging.getLogger("stt.orchestrator")
FOLDER = "finalists"


def final_name(trial_id: int) -> str:
    """`final_trial_003`: the name of a finalist's metrics folder and report files."""
    return f"final_trial_{trial_id:03d}"


def read_finalists(experiment: ExperimentHandle) -> list[tuple[int, dict[str, Any]]]:
    """The finalists of the search as (trial id, parameters), best Val loss first, from `reports/search_trials.json`.

    Errors: OrchestratorError (exit code 2) when the file is missing or damaged (run the search stage first).
    """
    path = experiment.paths.reports / "search_trials.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        params = {int(t["trial_id"]): dict(t["params"]) for t in data["trials"]}
        return [(int(i), params[int(i)]) for i in data["finalists"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise OrchestratorError(f"The search result {path} is missing or damaged ({type(exc).__name__}: {exc}). Run the search stage first (python -m stt run, or python -m stt search).", exit_code=2) from exc


def load_finalists(experiment: ExperimentHandle) -> list[Candidate]:
    """The retrained and Val-scored finalists of this experiment as Selector candidates (the order of the search ranking).

    Errors: OrchestratorError (exit code 2) when a finalist has no stored result (run the finalists stage), or has no Val loss.
    """
    candidates = []
    for trial_id, params in read_finalists(experiment):
        state_path = experiment.paths.reports / FOLDER / f"{final_name(trial_id)}.json"
        val_path = experiment.paths.reports / FOLDER / f"{final_name(trial_id)}.val.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            val = eval_result_from_json(val_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise OrchestratorError(f"The finalist of trial {trial_id} has no stored result ({state_path.name}, {val_path.name}): {exc}. Run the finalists stage first.", exit_code=2) from exc
        if state.get("val_loss") is None:
            raise OrchestratorError(f"The finalist of trial {trial_id} has no Val loss (the training made no evaluation). Check train.eval_every_steps.", exit_code=2)
        candidates.append(Candidate(trial_id, params, Path(state["model_dir"]), val, float(state["val_loss"]), int(state["steps"])))
    return candidates


class FinalistsStage:
    """finalists: retrain on the full data and score on Val."""

    name = "finalists"
    depends_on: tuple[str, ...] = ("prepare", "search")

    def config_slice(self, config: AppConfig) -> Mapping[str, Any]:
        """search, train, model, features, eval, text, the data sections, subsample, the pipeline split names and the seed."""
        return {"seed": config.runtime.seed, **{n: section(config, n) for n in ("search", "train", "model", "features", "eval", "text", "subsample", "data", "split", "filter", "pipeline")}}

    def run(self, ctx: StageContext) -> StageResult:
        """Retrain and score every finalist. Errors: OrchestratorError (exit code 2) for finalist_repeats above 1 or a missing search result; training and evaluation errors (exit code 4);
        StageInterrupted when the stop flag stopped a training (a rerun continues)."""
        repeats = int(section(ctx.config, "search").get("finalist_repeats", 1))
        if repeats > 1:
            raise OrchestratorError(f"search.finalist_repeats is {repeats}, but several seeds per finalist are not supported yet (the model folders of the repeats have no naming scheme). Set search.finalist_repeats to 1.", exit_code=2)
        service = ctx.services.search
        if service is None:
            raise OrchestratorError("The finalists stage needs the search service (for the screening Val slice), but this run has none wired.", exit_code=2)
        experiment = ctx.experiment
        finalists = read_finalists(experiment)
        prepared = prepared_data(ctx)
        screening = service.screening(prepared)
        eval_split = pipeline_value(ctx.config, "eval_split", "val")
        logger.info("Retraining %d finalist(s) on %d Train samples (Val loss on %d samples, WER on the %d samples of '%s').", len(finalists), len(prepared.kept[screening.train_split]), len(screening.val), len(prepared.kept[eval_split]), eval_split)
        artifacts: list[Path] = []
        for trial_id, params in finalists:
            artifacts += self._one(ctx, trial_id, params, screening.train_split, screening.val, screening.val_split, eval_split)
        candidates = load_finalists(experiment)
        report = experiment.write_report("finalists_report.txt", reporting.finalists_report_text(candidates))
        best = min(candidates, key=lambda c: (c.val.wer, c.trial_id))
        rows = [{"trial": c.trial_id, "val_wer": round(c.val.wer, 4), "val_loss": round(c.val_loss, 5), "steps": c.steps, "model": c.model_dir.name} for c in candidates]
        summary: dict[str, str | float | int] = {
            "finalists": len(candidates),
            "best_trial": best.trial_id,
            "best_val_wer": round(best.val.wer, 4),
            "finalist_results": json.dumps(rows),
            "val_split": eval_split,
            "val_files": best.val.n_files,
        }
        return StageResult("ok", summary, [report, *artifacts])

    @staticmethod
    def _one(ctx: StageContext, trial_id: int, params: Mapping[str, Any], train_split: str, val_slice: Any, val_split: str, eval_split: str) -> list[Path]:
        experiment, services = ctx.experiment, ctx.services
        name = final_name(trial_id)
        state_file, val_file = f"{FOLDER}/{name}.json", f"{FOLDER}/{name}.val.json"
        state_path = experiment.paths.reports / state_file
        prefix = model_name_prefix(experiment.experiment_id, "final", trial_id)
        state: dict[str, Any] | None = None
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if not Path(state["model_dir"]).is_dir():
                state = None  # the model folder is gone: train again (the old state file is replaced)
        except (OSError, ValueError, KeyError):
            state = None
        if state is None:
            prepared = prepared_data(ctx)
            output = experiment.candidate_dir(prefix)
            resume = prepare_trial_folder(experiment, services, output, f"{name}.params.json", params)
            model = services.models.load()
            features = services.models.features(model)
            train = features.build(prepared.kept[train_split], train_split)
            val = features.build(val_slice, val_split)
            request = TrainRequest(
                model=model,
                train=train.dataset,
                val=val.dataset,
                collator=features.build_collator(),
                hyperparams=apply_params(services.training.hyperparams(), dict(params)),
                output_dir=output,
                source_config_ref=experiment.config_ref,
                model_name=prefix,
                stop_flag=ctx.stop_flag,
            )
            logger.info("Finalist trial %d%s with %s: %d train and %d val examples.", trial_id, " (resumed)" if resume else "", dict(params), len(train.dataset), len(val.dataset))
            result = services.training.train(request, experiment.metrics_dir(name), resume, None)
            steps = result.history[-1].step if result.history else 0
            if result.stopped_reason == "stopped_by_callback" and ctx.stop_flag():
                raise StageInterrupted(f"finalist trial {trial_id} stopped on request at step {steps}; a checkpoint was saved and a rerun resumes it")
            state = {"trial_id": trial_id, "params": dict(params), "model_dir": str(result.model_dir), "val_loss": result.best_val_loss, "steps": steps, "best_step": result.best_step, "stopped_reason": result.stopped_reason}
            experiment.write_report(state_file, json.dumps(state, indent=1, ensure_ascii=False))
        artifacts = [state_path]
        val_path = experiment.paths.reports / val_file
        if not val_path.is_file():
            val_result = evaluate_model(ctx, eval_split, Path(state["model_dir"]), f"finalist trial {trial_id}")
            experiment.write_report(val_file, eval_result_to_json(val_result))
            experiment.write_report(f"{FOLDER}/predictions_{name}.csv", reporting.predictions_csv(val_result))
            experiment.write_report(f"{FOLDER}/review_{name}.txt", reporting.review_text(val_result))
        artifacts.append(val_path)
        return artifacts
