"""The pipeline stages of the Orchestrator: prepare, train, evaluate, search (they use the services through their Protocols only).

How to run: this module is not a script; `Orchestrator.run` runs the stages by name. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.orchestrator import default_stage_registry; print(default_stage_registry().names())"
Example output:
    ['evaluate', 'finalists', 'prepare', 'search', 'select_and_register', 'train']
prepare  : load, split, normalize and filter the data; write the plain-text reports of each step into `reports/`.
train    : one training run (trial) with the TrainerCore; the model folder is named after the experiment, the trial and the step
           (`exp_0004_trial_001_step_00100`); a trial folder with a complete checkpoint is RESUMED (the Orchestrator sets the resume itself, O69).
evaluate : score the baseline model (and the trained model of this experiment, if there is one) on the Val split, compare them, write the
           evaluation files. It REFUSES the Test split (the Test is scored once, for the winner, through the Selector; Wave 2).
search   : the hyperparameter search (C8) on the screening data: one trial after another (or `search.parallel_trials` worker processes), each trial a fresh model trained
           with its own hyperparameters; the objective is the Val loss; the study is stored in `search/study.json` after every trial, so an interrupted search is RESUMED
           by the same command with `--experiment`; it writes `reports/search_report.txt`, `search_trials.json` and `search_screening.txt`. It does NOT retrain the finalists,
           select or register (the next Orchestrator parts). The Test split is never touched.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from stt.config import AppConfig
from stt.evaluator import EvalResult, eval_result_to_json
from stt.orchestrator import reporting
from stt.orchestrator.errors import OrchestratorError, StageInterrupted
from stt.orchestrator.stageio import load_result
from stt.orchestrator.stagehelpers import BASELINE_RESULT_FILE, evaluate_model, make_items, pipeline_value, prepared_data, section
from stt.orchestrator.trials import SearchTrialRunner
from stt.orchestrator.types import PipelineStage, ScreeningSet, StageContext, StageResult
from stt.searcher import SearchError, SearchInterrupted, SearchResult, SearchSpec
from stt.trainercore import TrainRequest

logger = logging.getLogger("stt.orchestrator")


_section = section  # the helpers moved to stagehelpers.py; the old names stay for the tests and the stages below
_pipeline_value = pipeline_value


class PrepareStage:
    """prepare: the data steps up to the filter."""

    name = "prepare"
    depends_on: tuple[str, ...] = ()

    def config_slice(self, config: AppConfig) -> Mapping[str, Any]:
        """data, split, text, filter and the seed."""
        sections = {n: _section(config, n) for n in ("data", "split", "text", "filter")}
        subsample = _section(config, "subsample")
        if subsample.get("enabled"):  # a disabled Subsampler changes nothing, so older experiments keep their fingerprint (O76)
            sections["subsample"] = subsample
        return {"seed": config.runtime.seed, **sections}

    def run(self, ctx: StageContext) -> StageResult:
        """Load, split, normalize, filter; write the reports. Errors: the components' errors (data errors are exit code 3)."""
        prepared = prepared_data(ctx)
        artifacts: list[Path] = []
        for name, text in prepared.reports.items():
            artifacts.append(ctx.experiment.write_report(name, text))
        summary: dict[str, str | float | int] = dict(prepared.counts)
        summary.update({f"{kind}_fingerprint": value for kind, value in prepared.fingerprints.items() if value})  # for the model card (the subsample one only when the Subsampler is on)
        return StageResult("ok", summary, artifacts)


_items = make_items


class TrainStage:
    """train: one trial."""

    name = "train"
    depends_on: tuple[str, ...] = ("prepare",)

    def config_slice(self, config: AppConfig) -> Mapping[str, Any]:
        """model, features, train and the seed."""
        return {"seed": config.runtime.seed, **{n: _section(config, n) for n in ("model", "features", "train")}}

    @staticmethod
    def _trial(ctx: StageContext) -> tuple[int, bool]:
        """(trial number, resume): the latest trial folder is resumed when it has a complete checkpoint, otherwise a new trial number is used."""
        prefix = f"{ctx.experiment.experiment_id}_trial_"
        numbers = sorted(int(m.group(1)) for p in ctx.experiment.paths.candidates.glob(f"{prefix}*") if (m := re.fullmatch(rf"{re.escape(prefix)}(\d+)", p.name)))
        if numbers and ctx.services.training.can_resume(ctx.experiment.candidate_dir(f"{prefix}{numbers[-1]:03d}")):
            return numbers[-1], True
        return (numbers[-1] + 1 if numbers else 1), False

    def run(self, ctx: StageContext) -> StageResult:
        """Train. Errors: training errors (exit code 4); StageInterrupted when the stop flag stopped the run."""
        prepared = prepared_data(ctx)
        split_names = list(getattr(ctx.config.sections.get("split"), "names", ("train", "val", "test")))
        train_split, val_split = split_names[0], _pipeline_value(ctx.config, "eval_split", "val")
        model = ctx.services.models.load()
        features = ctx.services.models.features(model)
        train = features.build(prepared.kept[train_split], train_split)
        val = features.build(prepared.kept[val_split], val_split)
        trial, resume = self._trial(ctx)
        exp = ctx.experiment.experiment_id
        prefix = f"{exp}_trial_{trial:03d}"
        request = TrainRequest(
            model=model,
            train=train.dataset,
            val=val.dataset,
            collator=features.build_collator(),
            hyperparams=ctx.services.training.hyperparams(),
            output_dir=ctx.experiment.candidate_dir(prefix),
            source_config_ref=ctx.experiment.config_ref,
            model_name=prefix,
            stop_flag=ctx.stop_flag,
        )
        logger.info("Training trial %d%s: %d train and %d val examples.", trial, " (resumed)" if resume else "", len(train.dataset), len(val.dataset))
        result = ctx.services.training.train(request, ctx.experiment.metrics_dir(f"trial_{trial:03d}"), resume)
        if result.stopped_reason == "stopped_by_callback" and ctx.stop_flag():
            raise StageInterrupted(f"training stopped on request at step {result.history[-1].step}; a checkpoint was saved")
        summary: dict[str, str | float | int] = {
            "trial": trial,
            "stopped_reason": result.stopped_reason,
            "best_step": result.best_step,
            "steps": result.history[-1].step if result.history else 0,
            "mean_seconds_per_step": round(result.timings.mean_seconds_per_step, 3),
            "model_dir": str(result.model_dir),
            "resumed": int(resume),
        }
        if result.best_val_loss is not None:
            summary["best_val_loss"] = round(result.best_val_loss, 5)
        return StageResult("ok", summary, [result.model_dir])


class EvaluateStage:
    """evaluate: baseline and, when this experiment has one, the trained model, on the evaluation split (Val)."""

    name = "evaluate"
    depends_on: tuple[str, ...] = ("prepare",)

    def config_slice(self, config: AppConfig) -> Mapping[str, Any]:
        """model, features, eval, the pipeline split names and the seed."""
        return {"seed": config.runtime.seed, **{n: _section(config, n) for n in ("model", "features", "eval", "pipeline", "text")}}

    def _score(self, ctx: StageContext, split: str, path: Path | None, label: str) -> EvalResult:
        result = evaluate_model(ctx, split, path, label)
        ctx.experiment.write_report(f"evaluation_{label}.json", reporting.evaluation_json(result))
        ctx.experiment.write_report(f"predictions_{label}.csv", reporting.predictions_csv(result))
        ctx.experiment.write_report(f"review_{label}.txt", reporting.review_text(result))
        if label == "baseline":  # the lossless copy that the selection stage reads (DECISIONS 57; O82)
            ctx.experiment.write_report(BASELINE_RESULT_FILE, eval_result_to_json(result))
        return result

    def run(self, ctx: StageContext) -> StageResult:
        """Score and compare. Errors: OrchestratorError (exit code 2) for the Test split or an unknown split; evaluation errors (exit code 4)."""
        split = ctx.options.get("split") or _pipeline_value(ctx.config, "eval_split", "val")
        test_split = _pipeline_value(ctx.config, "test_split", "test")
        if split == test_split:
            raise OrchestratorError(
                f"The evaluate stage refuses the Test split ('{split}'): the Test is evaluated ONCE, only for the winner, through the Selector, "
                f"which is built in Wave 2. Evaluate the Val split ('{_pipeline_value(ctx.config, 'eval_split', 'val')}') instead.",
                exit_code=2,
            )
        prepared = prepared_data(ctx)
        if split not in prepared.kept:
            raise OrchestratorError(f"The split '{split}' does not exist. Known splits: {', '.join(sorted(prepared.kept))}. Set pipeline.eval_split or --split.", exit_code=2)
        baseline = self._score(ctx, split, None, "baseline")
        summary: dict[str, str | float | int] = {"split": split, "baseline_wer": round(baseline.wer, 4), "baseline_cer": round(baseline.cer or 0.0, 4), "n_files": baseline.n_files}
        artifacts = [ctx.experiment.paths.reports / "evaluation_baseline.json", ctx.experiment.paths.reports / BASELINE_RESULT_FILE]
        trained = self._trained_model(ctx)
        if trained is not None:
            candidate = self._score(ctx, split, trained, "trained")
            comparison = ctx.services.evaluation.compare(baseline, candidate)
            ctx.experiment.write_report("comparison.json", reporting.comparison_json(comparison))
            summary.update({"trained_wer": round(candidate.wer, 4), "trained_cer": round(candidate.cer or 0.0, 4), "verdict": comparison.verdict, "explanation": comparison.explanation})
            artifacts += [ctx.experiment.paths.reports / "evaluation_trained.json", ctx.experiment.paths.reports / "comparison.json"]
        else:
            summary["trained_wer"] = "no trained model in this experiment (run the train stage)"
        summary["test_split"] = f"'{test_split}' not evaluated (scored once, for the winner, through the Selector in Wave 2)"
        return StageResult("ok", summary, artifacts)

    @staticmethod
    def _trained_model(ctx: StageContext) -> Path | None:
        train = ctx.previous.get("train") or load_result(ctx.experiment, "train")
        folder = train.summary.get("model_dir") if train is not None else None
        return Path(str(folder)) if folder and Path(str(folder)).is_dir() else None


class SearchStage:
    """search: the hyperparameter search on the screening data (see the module docstring)."""

    name = "search"
    depends_on: tuple[str, ...] = ("prepare",)

    def config_slice(self, config: AppConfig) -> Mapping[str, Any]:
        """search, train, model, features, the data sections, subsample and the seed (a larger `search.n_trials` makes the stage run again and continue the study)."""
        return {"seed": config.runtime.seed, **{n: _section(config, n) for n in ("search", "train", "model", "features", "subsample", "data", "split", "text", "filter")}}

    def run(self, ctx: StageContext) -> StageResult:
        """Search. Errors: SearchConfigError (exit code 2), SearchError (exit code 4, the reports of what was done are still written), StageInterrupted when the stop flag
        stopped a trial (the study keeps the finished trials)."""
        service = ctx.services.search
        if service is None:
            raise OrchestratorError("The search stage needs the search service, but this run has none wired. Use `python -m stt search`.", exit_code=2)
        prepared = prepared_data(ctx)
        experiment = ctx.experiment
        spec = service.spec(experiment.paths.search)
        screening = service.screening(prepared)
        screening_report = experiment.write_report("search_screening.txt", screening.report)
        runner = SearchTrialRunner(experiment, screening, service.services_provider, ctx.services, ctx.stop_flag)
        logger.info("Search (%s, up to %d trials, %d parallel) on %d Train and %d Val samples.", spec.strategy, spec.n_trials, spec.parallel_trials, len(screening.train), len(screening.val))
        try:
            result = service.search(spec, runner)
        except SearchInterrupted as exc:
            raise StageInterrupted(str(exc)) from exc
        except SearchError as exc:
            if isinstance(exc.result, SearchResult):  # no trial completed: the reports still say what happened
                self._write(ctx, exc.result, spec, screening)
            raise
        artifacts = [screening_report, *self._write(ctx, result, spec, screening)]
        report = result.report
        summary: dict[str, str | float | int] = {
            "strategy": report.strategy,
            "trials_started": report.n_started,
            "trials_complete": report.n_complete,
            "trials_pruned": report.n_pruned,
            "trials_failed": report.n_failed,
            "trials_from_earlier_run": report.n_resumed,
            "stopped_reason": result.stopped_reason,
            "best_val_loss": round(report.best_val_loss, 5) if report.best_val_loss is not None else "none",
            "best_params": json.dumps(dict(report.best_params or {}), sort_keys=True),
            "finalist_trials": json.dumps([r.trial.trial_id for r in result.ranked]),
            "screening_train_samples": len(screening.train),
            "screening_val_samples": len(screening.val),
            "seconds": round(report.seconds, 1),
            "study_file": str(experiment.paths.search / "study.json"),
        }
        return StageResult("ok", summary, artifacts)

    @staticmethod
    def _write(ctx: StageContext, result: SearchResult, spec: SearchSpec, screening: ScreeningSet) -> list[Path]:
        known = {r.trial.trial_id for r in result.trials}
        pattern = re.compile(rf"{re.escape(ctx.experiment.experiment_id)}_trial_(\d+)(_stale_\d+)?")
        unrecorded = sorted(p.name for p in ctx.experiment.paths.candidates.glob("*") if (m := pattern.fullmatch(p.name)) and (m.group(2) or int(m.group(1)) not in known))
        return [
            ctx.experiment.write_report("search_report.txt", reporting.search_report_text(result, screening, unrecorded)),
            ctx.experiment.write_report("search_trials.json", reporting.search_trials_json(result, spec, screening)),
        ]


class StageRegistry:
    """A read-only name -> stage factory mapping."""

    def __init__(self, factories: Mapping[str, Callable[[], PipelineStage]]) -> None:
        self._factories = dict(factories)

    def names(self) -> list[str]:
        """The registered stage names, sorted."""
        return sorted(self._factories)

    def create(self, name: str) -> PipelineStage:
        """Create the stage `name`. Errors: OrchestratorError listing the registered names."""
        factory = self._factories.get(name)
        if factory is None:
            raise OrchestratorError(f"Unknown stage '{name}'. Registered stages: {', '.join(self.names())}.")
        return factory()


def default_stage_registry() -> StageRegistry:
    """The registry with prepare, train, evaluate, search, finalists and select_and_register."""
    from stt.orchestrator.finalists import FinalistsStage  # imported here: those modules use the helpers of this package, not this module
    from stt.orchestrator.selection import SelectAndRegisterStage

    return StageRegistry({"prepare": PrepareStage, "train": TrainStage, "evaluate": EvaluateStage, "search": SearchStage, "finalists": FinalistsStage, "select_and_register": SelectAndRegisterStage})
