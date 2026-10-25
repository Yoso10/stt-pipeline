r"""The `final-test` command: score the Test split ONCE for the trained model and the baseline (DECISIONS 90).

How to run (PowerShell, project folder, after `$env:PYTHONPATH = "src"`; the experiment is the one where `train` and `evaluate` ran, and the same --config / --set values):
    python -m stt final-test --config configs\my_work_cpu.yaml --experiment exp_0003
Example output (last lines): `FINAL TEST: confirmed` and the file `reports\test_result.json`.
What it does: loads the trained model of the experiment and the baseline model, scores both on the Test split (the encoder cache is reused when it is on), compares them with the same paired
bootstrap and the same rule as the Selector (the Test CONFIRMS only when the verdict is `improved` at the level of `eval.significance`, O81) and writes `reports/test_result.json`:
WER, CER, the relative reduction, the verdict and the status `confirmed` or `not_confirmed`.
SAFETY (the Test is looked at once): it REFUSES with exit code 3 when this experiment already has a `test_result.json`, when the marker file `test_split_scored.json` exists in the experiments
folder, or when any experiment of that folder has Test results from `python -m stt run` (`test_once.json`, `test_result_*.json`); it refuses with exit code 2 when the experiment has no
trained model. The marker is written BEFORE the scoring starts, so a stopped scoring also counts as a look; to go on knowingly after a crash, RENAME the marker by hand (nothing deletes it).
The command is never called by `run` and never by the work script unless `-WithTest` is given.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from stt.config import AppConfig
from stt.evaluator import EvalResult
from stt.orchestrator.errors import OrchestratorError
from stt.orchestrator.stageio import load_result
from stt.orchestrator.stagehelpers import evaluate_model, pipeline_value, section
from stt.orchestrator.types import StageContext, StageResult
from stt.runmanager import ExperimentHandle

logger = logging.getLogger("stt.orchestrator")
MARKER_FILE = "test_split_scored.json"
RESULT_FILE = "test_result.json"


def _trained_model(experiment: ExperimentHandle, previous: Mapping[str, StageResult] | None = None) -> Path | None:
    train = (previous or {}).get("train") or load_result(experiment, "train")
    folder = train.summary.get("model_dir") if train is not None else None
    return Path(str(folder)) if folder and Path(str(folder)).is_dir() else None


def earlier_test_looks(experiments_dir: Path) -> list[str]:
    """What shows that a Test split was scored before in this experiments folder: the marker, a `test_result.json`, or the Test records of `python -m stt run` (as readable lines)."""
    found: list[str] = []
    if (experiments_dir / MARKER_FILE).is_file():
        found.append(f"the marker {experiments_dir / MARKER_FILE}")
    if experiments_dir.is_dir():
        for folder in sorted(p for p in experiments_dir.iterdir() if p.is_dir()):
            reports = folder / "reports"
            if (reports / RESULT_FILE).is_file():
                found.append(f"{folder.name}: {RESULT_FILE}")
            if (reports / "test_once.json").is_file() or any(reports.glob("test_result_*.json")):
                found.append(f"{folder.name}: the Test look of `python -m stt run` (test_once.json / test_result_*.json)")
    return found


def check_allowed(experiment: ExperimentHandle, previous: Mapping[str, StageResult] | None = None) -> Path:
    """The refusals. Returns the folder of the trained model. Errors: OrchestratorError, exit code 2 (no trained model) or 3 (the Test was scored before)."""
    if _trained_model(experiment, previous) is None:
        raise OrchestratorError(
            f"{experiment.experiment_id} has no trained model, so there is nothing to score on the Test split. Run `python -m stt train` and `python -m stt evaluate --experiment {experiment.experiment_id}` first "
            "(final-test needs the experiment where the training ran).",
            exit_code=2,
        )
    looks = earlier_test_looks(experiment.paths.root.parent)
    if looks:
        raise OrchestratorError(
            "REFUSED: the Test split was already scored (" + "; ".join(looks) + "). The Test is looked at ONCE; a second look would make it a second validation set. "
            f"If the first look did not finish and you must score again knowingly, RENAME the marker {experiment.paths.root.parent / MARKER_FILE} by hand (nothing deletes it) and use a new experiment.",
            exit_code=3,
        )
    return _trained_model(experiment, previous)  # type: ignore[return-value]


def _numbers(result: EvalResult) -> dict[str, Any]:
    return {"wer": result.wer, "cer": result.cer, "n_files": result.n_files, "n_words": result.n_words, "flags": dict(result.flags)}


class FinalTestStage:
    """final_test: the one-time Test scoring of the trained model and the baseline (see the module docstring)."""

    name = "final_test"
    depends_on: tuple[str, ...] = ("prepare",)

    def config_slice(self, config: AppConfig) -> Mapping[str, Any]:
        """model, features, eval, the pipeline split names, text and the seed (the same slice as the evaluate stage, so the same cache and settings)."""
        return {"seed": config.runtime.seed, **{n: section(config, n) for n in ("model", "features", "eval", "pipeline", "text")}}

    def run(self, ctx: StageContext) -> StageResult:
        """Score the Test split once. Errors: OrchestratorError (exit code 2 or 3, see `check_allowed`); evaluation errors (exit code 4)."""
        experiment = ctx.experiment
        trained = check_allowed(experiment, ctx.previous)
        test_split = pipeline_value(ctx.config, "test_split", "test")
        marker = experiment.paths.root.parent / MARKER_FILE
        marker.write_text(json.dumps({"experiment_id": experiment.experiment_id, "split": test_split, "note": "The Test split is scored (or was started) by `python -m stt final-test`. Rename this file by hand only if you must score again knowingly."}, indent=1), encoding="utf-8")  # BEFORE the scoring
        logger.info("FINAL TEST: scoring the %s split once, for the baseline and for %s.", test_split, trained.name)
        baseline = evaluate_model(ctx, test_split, None, "Test, baseline", allow_test=True)
        model = evaluate_model(ctx, test_split, trained, "Test, trained model", allow_test=True)
        comparison = ctx.services.evaluation.compare(baseline, model)
        status = "confirmed" if comparison.verdict == "improved" else "not_confirmed"
        reduction = (baseline.wer - model.wer) / baseline.wer if baseline.wer > 0 else None
        body = {
            "split": test_split,
            "experiment_id": experiment.experiment_id,
            "eval_version": baseline.eval_version,
            "trained_model": trained.name,
            "baseline": _numbers(baseline),
            "trained": _numbers(model),
            "relative_reduction": reduction,
            "delta_wer": comparison.delta_wer,
            "ci_low": comparison.ci_low,
            "ci_high": comparison.ci_high,
            "confidence": comparison.confidence,
            "n_groups": comparison.n_groups,
            "verdict": comparison.verdict,
            "status": status,
            "explanation": comparison.explanation,
            "rule": "confirmed only when the paired bootstrap verdict is 'improved' at the level of eval.significance (the rule of the Selector, O81)",
        }
        path = experiment.write_report(RESULT_FILE, json.dumps(body, indent=1, ensure_ascii=False))
        logger.info("FINAL TEST: %s. Baseline WER %.4f, trained WER %.4f, relative reduction %s. %s", status, baseline.wer, model.wer, "n/a" if reduction is None else f"{reduction * 100:.1f}%", comparison.explanation)
        return StageResult("ok", {"test_status": status, "test_baseline_wer": round(baseline.wer, 4), "test_trained_wer": round(model.wer, 4), "test_verdict": comparison.verdict, "test_relative_reduction": "n/a" if reduction is None else round(reduction, 4)}, [path])
