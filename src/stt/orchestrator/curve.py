"""The `learning_curve` stage: the same settings trained on increasing Train sizes, scored on the same Val set (Orchestrator Part 3, DECISIONS 59).

How to run: `python -m stt learning-curve --config my.yaml` (PowerShell, project root, after `$env:PYTHONPATH = "src"`); `--params-from exp_0003` takes the hyperparameters from an earlier experiment. Example
output (the last lines of the console, a synthetic example):
    READING: STILL IMPROVING. From 12.0 to 24.0 minutes of Train audio the Val WER fell from 0.9100 to 0.8300 ...
What it does: for each Subsampler preset of `learning_curve.presets` (default smoke, tiny, half, full; built-in or from `subsample.presets`) it takes that many Train samples (scope `train_only`, so the
Val set is the SAME for every size; `subsample.val_samples` makes it a slice), trains a fresh model with the SAME hyperparameters (`train.hyperparams`, or with `learning_curve.params_from` / `--params-from`
the selected winner's searched values of that experiment, or its best finalist's when the baseline was retained) for the same number of steps, and scores it by WER on the Val set. The table lists the Train size
(samples and minutes) against the Val WER, CER and Val loss, with the paired comparison of each point with the previous size. The READING is the verdict of the Evaluator's comparison (the paired bootstrap of
`eval.significance`, the same level of evidence as the selection) of the last two sizes: `improved` = still improving (more data is likely to help), `no_clear_difference` = flat, `worse` = got worse; with too
few Val groups for a confidence statement the plain difference is read and the text says so. Files: `candidates/exp_NNNN_curve_<preset>/`, `metrics/curve_<preset>/`, `reports/curve/<preset>.json` and
`<preset>.val.json` (the lossless Val result), `reports/learning_curve.csv`, `learning_curve.json`, `learning_curve.txt`. A finished point is not trained again; an interrupted one resumes. Nothing is deleted.
The Test split is never built or scored. A size larger than the data FAILS in the Subsampler with the failing size (exit code 3).
"""

from __future__ import annotations

import csv
import io
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stt.config import AppConfig
from stt.evaluator import ComparisonResult, EvalResult, eval_result_from_json, eval_result_to_json
from stt.orchestrator import reporting
from stt.orchestrator.errors import OrchestratorError, StageInterrupted
from stt.orchestrator.stagehelpers import evaluate_model, prepared_data, section
from stt.orchestrator.trials import prepare_trial_folder
from stt.orchestrator.types import StageContext, StageResult
from stt.searcher import apply_params
from stt.trainercore import TrainRequest

logger = logging.getLogger("stt.orchestrator")
FOLDER = "curve"


@dataclass(frozen=True)
class CurvePoint:
    """One size of the curve with its Val result."""

    preset: str
    n_train: int
    minutes: float
    steps: int
    val_loss: float | None
    seconds: float
    model_dir: str
    val: EvalResult


def read_params_from(experiments_dir: Path, name: str) -> tuple[dict[str, Any], str]:
    """The searched hyperparameter values of an earlier experiment: its selected winner's, or its best finalist's when the baseline was retained.

    Returns (values, a plain-words source). Errors: OrchestratorError (exit code 2) when the experiment or its search result is missing (the existing experiments are listed).
    """
    root = experiments_dir / name
    if not root.is_dir():
        known = sorted(p.name for p in experiments_dir.glob("exp_*") if p.is_dir()) if experiments_dir.is_dir() else []
        raise OrchestratorError(f"learning_curve.params_from names '{name}', but {root} does not exist. Existing experiments: {', '.join(known) or 'none'}.", exit_code=2)
    try:
        trials = json.loads((root / "reports" / "search_trials.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise OrchestratorError(f"The experiment {name} has no readable search result ({root / 'reports' / 'search_trials.json'}: {exc}). Run `python -m stt search` or `run` for it first.", exit_code=2) from exc
    params = {int(t["trial_id"]): dict(t["params"]) for t in trials["trials"]}
    winner = None
    try:
        winner = json.loads((root / "reports" / "selection.json").read_text(encoding="utf-8")).get("winner_trial")
    except (OSError, ValueError):
        pass
    if winner is not None and int(winner) in params:
        return params[int(winner)], f"the selected winner (trial {winner}) of {name}"
    if trials.get("finalists"):
        best = int(trials["finalists"][0])
        return params[best], f"the best finalist (trial {best}) of {name}" + (" (its baseline was retained)" if winner is None and (root / "reports" / "selection.json").is_file() else " (no selection yet)")
    raise OrchestratorError(f"The experiment {name} has neither a selected winner nor a finalist to take the hyperparameters from.", exit_code=2)


def reading(points: list[CurvePoint], compare: Callable[[EvalResult, EvalResult], ComparisonResult]) -> tuple[str, str, list[ComparisonResult | None]]:
    """(trend, plain-words reading, the comparison of each point with the previous size). Trends: too_few, improving, flat, worse, unclear."""
    steps: list[ComparisonResult | None] = [None] + [compare(a.val, b.val) for a, b in zip(points, points[1:], strict=False)]
    if len(points) < 2:
        return "too_few", "A curve needs at least two sizes: list more presets in learning_curve.presets.", steps
    a, b, last = points[-2], points[-1], steps[-1]
    assert last is not None
    span = f"From {a.minutes:.1f} to {b.minutes:.1f} minutes of Train audio ({a.n_train} to {b.n_train} samples) the Val WER went from {a.val.wer:.4f} to {b.val.wer:.4f}"
    if last.verdict in ("insufficient_groups", "not_comparable"):
        delta = b.val.wer - a.val.wer
        trend = "improving" if delta < 0 else "worse" if delta > 0 else "flat"
        word = {"improving": "STILL IMPROVING by the plain difference", "worse": "GOT WORSE by the plain difference", "flat": "FLAT by the plain difference"}[trend]
        return "unclear", f"{word} ({span}), but there is no confidence statement: {last.explanation}", steps
    if last.verdict == "improved":
        return "improving", f"STILL IMPROVING. {span}, a gain of {last.delta_wer:+.4f} that holds at the {last.confidence:.0%} level: more data of this kind is likely to help.", steps
    if last.verdict == "worse":
        return "worse", f"GOT WORSE. {span}, a change of {last.delta_wer:+.4f} that holds at the {last.confidence:.0%} level: check the data of the larger size and the number of steps (they are the same for every size).", steps
    return "flat", f"FLAT. {span}: no clear difference at the {last.confidence:.0%} level, so more data of this kind is unlikely to help much at these settings.", steps


def curve_text(points: list[CurvePoint], steps: list[ComparisonResult | None], trend_reading: str, source: str, val_files: int) -> str:
    """The table and the reading in plain words."""
    lines = [
        f"LEARNING CURVE: the same hyperparameters ({source}) and the same number of steps on increasing Train sizes; Val WER on the same {val_files} Val files for every size.",
        "",
        "preset     Train samples  minutes   Val WER   Val CER  Val loss  steps   vs the previous size",
    ]
    for p, c in zip(points, steps, strict=True):
        cer = "-" if p.val.cer is None else f"{p.val.cer:.4f}"
        loss = "-" if p.val_loss is None else f"{p.val_loss:.4f}"
        versus = "-" if c is None else f"{c.verdict} ({c.delta_wer:+.4f})" if c.delta_wer is not None else c.verdict
        lines.append(f"{p.preset:<10} {p.n_train:>13}  {p.minutes:>7.1f}  {p.val.wer:>8.4f}  {cer:>8}  {loss:>8}  {p.steps:>5}   {versus}")
    return "\n".join(lines) + f"\n\nREADING: {trend_reading}\n"


def curve_csv(points: list[CurvePoint], steps: list[ComparisonResult | None]) -> str:
    """The curve data as CSV (UTF-8 with a byte order mark so Excel opens it)."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["preset", "train_samples", "train_minutes", "val_wer", "val_cer", "val_loss", "steps", "train_seconds", "vs_previous_verdict", "vs_previous_delta_wer"])
    for p, c in zip(points, steps, strict=True):
        writer.writerow([p.preset, p.n_train, f"{p.minutes:.2f}", f"{p.val.wer:.4f}", "" if p.val.cer is None else f"{p.val.cer:.4f}", "" if p.val_loss is None else f"{p.val_loss:.5f}", p.steps, f"{p.seconds:.1f}",
                         "" if c is None else c.verdict, "" if c is None or c.delta_wer is None else f"{c.delta_wer:.4f}"])
    return "﻿" + out.getvalue()


class LearningCurveStage:
    """learning_curve: train on each preset size and score on Val."""

    name = "learning_curve"
    depends_on: tuple[str, ...] = ("prepare",)

    def config_slice(self, config: AppConfig) -> Mapping[str, Any]:
        """learning_curve, train, model, features, eval, the data sections, subsample, the pipeline split names and the seed."""
        names = ("learning_curve", "train", "model", "features", "eval", "subsample", "data", "split", "text", "filter", "pipeline")
        return {"seed": config.runtime.seed, **{n: section(config, n) for n in names}}

    def run(self, ctx: StageContext) -> StageResult:
        """Train and score every size. Errors: OrchestratorError (exit code 2) for a missing `params_from` experiment; SubsampleError (exit code 3) for a size larger than the data; training and
        evaluation errors (exit code 4); StageInterrupted on a stop request (a rerun continues)."""
        services, experiment = ctx.services, ctx.experiment
        service = services.search
        if service is None:
            raise OrchestratorError("The learning curve needs the Subsampler service (wired with the search service), but this run has none.", exit_code=2)
        settings = section(ctx.config, "learning_curve")
        base = services.training.hyperparams()
        params_from = settings.get("params_from")
        if params_from:
            experiments_dir = Path(str(section(ctx.config, "run").get("experiments_dir", "experiments")))
            params, source = read_params_from(experiments_dir, str(params_from))
            hp = apply_params(base, params)
        else:
            hp, source = base, "train.hyperparams of the config"
        logger.info("Learning curve on %s with the hyperparameters from %s.", ", ".join(settings["presets"]), source)
        prepared = prepared_data(ctx)
        val_ids: list[str] | None = None
        for preset in settings["presets"]:
            subset = service.subset(prepared, preset)  # a size larger than the data fails here, naming the size
            ids = [n.sample.id for n in subset.val]
            if val_ids is not None and ids != val_ids:
                raise OrchestratorError(f"The Val set of the preset '{preset}' differs from the earlier presets: a learning curve needs the same Val set for every size. Use presets with scope train_only.", exit_code=2)
            val_ids = ids
            self._point(ctx, preset, subset, hp)
        points = self._load(experiment, settings["presets"])
        trend, text_reading, steps = reading(points, services.evaluation.compare)
        report = curve_text(points, steps, text_reading, source, points[0].val.n_files)
        for line in report.splitlines():
            logger.info("%s", line)
        body = {
            "hyperparameters_from": source,
            "points": [dict(preset=p.preset, train_samples=p.n_train, train_minutes=round(p.minutes, 3), val_wer=p.val.wer, val_cer=p.val.cer, val_loss=p.val_loss, steps=p.steps, model_dir=p.model_dir) for p in points],
            "trend": trend,
            "reading": text_reading,
            "val_files": points[0].val.n_files,
        }
        artifacts = [
            experiment.write_report("learning_curve.txt", report),
            experiment.write_report("learning_curve.csv", curve_csv(points, steps)),
            experiment.write_report("learning_curve.json", json.dumps(body, indent=1, ensure_ascii=False)),
        ]
        rows = [{"preset": p.preset, "train_minutes": round(p.minutes, 1), "train_samples": p.n_train, "val_wer": round(p.val.wer, 4)} for p in points]
        summary: dict[str, str | float | int] = {"points": len(points), "trend": trend, "reading": text_reading, "table": json.dumps(rows, ensure_ascii=False), "hyperparameters_from": source, "val_files": points[0].val.n_files}
        return StageResult("ok", summary, artifacts)

    @staticmethod
    def _point(ctx: StageContext, preset: str, subset: Any, hp: Any) -> None:
        experiment, services = ctx.experiment, ctx.services
        state_file, val_file = f"{FOLDER}/{preset}.json", f"{FOLDER}/{preset}.val.json"
        state_path = experiment.paths.reports / state_file
        prefix = f"{experiment.experiment_id}_curve_{preset}"
        state: dict[str, Any] | None = None
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if not Path(state["model_dir"]).is_dir():
                state = None
        except (OSError, ValueError, KeyError):
            state = None
        if state is None:
            output = experiment.candidate_dir(prefix)
            resume = prepare_trial_folder(experiment, services, output, f"curve_{preset}.params.json", {"preset": preset, "hyperparams": asdict(hp)})
            model = services.models.load()
            features = services.models.features(model)
            train, val = features.build(subset.train, subset.train_split), features.build(subset.val, subset.val_split)
            request = TrainRequest(
                model=model, train=train.dataset, val=val.dataset, collator=features.build_collator(), hyperparams=hp, output_dir=output,
                source_config_ref=experiment.config_ref, model_name=prefix, stop_flag=ctx.stop_flag,
            )
            logger.info("Learning curve point '%s'%s: %d Train samples, %d Val samples.", preset, " (resumed)" if resume else "", len(train.dataset), len(val.dataset))
            result = services.training.train(request, experiment.metrics_dir(f"curve_{preset}"), resume, None)
            steps = result.history[-1].step if result.history else 0
            if result.stopped_reason == "stopped_by_callback" and ctx.stop_flag():
                raise StageInterrupted(f"learning curve point '{preset}' stopped on request at step {steps}; a checkpoint was saved and a rerun resumes it")
            minutes = sum(n.sample.duration_sec for n in subset.train) / 60.0
            state = {"preset": preset, "n_train": len(subset.train), "minutes": minutes, "steps": steps, "val_loss": result.best_val_loss, "seconds": result.timings.total_seconds, "model_dir": str(result.model_dir)}
            experiment.write_report(state_file, json.dumps(state, indent=1, ensure_ascii=False))
        if not (experiment.paths.reports / val_file).is_file():
            scored = evaluate_model(ctx, subset.val_split, Path(state["model_dir"]), f"learning curve '{preset}'", samples=subset.val)
            experiment.write_report(val_file, eval_result_to_json(scored))
            experiment.write_report(f"{FOLDER}/predictions_{preset}.csv", reporting.predictions_csv(scored))

    @staticmethod
    def _load(experiment: Any, presets: list[str]) -> list[CurvePoint]:
        points = []
        for preset in presets:
            state = json.loads((experiment.paths.reports / FOLDER / f"{preset}.json").read_text(encoding="utf-8"))
            val = eval_result_from_json((experiment.paths.reports / FOLDER / f"{preset}.val.json").read_text(encoding="utf-8"))
            points.append(CurvePoint(preset, int(state["n_train"]), float(state["minutes"]), int(state["steps"]), state["val_loss"], float(state["seconds"]), state["model_dir"], val))
        return sorted(points, key=lambda p: (p.minutes, p.n_train))
