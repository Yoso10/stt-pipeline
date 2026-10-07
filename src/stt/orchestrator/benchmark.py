"""The `benchmark` stage: measure this machine and print the planning numbers (Orchestrator Part 3, DECISIONS 59).

How to run: `python -m stt benchmark --config my.yaml` (PowerShell, project root, after `$env:PYTHONPATH = "src"`). Example output (the last lines of the console):
    BENCHMARK on this machine: model whisper-tiny, 2 thread(s), 3 training step(s) measured.
    Measured: 1.24 s per training step ... 0.415 s per example for the Val loss ... 1.9 s per clip for the WER scoring
    Planned time of `python -m stt run` with the current settings: ... TOTAL 3 h 10 min
What it does: trains `benchmark.steps` steps of the configured model on the first Train samples (the hyperparameters of the config, only `max_steps` replaced), with a small Val probe
(`benchmark.steps` x `eval.batch_size` clips) for the Val-loss time, and scores the same probe clips with the base model by WER for the decoding time; then `planning.project` turns the three numbers
and the sizes of the configured run (`search.n_trials`, `search.finalists.k`, `train.hyperparams.max_steps`, `train.eval_every_steps`, the Val and Test clip COUNTS of the prepared data) into the planned
time of a `run`. Files: `reports/benchmark.txt` and `reports/benchmark.json`; the measuring training is kept in `candidates/exp_NNNN_benchmark/` (nothing is deleted; it is a real model folder, so it
uses disk space like any trial). The Test split is never evaluated and never built (only its clip count is read). The first steps are slower than the later ones: raise `benchmark.steps` for a steadier number.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from stt.config import AppConfig
from stt.orchestrator.errors import OrchestratorError, StageInterrupted
from stt.orchestrator.planned import run_sizes
from stt.orchestrator.resources import folder_size, human_bytes, peak_memory_bytes, resources_text
from stt.orchestrator.planning import Measured, benchmark_text, format_duration, project
from stt.orchestrator.stagehelpers import evaluate_model, pipeline_value, prepared_data, section
from stt.orchestrator.types import StageContext, StageResult
from stt.trainercore import TrainConfig, TrainRequest, resolve_threads
from stt.trainercore.run_folder import checkpoints_dir, list_checkpoints

logger = logging.getLogger("stt.orchestrator")


def free_folder(base: Path) -> Path:
    """`base`, or `base_2`, `base_3`, ... the first that does not exist (a measuring folder is never reused and never deleted)."""
    candidate, number = base, 1
    while candidate.exists():
        number += 1
        candidate = base.with_name(f"{base.name}_{number}")
    return candidate


class BenchmarkStage:
    """benchmark: seconds per step, per Val-loss example and per decoded clip, and the planned time of a run."""

    name = "benchmark"
    depends_on: tuple[str, ...] = ("prepare",)

    def config_slice(self, config: AppConfig) -> Mapping[str, Any]:
        """benchmark, train, model, features, eval, search, the data sections, subsample, the pipeline split names and the seed."""
        names = ("benchmark", "train", "model", "features", "eval", "search", "subsample", "data", "split", "text", "filter", "pipeline")
        return {"seed": config.runtime.seed, **{n: section(config, n) for n in names}}

    def run(self, ctx: StageContext) -> StageResult:
        """Measure and project. Errors: training and evaluation errors (exit code 4), OrchestratorError (exit code 2) for data that is too small to measure; StageInterrupted on a stop request."""
        services, experiment = ctx.services, ctx.experiment
        prepared = prepared_data(ctx)
        names = list(section(ctx.config, "split").get("names") or ["train", "val", "test"])
        train_split, eval_split, test_split = names[0], pipeline_value(ctx.config, "eval_split", "val"), pipeline_value(ctx.config, "test_split", "test")
        steps = int(section(ctx.config, "benchmark")["steps"])
        base = services.training.hyperparams()
        hp = replace(base, max_steps=steps)
        train_samples = prepared.kept[train_split][: steps * hp.batch_size * hp.grad_accum_steps]
        probe = max(1, steps * int(section(ctx.config, "eval")["batch_size"]))
        val_samples = prepared.kept[eval_split][:probe]
        if not train_samples or not val_samples:
            raise OrchestratorError(f"The benchmark needs Train and Val samples ({len(train_samples)} Train, {len(val_samples)} Val after the filter). Check the data and the split.", exit_code=2)
        model = services.models.load()
        features = services.models.features(model)
        train, val = features.build(train_samples, train_split), features.build(val_samples, eval_split)
        output = free_folder(experiment.candidate_dir(f"{experiment.experiment_id}_benchmark"))
        request = TrainRequest(
            model=model, train=train.dataset, val=val.dataset, collator=features.build_collator(), hyperparams=hp, output_dir=output,
            source_config_ref=experiment.config_ref, model_name=output.name, stop_flag=ctx.stop_flag,
        )
        logger.info("Benchmark: %d training step(s) on %d Train examples, Val probe of %d examples.", steps, len(train.dataset), len(val.dataset))
        result = services.training.train(request, experiment.metrics_dir("benchmark"), False, None)
        if result.stopped_reason == "stopped_by_callback" and ctx.stop_flag():
            raise StageInterrupted("the benchmark was stopped on request")
        t = result.timings
        per_val = t.eval_seconds / (t.eval_runs * len(val.dataset)) if t.eval_runs else float("nan")
        scored = evaluate_model(ctx, eval_split, None, "benchmark probe", samples=val_samples)
        per_clip = scored.seconds / max(1, scored.n_files)
        measured = Measured(model.info.name, resolve_threads(ctx.config.section("train", TrainConfig)), steps, hp.batch_size, hp.grad_accum_steps, t.mean_seconds_per_step, per_val, len(val.dataset), per_clip, scored.n_files)
        sizes = run_sizes(ctx, prepared)  # the same numbers the plan at the start of a run uses
        projection = project(measured, sizes)
        checkpoint_folders = [folder for _, folder in list_checkpoints(checkpoints_dir(output))]
        resources = {
            "trial_folder_bytes": folder_size(output),
            "checkpoint_folders": len(checkpoint_folders),
            "largest_checkpoint_bytes": max((folder_size(p) for p in checkpoint_folders), default=0),
            "peak_memory_bytes": peak_memory_bytes(),  # None when it cannot be measured (not Windows, or the call failed)
            "trial_folders_of_a_run": sizes.n_trials + sizes.finalists_k,
        }
        text = benchmark_text(measured, sizes, projection) + "\n" + "\n".join(
            resources_text(resources["trial_folder_bytes"], resources["checkpoint_folders"], resources["largest_checkpoint_bytes"], resources["peak_memory_bytes"], resources["trial_folders_of_a_run"])
        ) + "\n"
        for line in text.splitlines():
            logger.info("%s", line)
        body = {"measured": asdict(measured), "sizes": asdict(sizes), "parts": [{"part": label, "seconds": seconds} for label, seconds in projection.parts], "total_seconds": projection.total_seconds, "notes": list(projection.notes), "resources": resources}
        artifacts = [experiment.write_report("benchmark.txt", text), experiment.write_report("benchmark.json", json.dumps(body, indent=1, ensure_ascii=False))]
        summary: dict[str, str | float | int] = {
            "seconds_per_step": round(measured.seconds_per_step, 3),
            "seconds_per_val_loss_example": round(per_val, 4),
            "seconds_per_decode_clip": round(per_clip, 3),
            "threads": measured.threads,
            "model": measured.model,
            "projected_run_seconds": round(projection.total_seconds),
            "projected_run": format_duration(projection.total_seconds),
            "projection": json.dumps({label: round(seconds) for label, seconds in projection.parts}, ensure_ascii=False),
            "measuring_model_dir": str(result.model_dir),
            "trial_folder": human_bytes(resources["trial_folder_bytes"]),
            "peak_memory": human_bytes(resources["peak_memory_bytes"]) if resources["peak_memory_bytes"] is not None else "not measured",
        }
        return StageResult("ok", summary, artifacts)

