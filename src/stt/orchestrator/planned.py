"""The planned time at the start of `run`, `search` and `learning-curve` (Wave 3 task 7; DECISIONS 78, O114).

How to run: this module is not a script; the Orchestrator calls `announce_plan` right after the `prepare` stage (the Val and Test clip counts are only known then). Example console lines:
    PLAN: planned time of `python -m stt run` with the current settings: about 2 h 58 min (from the benchmark of exp_0003: whisper-tiny, 2 thread(s), 4.77 s per step, 1.93 s per clip; Val 875 clips, Test 875 clips)
    PLAN:       28 min  baseline on Val (875 clips, WER)
    ...
and when no benchmark fits:
    PLAN: no benchmark numbers for this model and thread count: run python -m stt benchmark (the run goes on without a planned time)
Where the numbers come from (no new key): the LATEST experiment in the experiments folder whose `reports/benchmark.json` was measured with the SAME model folder name and the SAME thread count as this run.
Nothing is measured or evaluated here; a problem while working it out is logged and never stops the run. `run_sizes` is also used by the `benchmark` stage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from stt.orchestrator.planning import Measured, Projection, Sizes, format_duration, project
from stt.orchestrator.stagehelpers import pipeline_value, prepared_data, section
from stt.orchestrator.types import PreparedData, StageContext
from stt.trainercore import TrainConfig, resolve_threads

logger = logging.getLogger("stt.orchestrator")
NO_BENCHMARK = "no benchmark numbers for this model and thread count: run python -m stt benchmark"


@dataclass(frozen=True)
class BenchmarkSource:
    """A stored benchmark that fits this run: the experiment that holds it and what it measured."""

    experiment_id: str
    measured: Measured


def run_sizes(ctx: StageContext, prepared: PreparedData) -> Sizes:
    """The sizes of the configured run (from the config and the clip COUNTS of the prepared data; nothing is evaluated or built to get them)."""
    eval_split, test_split = pipeline_value(ctx.config, "eval_split", "val"), pipeline_value(ctx.config, "test_split", "test")
    search = section(ctx.config, "search")
    val_files = len(prepared.kept[eval_split])
    return Sizes(
        max_steps=ctx.services.training.hyperparams().max_steps,
        eval_every_steps=int(section(ctx.config, "train")["eval_every_steps"]),
        n_trials=int(search["n_trials"]),
        finalists_k=int(search["finalists"]["k"]),
        parallel_trials=int(search["parallel_trials"]),
        screening_val=min(val_files, int(section(ctx.config, "subsample").get("val_samples") or val_files)),  # the Val slice of a trial: a COUNT from the config, no subset is made
        val_files=val_files,
        test_files=len(prepared.kept.get(test_split, ())),  # a COUNT only: the Test split is not built and not scored
    )


def find_benchmark(experiments_dir: Path, model_name: str, threads: int) -> BenchmarkSource | None:
    """The latest experiment (highest number) whose `reports/benchmark.json` was measured with `model_name` and `threads`, or None. A file that cannot be read is skipped."""
    if not experiments_dir.is_dir():
        return None
    for folder in sorted((p for p in experiments_dir.glob("exp_*") if p.is_dir()), key=lambda p: p.name, reverse=True):
        try:
            measured = Measured(**json.loads((folder / "reports" / "benchmark.json").read_text(encoding="utf-8"))["measured"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if measured.model == model_name and measured.threads == threads:
            return BenchmarkSource(folder.name, measured)
    return None


def plan_lines(kind: str, source: BenchmarkSource, sizes: Sizes, projection: Projection, points: int = 0) -> list[str]:
    """The `PLAN:` lines for the command `kind` (`run`, `search` or `learning-curve`; `points` = the number of learning-curve points)."""
    m = source.measured
    if kind == "search":
        parts = [projection.parts[1]]
    elif kind == "learning-curve":
        per_point = projection.one_training_seconds + sizes.screening_val * m.seconds_per_decode_clip
        parts = [(f"learning curve ({points} points: each a training of {sizes.max_steps} steps and a WER scoring of {sizes.screening_val} Val clips)", points * per_point)]
    else:
        parts = list(projection.parts)
    total = sum(seconds for _, seconds in parts)
    lines = [
        f"PLAN: planned time of `python -m stt {kind}` with the current settings: about {format_duration(total)} (from the benchmark of {source.experiment_id}: {m.model}, {m.threads} thread(s), "
        f"{m.seconds_per_step:.2f} s per step, {m.seconds_per_decode_clip:.2f} s per clip; Val {sizes.val_files} clips, Test {sizes.test_files} clips)"
    ]
    lines += [f"PLAN: {format_duration(seconds):>10}  {label}" for label, seconds in parts]
    lines.append("PLAN: a rough estimate: early stopping and pruning make a run shorter; the first training steps are slower; the Test part counts only when a finalist wins.")
    return lines


def announce_plan(ctx: StageContext, kind: str) -> list[str]:
    """Write the planned total time of the command `kind` (see the module docstring) and return the lines. Errors: none are raised for a missing or unreadable benchmark (a line says so)."""
    model_name = Path(str(section(ctx.config, "model").get("path", ""))).name
    threads = resolve_threads(ctx.config.section("train", TrainConfig))
    experiments_dir = ctx.experiment.paths.root.parent
    source = find_benchmark(experiments_dir, model_name, threads)
    if source is None:
        lines = [f"PLAN: {NO_BENCHMARK} (the run goes on without a planned time)"]
    else:
        prepared = prepared_data(ctx)
        sizes = run_sizes(ctx, prepared)
        points = len(section(ctx.config, "learning_curve").get("presets") or [])
        lines = plan_lines(kind, source, sizes, project(source.measured, sizes), points)
    for line in lines:
        logger.info("%s", line)
    return lines
