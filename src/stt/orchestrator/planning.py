"""Planning numbers: from what the benchmark measured on this machine to how long a run would take (pure functions, standard library only; Orchestrator Part 3, DECISIONS 59).

How to run: this module is not a script; the `benchmark` stage uses it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.orchestrator.planning import format_duration; print(format_duration(5832))"
Example output:
    1 h 37 min
The model of the projection is simple and honest: a training step costs the same whatever the size of the data (the number of steps is fixed), a Val-loss evaluation costs a fixed time per
example, a WER scoring costs a fixed time per clip. So: baseline = Val clips x decode time; search = trials x (steps x step time + evaluations x slice x Val-loss time); finalists = k x (the same +
all Val clips x decode time); Test = 2 x Test clips x decode time (only when a finalist wins; the Test is counted, never evaluated). Early stopping, pruning and the machine's other work make real runs
shorter or longer; parallel trials share the same cores, so they do not make the total shorter. The numbers belong to THIS machine and THIS model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Measured:
    """What the benchmark measured. The times are seconds; `*_probe` say on how many examples or clips."""

    model: str
    threads: int
    steps: int
    batch_size: int
    grad_accum_steps: int
    seconds_per_step: float
    seconds_per_val_loss_example: float
    val_loss_probe: int
    seconds_per_decode_clip: float
    decode_probe: int


@dataclass(frozen=True)
class Sizes:
    """The sizes of the configured run (from the config and the prepared data; nothing is evaluated to get them)."""

    max_steps: int
    eval_every_steps: int
    n_trials: int
    finalists_k: int
    parallel_trials: int
    screening_val: int
    val_files: int
    test_files: int


@dataclass(frozen=True)
class Projection:
    """The planned seconds of each part of a `run` and their total, plus the plain-words notes."""

    parts: tuple[tuple[str, float], ...]
    total_seconds: float
    one_training_seconds: float
    evaluations_per_training: int
    notes: tuple[str, ...]


def evaluations(max_steps: int, eval_every_steps: int) -> int:
    """How many Val-loss evaluations one training makes: one every `eval_every_steps` steps and one at the end."""
    return max_steps // eval_every_steps + (1 if max_steps % eval_every_steps else 0)


def project(m: Measured, s: Sizes) -> Projection:
    """The projected seconds of a `run` with the sizes `s` and the measured times `m`."""
    n_evals = evaluations(s.max_steps, s.eval_every_steps)
    training = s.max_steps * m.seconds_per_step + n_evals * s.screening_val * m.seconds_per_val_loss_example
    parts = (
        (f"baseline on Val ({s.val_files} clips, WER)", s.val_files * m.seconds_per_decode_clip),
        (f"search ({s.n_trials} trials of {s.max_steps} steps)", s.n_trials * training),
        (f"finalists ({s.finalists_k} retrainings of {s.max_steps} steps)", s.finalists_k * training),
        (f"finalists on Val ({s.finalists_k} x {s.val_files} clips, WER)", s.finalists_k * s.val_files * m.seconds_per_decode_clip),
        (f"Test, winner and baseline (2 x {s.test_files} clips, WER; only when a finalist wins)", 2 * s.test_files * m.seconds_per_decode_clip),
    )
    total = sum(seconds for _, seconds in parts)
    notes = [
        f"a training of {s.max_steps} steps with {n_evals} Val-loss evaluation(s) of {s.screening_val} example(s) takes about {format_duration(training)}",
        "early stopping and pruning can make a run shorter; the first training steps are slower than the later ones",
    ]
    if s.parallel_trials > 1:
        notes.append(f"search.parallel_trials is {s.parallel_trials}: the trials share the same cores, so the total is about the same (each trial gets {max(1, m.threads // s.parallel_trials)} thread(s))")
    if m.val_loss_probe < 8 or m.decode_probe < 8:
        notes.append(f"the Val probe was small ({m.val_loss_probe} examples, {m.decode_probe} clips): raise benchmark.steps or eval.batch_size for steadier per-example times")
    notes.append("the numbers belong to this machine, these threads and this model: measure again on the target machine and model (O66)")
    return Projection(parts, total, training, n_evals, tuple(notes))


def format_duration(seconds: float) -> str:
    """A duration in plain words: `45 s`, `12 min`, `1 h 37 min`, `2 d 3 h`."""
    if not math.isfinite(seconds):
        return "unknown"
    total = int(round(seconds))
    if total < 60:
        return f"{total} s"
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return f"{minutes} min"
    hours, rest = divmod(minutes, 60)
    if hours < 24:
        return f"{hours} h {rest} min"
    days, hours = divmod(hours, 24)
    return f"{days} d {hours} h"


def benchmark_text(m: Measured, s: Sizes, p: Projection) -> str:
    """The benchmark in plain words (the file `reports/benchmark.txt` and the console)."""
    lines = [
        f"BENCHMARK on this machine: model {m.model}, {m.threads} thread(s), {m.steps} training step(s) measured.",
        "",
        "Measured:",
        f"  {m.seconds_per_step:.2f} s per training step (batch {m.batch_size} x {m.grad_accum_steps} accumulation = {m.batch_size * m.grad_accum_steps} examples)",
        f"  {m.seconds_per_val_loss_example:.3f} s per example for the Val loss (probe: {m.val_loss_probe} examples)",
        f"  {m.seconds_per_decode_clip:.2f} s per clip for the WER scoring (probe: {m.decode_probe} clips)",
        "",
        f"Planned time of `python -m stt run` with the current settings (n_trials {s.n_trials}, finalists {s.finalists_k}, max_steps {s.max_steps}, Val loss on {s.screening_val} example(s), Val {s.val_files} clips, Test {s.test_files} clips):",
    ]
    lines += [f"  {format_duration(seconds):>12}  {label}" for label, seconds in p.parts]
    lines += [f"  {format_duration(p.total_seconds):>12}  TOTAL (the Test part counted)", "", "Notes:"]
    lines += [f"  - {n}" for n in p.notes]
    return "\n".join(lines) + "\n"
