"""A tiny `train` + `evaluate` experiment on synthetic clips and the stored tiny test model, for the tests of the convergence report and of `final-test` (DECISIONS 90).

How to run: this module is not a test; the tests import it. Example (project root):
    python -c "import sys; sys.path[:0]=['src','tests']; import tiny_flow; print(tiny_flow.TINY.name)"
Everything happens in the temporary folder that the caller gives; nothing is written into the project. The run takes about 20 seconds (CPU, a few steps).
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stt.orchestrator import cli  # noqa: E402
from stt.orchestrator.smoke import SmokeConfig, write_smoke_data  # noqa: E402

TINY = ROOT / "tests" / "assets" / "tiny_whisper"
SMOKE_YAML = ROOT / "configs" / "smoke.yaml"


def run_cli(*argv: str) -> tuple[int, str, str]:
    """Run the command line in-process; returns (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def base_args(work: Path, cache: bool = False, dtype: str = "float32", steps: int = 8, eval_every: int = 2) -> list[str]:
    """The `--config` / `--set` words shared by the commands of one tiny experiment (the same words must be given to every command of the experiment)."""
    sets = [
        f"data.source.path={(work / 'smoke').as_posix()}",
        f"model.path={TINY.as_posix()}",
        f"run.experiments_dir={(work / 'e').as_posix()}",
        "train.freeze_policy=freeze_encoder",
        f"train.hyperparams.max_steps={steps}",
        "train.hyperparams.warmup_steps=0",
        f"train.eval_every_steps={eval_every}",
        "train.log_every_steps=1",
        f"train.save_every_steps={steps}",
        "train.cpu_threads=2",
        f"features.encoder_cache.enabled={'true' if cache else 'false'}",
    ]
    if cache:
        sets += [f"features.encoder_cache.dir={(work / 'ec').as_posix()}", f"features.encoder_cache.dtype={dtype}"]
    args = ["--config", str(SMOKE_YAML)]
    for item in sets:
        args += ["--set", item]
    return args


def make_experiment(work: Path, cache: bool = False, dtype: str = "float32", steps: int = 8, eval_every: int = 2) -> tuple[list[str], str]:
    """Write the synthetic data, run `train` and then `evaluate --experiment` in `work`. Returns (the base words, the experiment id). Raises AssertionError when a command fails."""
    write_smoke_data(SmokeConfig(n_samples=24), 42, work / "smoke")
    base = base_args(work, cache, dtype, steps, eval_every)
    code, out, err = run_cli("train", *base)
    assert code == 0, err[-2000:]
    experiment = sorted(p.name for p in (work / "e").iterdir() if p.is_dir() and p.name.startswith("exp_"))[-1]
    code, out, err = run_cli("evaluate", "--experiment", experiment, *base)
    assert code == 0, err[-2000:]
    return base, experiment
