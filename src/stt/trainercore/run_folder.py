"""The output folder of a run: the never-overwrite rule, checkpoints, saved bookkeeping, the resume check (standard library only).

How to run: this module is not a script; TrainerCore calls it. Layout of a run folder:
    <output_dir>/run_info.json          what the run was started with (read again to check a resume)
    <output_dir>/metrics.csv            the live metrics file (see sinks.py)
    <output_dir>/checkpoints/checkpoint-<step>/   only the latest and the best checkpoint are kept
    <output_dir>/model/                 the returned model (safetensors + tokenizer files + pipeline_model_info.json)
Example (PowerShell, project root; lists the checkpoints of a run folder):
    $env:PYTHONPATH = "src"
    python -c "from pathlib import Path; from stt.trainercore.run_folder import list_checkpoints; print(list_checkpoints(Path('data/run_x/checkpoints')))"
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from stt.trainercore.errors import TrainerConfigError, TrainerCoreError
from stt.trainercore.types import RunState, StepMetrics, TrainRequest

CHECKPOINTS = "checkpoints"
MODEL = "model"
RUN_INFO = "run_info.json"
STATE_FILE = "trainercore_state.json"
OPTIMIZER_FILE = "optimizer.pt"  # written by the Trainer into a checkpoint only when `train.save_optimizer_state` is true; a checkpoint without it cannot be resumed
PREFIX = "checkpoint-"


def checkpoints_dir(output_dir: Path) -> Path:
    """The folder of the checkpoints of a run."""
    return output_dir / CHECKPOINTS


def list_checkpoints(directory: Path) -> list[tuple[int, Path]]:
    """The COMPLETE checkpoints (those with a saved state file) as (step, folder), lowest step first."""
    found: list[tuple[int, Path]] = []
    if directory.is_dir():
        for child in directory.iterdir():
            suffix = child.name[len(PREFIX):]
            if child.is_dir() and child.name.startswith(PREFIX) and suffix.isdigit() and (child / STATE_FILE).is_file():
                found.append((int(suffix), child))
    return sorted(found)


def write_state(checkpoint: Path, state: RunState) -> None:
    """Save the bookkeeping inside a checkpoint folder (written after the Trainer finished the checkpoint: the completeness marker)."""
    (checkpoint / STATE_FILE).write_text(json.dumps(asdict(state), indent=1), encoding="utf-8")


def read_state(checkpoint: Path) -> RunState:
    """Load the bookkeeping of a checkpoint. Errors: TrainerCoreError when the file is damaged."""
    try:
        raw = json.loads((checkpoint / STATE_FILE).read_text(encoding="utf-8"))
        history = [StepMetrics(**row) for row in raw.pop("history")]
        return RunState(history=history, **raw)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise TrainerCoreError(
            f"The saved state {checkpoint / STATE_FILE} is damaged ({type(exc).__name__}: {exc}). "
            "Delete that checkpoint folder to resume from an earlier one, or start a new run folder."
        ) from exc


def prune(directory: Path, keep: set[int]) -> None:
    """Delete every checkpoint folder whose step is not in `keep` (the latest and the best are kept by the caller)."""
    for step, folder in list_checkpoints(directory):
        if step not in keep:
            shutil.rmtree(folder, ignore_errors=True)


def fingerprint(request: TrainRequest, freeze_policy: str) -> dict[str, Any]:
    """What identifies a run for the resume check: the model, the freeze policy, every hyperparameter and the data sizes."""
    return {
        "model": request.model.info.name,
        "freeze_policy": freeze_policy,
        "hyperparams": asdict(request.hyperparams),
        "train_examples": len(request.train),
        "val_examples": len(request.val),
    }


def prepare(output_dir: Path, resume: bool, current: dict[str, Any]) -> tuple[Path, RunState] | None:
    """Apply the folder rules and return what to resume from.

    resume false: the folder must not exist or must be empty (NEVER overwritten); returns None and writes `run_info.json`.
    resume true: needs a complete checkpoint and a `run_info.json` equal to `current`; returns (latest checkpoint, its state).
    Errors: TrainerConfigError (both cases name the folder and the fix).
    """
    info = output_dir / RUN_INFO
    if not resume:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise TrainerConfigError(
                f"The output folder {output_dir} is not empty and train.resume is false: it is never overwritten. "
                "Use a new experiment folder, or set train.resume to true to continue the run that is in it."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        info.write_text(json.dumps({"fingerprint": current}, indent=1), encoding="utf-8")
        return None
    checkpoints = list_checkpoints(checkpoints_dir(output_dir))
    if not checkpoints:
        raise TrainerConfigError(
            f"train.resume is true but {checkpoints_dir(output_dir)} holds no complete checkpoint. "
            "Set train.resume to false and use a new output folder, or point output_dir at the run to continue."
        )
    try:
        saved = json.loads(info.read_text(encoding="utf-8"))["fingerprint"]
    except (OSError, ValueError, KeyError) as exc:
        raise TrainerConfigError(
            f"Cannot resume: {info} is missing or damaged ({type(exc).__name__}). "
            "It tells which settings the run was started with. Use a new output folder."
        ) from exc
    if saved != current:
        differences = [f"{key}: checkpoint {saved.get(key)!r}, now {current.get(key)!r}" for key in sorted(set(saved) | set(current)) if saved.get(key) != current.get(key)]
        raise TrainerConfigError(
            f"Cannot resume: the checkpoint in {output_dir} belongs to a different configuration ({'; '.join(differences)}). "
            "Restore the original settings to continue it, or start a new output folder."
        )
    step, folder = checkpoints[-1]
    return folder, read_state(folder)
