"""Names of the model folders of an experiment (pure function).

How to run: this module is not a script; the Orchestrator uses it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.runmanager import model_folder_name; print(model_folder_name('exp_0004', 'trial', trial=3, step=500))"
Example output:
    exp_0004_trial_003_step_00500
A model folder is named after what it is, because the name of the folder is the name of the model in every result and report
(`ModelInfo.name`). Schemes: baseline `exp_0004_baseline`; a trial `exp_0004_trial_003_step_00500`; a finalist retrained on the full data
`exp_0004_final_trial_003_step_01200`. Numbers are zero-padded so that folders sort in order. The folder lives in `candidates/` of the
experiment (`ExperimentHandle.candidate_dir`). DECISIONS 43 has the options and the reason.
"""

from __future__ import annotations

from typing import Literal

from stt.runmanager.errors import RunManagerError

Kind = Literal["baseline", "trial", "final"]


def model_folder_name(experiment_id: str, kind: Kind, trial: int | None = None, step: int | None = None) -> str:
    """The folder name of a model. Errors: RunManagerError when the trial or the step is missing for a kind that needs it."""
    if kind == "baseline":
        return f"{experiment_id}_baseline"
    if kind not in ("trial", "final"):
        raise RunManagerError(f"Unknown model kind '{kind}'. Known kinds: baseline, trial, final.")
    if trial is None or step is None or trial < 0 or step < 0:
        raise RunManagerError(f"A '{kind}' model folder name needs a trial number and a step (both zero or more), got trial={trial} step={step}.")
    prefix = f"{experiment_id}_final_trial" if kind == "final" else f"{experiment_id}_trial"
    return f"{prefix}_{trial:03d}_step_{step:05d}"
