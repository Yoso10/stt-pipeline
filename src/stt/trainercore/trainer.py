"""TrainerCore: runs ONE training run on a given model and datasets with given hyperparameters (C6).

How to run: from Python (PowerShell, project root; the full working example is in docs/COMPONENTS.md, C6):
    $env:PYTHONPATH = "src"
    python -c "from stt.trainercore import TrainConfig, TrainerCore; print(TrainerCore(TrainConfig()))"
`TrainerCore(config).train(request) -> TrainResult`. Steps of a call: check the request, apply the folder rules (never overwrite; resume
only with a matching checkpoint), apply the freeze policy, create the sinks, run the Hugging Face adapter, build the returned model
folder from the best checkpoint. Importing this module does not import torch or transformers.
"""

from __future__ import annotations

import logging
import math
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from stt.trainercore.config import TrainConfig, resolve_threads
from stt.trainercore.errors import TrainerConfigError, TrainerCoreError
from stt.trainercore.final_model import INFO_NAME, build_model_dir
from stt.trainercore.registry import FreezeRegistry, SinkRegistry, default_freeze_registry, default_sink_registry
from stt.trainercore.run_folder import MODEL, checkpoints_dir, fingerprint, list_checkpoints, prepare, read_state
from stt.trainercore.types import (
    FreezeReport,
    RunState,
    SinkContext,
    StoppedReason,
    TrainHyperparams,
    TrainingJob,
    TrainRequest,
    TrainResult,
    TrainTimings,
)

logger = logging.getLogger("stt.trainercore")


def check_hyperparams(h: TrainHyperparams) -> None:
    """Fail with the key and the fix for an impossible value. Errors: TrainerConfigError (all problems in one message)."""
    problems = []
    if not (math.isfinite(h.learning_rate) and h.learning_rate > 0):
        problems.append(f"learning_rate must be a positive number, got {h.learning_rate}")
    if h.max_steps < 1:
        problems.append(f"max_steps must be at least 1, got {h.max_steps}")
    if h.batch_size < 1:
        problems.append(f"batch_size must be at least 1, got {h.batch_size}")
    if h.grad_accum_steps < 1:
        problems.append(f"grad_accum_steps must be at least 1, got {h.grad_accum_steps}")
    if h.warmup_steps < 0 or h.warmup_steps > h.max_steps:
        problems.append(f"warmup_steps must be between 0 and max_steps ({h.max_steps}), got {h.warmup_steps}")
    if h.weight_decay < 0:
        problems.append(f"weight_decay must not be negative, got {h.weight_decay}")
    if not (math.isfinite(h.max_grad_norm) and h.max_grad_norm > 0):
        problems.append(f"max_grad_norm must be a positive number, got {h.max_grad_norm}")
    if problems:
        raise TrainerConfigError("Invalid training hyperparameters: " + "; ".join(problems) + ". Fix train.hyperparams (or the searcher's values).")


class TrainerCore:
    """Runs one training run. Hugging Face and torch are used only by `hf_trainer` (imported when a run starts).

    The registries select the freeze policy (`train.freeze_policy`) and the sinks (`train.sinks`) by name. `clock` is injectable for tests.
    """

    def __init__(
        self,
        config: TrainConfig,
        freeze_registry: FreezeRegistry | None = None,
        sink_registry: SinkRegistry | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._freeze = freeze_registry or default_freeze_registry()
        self._sinks = sink_registry or default_sink_registry()
        self._clock = clock

    def train(self, request: TrainRequest) -> TrainResult:
        """Run the training and return the result.

        Inputs: a TrainRequest (model, Train and Val datasets, collator, hyperparameters, output folder, optional stop callback).
        Output: a TrainResult whose `model_dir` loads through the ModelLoader.
        Errors: TrainerConfigError (exit code 2: wrong settings, unknown policy or sink, output folder not empty, resume without a matching
        checkpoint) and TrainerCoreError (exit code 4: NaN loss, out of memory, damaged state).
        """
        config = self._config
        if len(request.train) == 0 or len(request.val) == 0:
            raise TrainerCoreError(
                f"The {'Train' if len(request.train) == 0 else 'Val'} dataset is empty, so nothing can be trained. "
                "Check the split and the SampleFilter report."
            )
        check_hyperparams(request.hyperparams)
        if request.model_name is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", request.model_name):
            raise TrainerConfigError(f"TrainRequest.model_name '{request.model_name}' may only contain letters, digits, '_', '.' and '-' (it becomes a folder name).")
        policy = self._freeze.create(config.freeze_policy)  # unknown names fail here, before any folder is made
        sink_names = list(config.sinks)
        self._sinks.require(sink_names)  # unknown names fail here too
        output_dir = request.output_dir
        resumed = prepare(output_dir, config.resume, fingerprint(request, config.freeze_policy))
        checkpoints = checkpoints_dir(output_dir)
        checkpoint, state = resumed if resumed is not None else (None, RunState())
        resume_step = int(checkpoint.name.rsplit("-", 1)[1]) if checkpoint is not None else None
        if checkpoint is not None:
            logger.info("Resuming from %s (step %d).", checkpoint, resume_step)
        freeze_report = policy.apply(request.model.model)
        logger.info(
            "Freeze policy %s: %d of %d parameters are trained.", freeze_report.policy, freeze_report.trainable_params, freeze_report.total_params
        )
        context = SinkContext(output_dir, resume_step)
        job = TrainingJob(
            request=request,
            checkpoints_dir=checkpoints,
            resume_checkpoint=checkpoint,
            state=state,
            sinks=tuple(self._sinks.create(name, context) for name in sink_names),
            threads=resolve_threads(config),
            log_every_steps=config.log_every_steps,
            eval_every_steps=config.eval_every_steps,
            save_every_steps=config.save_every_steps,
            patience=config.early_stopping.patience,
            min_delta=config.early_stopping.min_delta,
            max_runtime_seconds=None if config.max_runtime_minutes is None else config.max_runtime_minutes * 60.0,
            clock=self._clock,
        )
        from stt.trainercore.hf_trainer import run_training  # imported only when a run starts

        outcome = run_training(job)
        return self._finish(request, outcome.state, outcome.stopped_reason, outcome.timings, freeze_report)

    def _finish(
        self, request: TrainRequest, state: RunState, reason: StoppedReason, timings: TrainTimings, freeze_report: FreezeReport
    ) -> TrainResult:
        """Build the returned model folder from the best checkpoint (the latest one when no evaluation happened)."""
        checkpoints = list_checkpoints(checkpoints_dir(request.output_dir))
        if not checkpoints:
            raise TrainerCoreError(f"The run ended without a checkpoint in {checkpoints_dir(request.output_dir)}, so there is no model to return.")
        by_step = dict(checkpoints)
        chosen_step = state.best_step if state.best_step in by_step else checkpoints[-1][0]
        h = request.hyperparams
        info = {
            "source_config_ref": request.source_config_ref,
            "base_model": request.model.info.name,
            "stopped_reason": reason,
            "best_step": chosen_step,
            "best_val_loss": state.best_val_loss if chosen_step == state.best_step else None,
            "seed": h.seed,
            "freeze_policy": freeze_report.policy,
            "steps_completed": state.history[-1].step if state.history else chosen_step,
            "hyperparams": {
                "learning_rate": h.learning_rate,
                "max_steps": h.max_steps,
                "batch_size": h.batch_size,
                "grad_accum_steps": h.grad_accum_steps,
                "warmup_steps": h.warmup_steps,
                "weight_decay": h.weight_decay,
                "max_grad_norm": h.max_grad_norm,
            },
        }
        name = MODEL if request.model_name is None else f"{request.model_name}_step_{chosen_step:05d}"
        if request.model_name is not None:
            for old in request.output_dir.glob(f"{request.model_name}_step_*"):  # the model of an earlier part of a resumed run is replaced
                if old.name != name and (old / INFO_NAME).is_file():
                    shutil.rmtree(old, ignore_errors=True)
        model_dir = build_model_dir(by_step[chosen_step], request.model.info.path, request.output_dir / name, info)
        logger.info("Run ended (%s). Model of step %d saved in %s (%s).", reason, chosen_step, model_dir, INFO_NAME)
        return TrainResult(
            model_dir=model_dir,
            history=tuple(state.history),
            best_step=chosen_step,
            best_val_loss=info["best_val_loss"],
            stopped_reason=reason,
            timings=timings,
            freeze_report=freeze_report,
            seed=h.seed,
        )
