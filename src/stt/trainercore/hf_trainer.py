"""Hugging Face adapter of the TrainerCore: runs ONE training with `Seq2SeqTrainer` (the ONLY module of the component that imports torch and transformers).

How to run: this module is not a script; `TrainerCore.train` imports it when a run starts (the full working example is in
docs/COMPONENTS.md, C6). It does the following and nothing else:
  - applies the CPU thread limit and the seed (Python, numpy, torch, data order) and restores the thread count afterwards;
  - trains with a loss-only evaluation on Val (`predict_with_generate` is off: nothing is generated in the loop);
  - turns the Trainer's events into `StepMetrics` records for the sinks (live monitoring), the best-loss bookkeeping and early stopping
    (`EvalMonitor`), the optional `on_eval` stop callback, the time cap, and the NaN guard;
  - saves a checkpoint at a new best Val loss, at every stop and at the end, and keeps only the latest and the best checkpoint.
The Val loss is computed on the whole Val set at every evaluation; evaluations are NOT part of `seconds_per_step`.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, TrainerControl, TrainerState, set_seed
from transformers.trainer_callback import PrinterCallback, ProgressCallback

from stt.trainercore.errors import TrainerCoreError
from stt.trainercore.monitor import EvalMonitor
from stt.trainercore.run_folder import PREFIX, list_checkpoints, prune, write_state
from stt.trainercore.types import StepMetrics, StoppedReason, TrainingJob, TrainingOutcome, TrainTimings

logger = logging.getLogger("stt.trainercore")
_OOM_SIGNS = ("out of memory", "defaultcpuallocator", "can't allocate memory", "not enough memory")


def _finite(value: float | None) -> bool:
    return value is None or math.isfinite(value)


class _ControlCallback(TrainerCallback):  # type: ignore[misc]
    """Turns Trainer events into records, stops and checkpoints. All the decisions live in EvalMonitor and the job."""

    def __init__(self, job: TrainingJob) -> None:
        self._job = job
        self._state = job.state
        self._monitor = EvalMonitor(job.patience, job.min_delta, job.state)
        self._clock = job.clock
        self._base_elapsed = job.state.elapsed_sec
        self._t0 = 0.0
        self._marker = 0.0  # end of the last overhead (log, evaluation, save): the next step is timed from here
        self._after_step = 0.0  # end of the last step: evaluation time is measured from here
        self._window: list[float] = []
        self._steps_run = 0
        self._step_seconds = 0.0
        self._eval_runs = 0
        self._eval_seconds = 0.0
        self._save_seconds = 0.0
        self.reason: StoppedReason | None = None
        self.nan_step: int | None = None
        self.timings = TrainTimings(0.0, 0, 0.0, 0, 0.0, 0.0)

    # ---- helpers -------------------------------------------------------------------------------------------------
    def _elapsed(self) -> float:
        return self._base_elapsed + (self._clock() - self._t0)

    def _last(self) -> StepMetrics | None:
        return self._state.history[-1] if self._state.history else None

    def _emit(self, record: StepMetrics) -> None:
        self._state.history.append(record)
        for sink in self._job.sinks:
            try:
                sink.write(record)
            except Exception:  # a monitoring sink must never stop the training
                logger.exception("A metrics sink failed on step %d; training continues.", record.step)

    def _stop(self, reason: StoppedReason, control: TrainerControl, save: bool) -> None:
        if self.reason is None:
            self.reason = reason
        control.should_training_stop = True
        if save:
            control.should_save = True

    def _abort_nan(self, step: int, control: TrainerControl) -> None:
        if self.nan_step is None:
            self.nan_step = step
        control.should_training_stop = True
        control.should_save = False  # the last good checkpoint stays the last one
        control.should_evaluate = False

    def _weights_finite(self) -> bool:
        model: Any = self._job.request.model.model
        return all(bool(torch.isfinite(p).all()) for p in model.parameters() if p.requires_grad)

    # ---- Trainer events ------------------------------------------------------------------------------------------
    def on_train_begin(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        self._t0 = self._marker = self._after_step = self._clock()

    def on_step_end(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        now = self._clock()
        duration = now - self._marker
        self._window.append(duration)
        self._steps_run += 1
        self._step_seconds += duration
        self._after_step = now
        cap = self._job.max_runtime_seconds
        if cap is not None and now - self._t0 >= cap:
            logger.info("The time cap of %.0f s is reached at step %d: saving and stopping.", cap, state.global_step)
            self._stop("time_limit", control, save=True)
        stop_flag = self._job.request.stop_flag
        if stop_flag is not None and stop_flag():
            logger.info("A stop was requested at step %d: saving a checkpoint and stopping.", state.global_step)
            self._stop("stopped_by_callback", control, save=True)
        if control.should_save and not self._weights_finite():
            self._abort_nan(state.global_step, control)

    def on_log(self, args: Any, state: TrainerState, control: TrainerControl, logs: dict[str, float] | None = None, **kwargs: Any) -> None:
        if not logs or "loss" not in logs:  # an evaluation log or the final summary: records are made in on_evaluate
            return
        last = self._last()
        seconds = sum(self._window) / len(self._window) if self._window else (last.seconds_per_step if last else 0.0)
        eta = (self._job.request.hyperparams.max_steps - state.global_step) * seconds if self._window else None
        self._window = []
        record = StepMetrics(
            step=state.global_step,
            epoch_fraction=float(state.epoch or 0.0),
            train_loss=float(logs["loss"]),
            val_loss=None,
            learning_rate=_as_float(logs.get("learning_rate")),
            grad_norm=_as_float(logs.get("grad_norm")),
            seconds_per_step=seconds,
            elapsed_sec=self._elapsed(),
            eta_sec=eta,
        )
        self._emit(record)
        if not (_finite(record.train_loss) and _finite(record.grad_norm)):
            self._abort_nan(state.global_step, control)
        self._marker = self._clock()

    def on_evaluate(self, args: Any, state: TrainerState, control: TrainerControl, metrics: dict[str, float] | None = None, **kwargs: Any) -> None:
        now = self._clock()
        self._eval_runs += 1
        self._eval_seconds += now - self._after_step
        step = state.global_step
        val_loss = float((metrics or {}).get("eval_loss", float("nan")))
        last = self._last()
        record = StepMetrics(
            step=step,
            epoch_fraction=float(state.epoch or 0.0),
            train_loss=last.train_loss if last else None,
            val_loss=val_loss,
            learning_rate=last.learning_rate if last else None,
            grad_norm=last.grad_norm if last else None,
            seconds_per_step=last.seconds_per_step if last else 0.0,
            elapsed_sec=self._elapsed(),
            eta_sec=last.eta_sec if last else None,
        )
        self._emit(record)
        if not math.isfinite(val_loss):
            self._abort_nan(step, control)
        else:
            verdict = self._monitor.record(step, val_loss)
            if verdict.improved:
                control.should_save = True  # the best model must exist as a checkpoint
            stop_requested = bool(self._job.request.on_eval(record)) if self._job.request.on_eval else False
            if verdict.stop:
                self._stop("early_stopping", control, save=True)
                if verdict.never_improved_after_first:
                    logger.warning(
                        "Early stopping at step %d: the Val loss did not improve after the first evaluation (step %d, %.4f). "
                        "Check the learning rate and the data; see TROUBLESHOOTING (TrainerCore).",
                        step,
                        self._state.first_eval_step,
                        self._state.best_val_loss or float("nan"),
                    )
            elif stop_requested:
                self._stop("stopped_by_callback", control, save=True)
        self._marker = self._clock()

    def on_save(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        self._save_seconds += self._clock() - self._marker
        step = state.global_step
        self._state.elapsed_sec = self._elapsed()
        write_state(Path(args.output_dir) / f"{PREFIX}{step}", self._state)
        keep = {step} | ({self._state.best_step} if self._state.best_step is not None else set())
        prune(self._job.checkpoints_dir, keep)
        self._marker = self._clock()

    def on_train_end(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        total = self._clock() - self._t0
        mean = self._step_seconds / self._steps_run if self._steps_run else 0.0
        self.timings = TrainTimings(total, self._steps_run, mean, self._eval_runs, self._eval_seconds, self._save_seconds)


def _as_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _is_out_of_memory(exc: BaseException) -> bool:
    return isinstance(exc, MemoryError) or any(sign in str(exc).lower() for sign in _OOM_SIGNS)


def run_training(job: TrainingJob) -> TrainingOutcome:
    """Run the training described by `job` and return the final bookkeeping.

    Errors: TrainerCoreError for NaN or infinite loss (naming the step and the last good checkpoint) and for running out of memory.
    """
    request = job.request
    h = request.hyperparams
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(job.threads)
    callback = _ControlCallback(job)
    try:
        set_seed(h.seed)
        args = Seq2SeqTrainingArguments(
            output_dir=str(job.checkpoints_dir),
            use_cpu=True,
            per_device_train_batch_size=h.batch_size,
            per_device_eval_batch_size=h.batch_size,
            gradient_accumulation_steps=h.grad_accum_steps,
            learning_rate=h.learning_rate,
            weight_decay=h.weight_decay,
            max_grad_norm=h.max_grad_norm,
            max_steps=h.max_steps,
            warmup_steps=h.warmup_steps,
            lr_scheduler_type="linear",
            optim="adamw_torch",
            logging_strategy="steps",
            logging_steps=job.log_every_steps,
            logging_first_step=True,
            eval_strategy="steps",
            eval_steps=job.eval_every_steps,
            save_strategy="steps",
            save_steps=job.save_every_steps,
            save_total_limit=None,
            predict_with_generate=False,
            remove_unused_columns=False,
            label_names=["labels"],
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            seed=h.seed,
            data_seed=h.seed,
            report_to="none",
            disable_tqdm=True,
        )
        trainer = Seq2SeqTrainer(
            model=request.model.model,
            args=args,
            data_collator=request.collator,
            train_dataset=request.train,
            eval_dataset=request.val,
            callbacks=[callback],
        )
        for default in (PrinterCallback, ProgressCallback):  # the console is written by the sinks, not by the Trainer
            trainer.remove_callback(default)
        resume = str(job.resume_checkpoint) if job.resume_checkpoint is not None else None
        try:
            trainer.train(resume_from_checkpoint=resume)
        except (MemoryError, RuntimeError) as exc:
            if not _is_out_of_memory(exc):
                raise
            raise TrainerCoreError(
                f"Out of memory during training ({exc}). Lower train.hyperparams.batch_size and raise grad_accum_steps, "
                "use train.freeze_policy freeze_encoder, or use a smaller model. There is no automatic retry."
            ) from exc
    finally:
        torch.set_num_threads(previous_threads)
    if callback.nan_step is not None:
        saved = list_checkpoints(job.checkpoints_dir)
        last_good = f"The last good checkpoint is {saved[-1][1]}." if saved else "No checkpoint was saved before it."
        raise TrainerCoreError(
            f"The loss became NaN or infinite at step {callback.nan_step}. The most likely cause is a learning rate that is too high "
            f"(now {h.learning_rate:g}): lower train.hyperparams.learning_rate and read the first lines of metrics.csv. {last_good}"
        )
    return TrainingOutcome(job.state, callback.reason or "max_steps", callback.timings)
