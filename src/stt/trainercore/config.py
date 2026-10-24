"""The `train` configuration section, owned by the TrainerCore component.

How to run: this module is not a script. The composition root registers the section:
    from stt.config import SectionRegistry, load_config
    from stt.trainercore.config import TrainConfig
    config = load_config(path, registry=SectionRegistry({"train": TrainConfig}))
    train = config.section("train", TrainConfig)
Example YAML with every key and its DEFAULT: configs/train_default.yaml. The search space of the hyperparameters
belongs to the HyperparamSearcher, not here. One "step" is one optimizer update: it consumes
`batch_size * grad_accum_steps` examples.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field

from stt.config import SectionModel
from stt.trainercore.types import TrainHyperparams


class HyperparamsConfig(SectionModel):
    """`train.hyperparams`: the values handed to one run. Placeholders until the owner decides them (O39)."""

    learning_rate: float = Field(default=1e-5, gt=0, description="DEFAULT 1e-5 (brief). Peak learning rate.")
    max_steps: int = Field(default=100, ge=1, description="DEFAULT 100 (placeholder, small for smoke runs; O39). Optimizer steps to run.")
    batch_size: int = Field(default=4, ge=1, description="DEFAULT 4 (brief). Examples per forward pass.")
    grad_accum_steps: int = Field(default=4, ge=1, description="DEFAULT 4 (placeholder; O39). Passes added up per optimizer step.")
    warmup_steps: int = Field(default=10, ge=0, description="DEFAULT 10 (placeholder; O39). Linear warmup, then linear decay to 0.")
    weight_decay: float = Field(default=0.0, ge=0, description="DEFAULT 0.0. AdamW weight decay.")
    max_grad_norm: float = Field(default=1.0, gt=0, description="DEFAULT 1.0 (brief). Gradient clipping.")
    seed: int | None = Field(default=None, description="DEFAULT none = use runtime.seed. Applied to Python, numpy, torch and the data order.")


class EarlyStoppingConfig(SectionModel):
    """`train.early_stopping`: stop when the Val loss has not improved for `patience` evaluations."""

    patience: int = Field(default=5, ge=1, description="DEFAULT 5 evaluations without improvement.")
    min_delta: float = Field(default=0.0, ge=0, description="DEFAULT 0.0. An evaluation improves only if it is lower by more than this.")


class TrainConfig(SectionModel):
    """The `train` section."""

    freeze_policy: str = Field(default="none", description="DEFAULT none (full fine-tuning). Registered: none, freeze_encoder.")
    hyperparams: HyperparamsConfig = HyperparamsConfig()
    log_every_steps: int = Field(default=10, ge=1, description="DEFAULT 10. A metrics record every N steps (and at step 1).")
    eval_every_steps: int = Field(default=50, ge=1, description="DEFAULT 50. Val loss every N steps (and once at the end).")
    save_every_steps: int = Field(default=100, ge=1, description="DEFAULT 100. A checkpoint every N steps (more are saved at a new best, a stop and the end).")
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    max_runtime_minutes: float | None = Field(default=None, gt=0, description="DEFAULT none = no cap. Per call: the run saves and stops gracefully.")
    cpu_threads: int | None = Field(default=None, ge=1, description="DEFAULT none = half of the cores (at least 1).")
    resume: bool = Field(default=False, description="DEFAULT false. true continues from the latest checkpoint of the output folder.")
    save_optimizer_state: bool = Field(
        default=True,
        description=(
            "DEFAULT true (the behaviour before this key existed). true: a checkpoint holds the model weights AND the optimizer, scheduler and random state, so the run can be resumed. "
            "false: a checkpoint holds only the model weights (about one third of the disk space, the Hugging Face option `save_only_model`), and the run CANNOT be resumed: "
            "`train.resume` true is refused with a clear message, in this mode and for any checkpoint made in it."
        ),
    )
    rename_retries: int = Field(
        default=5,
        ge=0,
        description=(
            "DEFAULT 5. When the finished model folder cannot be renamed into place (on Windows another program, for example the antivirus, the search indexer or a sync tool, can hold the new folder "
            "for a moment: `PermissionError`, O122), the rename is tried again this many times after the first failure (0 = no retry). If every attempt fails, the `.tmp` folder with the finished "
            "model and the finished checkpoint are kept and the run fails with a message that says how to continue."
        ),
    )
    rename_retry_seconds: float = Field(default=2.0, ge=0, description="DEFAULT 2.0. Seconds to wait before each new rename attempt (see `train.rename_retries`).")
    output_dir: Path | None = Field(default=None, description="Per experiment, set by the RunManager. The caller copies it into TrainRequest.output_dir.")
    sinks: list[str] = Field(default_factory=lambda: ["csv", "console"], min_length=1, description="DEFAULT csv, console. Registered: csv, console.")


def resolve_seed(hyperparams: HyperparamsConfig, runtime_seed: int) -> int:
    """The seed of the run: `train.hyperparams.seed` when set, else `runtime.seed`."""
    return hyperparams.seed if hyperparams.seed is not None else runtime_seed


def hyperparams_from_config(train: TrainConfig, runtime_seed: int) -> TrainHyperparams:
    """The typed `TrainHyperparams` of the config (the manual path; the HyperparamSearcher builds its own)."""
    h = train.hyperparams
    return TrainHyperparams(
        learning_rate=h.learning_rate,
        max_steps=h.max_steps,
        batch_size=h.batch_size,
        grad_accum_steps=h.grad_accum_steps,
        warmup_steps=h.warmup_steps,
        weight_decay=h.weight_decay,
        max_grad_norm=h.max_grad_norm,
        seed=resolve_seed(h, runtime_seed),
    )


def resolve_threads(train: TrainConfig) -> int:
    """The number of CPU threads: `train.cpu_threads` when set, else half of the cores (at least 1)."""
    return train.cpu_threads if train.cpu_threads is not None else max(1, (os.cpu_count() or 2) // 2)
