"""Freeze policies: which parameters of the model are trained (`train.freeze_policy`).

How to run: this module is not a script; the registry creates the policies. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.trainercore import default_freeze_registry; print(default_freeze_registry().names())"
Example output:
    ['freeze_encoder', 'none']
A policy only sets `requires_grad` on the parameters of the opaque model (it needs `named_parameters()`, nothing else), so this
module imports no torch. Both policies are in-place and idempotent. LoRA is NOT included (`peft` is not approved).
"""

from __future__ import annotations

from stt.trainercore.errors import TrainerConfigError
from stt.trainercore.types import FreezeReport

WHISPER_ENCODER_PREFIX = "model.encoder."


def _parameters(model: object) -> list[tuple[str, object]]:
    named = getattr(model, "named_parameters", None)
    if not callable(named):
        raise TrainerConfigError(
            f"The model object ({type(model).__name__}) has no named_parameters(), so no freeze policy can be applied. "
            "Use a model loaded by the ModelLoader."
        )
    return list(named())


def _count(parameters: list[tuple[str, object]]) -> tuple[int, int]:
    total = sum(int(p.numel()) for _, p in parameters)  # type: ignore[attr-defined]
    trainable = sum(int(p.numel()) for _, p in parameters if p.requires_grad)  # type: ignore[attr-defined]
    return total, trainable


class NoFreeze:
    """`none`: full fine-tuning. Every parameter is made trainable."""

    name = "none"

    def apply(self, model: object) -> FreezeReport:
        """Make all parameters trainable; nothing is frozen."""
        parameters = _parameters(model)
        for _, p in parameters:
            p.requires_grad_(True)  # type: ignore[attr-defined]
        total, trainable = _count(parameters)
        return FreezeReport(self.name, total, trainable, total - trainable, ())


class FreezeEncoder:
    """`freeze_encoder`: the audio encoder is frozen (every parameter named `model.encoder.*`); the decoder is trained."""

    name = "freeze_encoder"

    def apply(self, model: object) -> FreezeReport:
        """Freeze exactly the encoder parameters and make all the others trainable.

        Errors: TrainerConfigError when the model has no encoder parameters (not a Whisper model).
        """
        parameters = _parameters(model)
        frozen = 0
        for name, p in parameters:
            is_encoder = name.startswith(WHISPER_ENCODER_PREFIX)
            p.requires_grad_(not is_encoder)  # type: ignore[attr-defined]
            frozen += int(is_encoder)
        if frozen == 0:
            raise TrainerConfigError(
                f"train.freeze_policy 'freeze_encoder' found no parameter named '{WHISPER_ENCODER_PREFIX}*' in the model. "
                "It supports Whisper models only: set train.freeze_policy to 'none'."
            )
        total, trainable = _count(parameters)
        return FreezeReport(self.name, total, trainable, total - trainable, (WHISPER_ENCODER_PREFIX.rstrip("."),))
