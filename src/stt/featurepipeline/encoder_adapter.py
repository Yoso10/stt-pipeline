"""The Hugging Face side of the encoder-output cache: the only module besides the collator that imports torch (DECISIONS 89).

How to run: this module is not a script; the FeaturePipeline imports it only when `features.encoder_cache.enabled` is true. Example (PowerShell, project root; `loaded` is a LoadedModel):
    $env:PYTHONPATH = "src"
    python -c "from stt.featurepipeline.encoder_adapter import encoder_weights_fingerprint"
What it does: hashes the weights of the base encoder, runs the encoder once on the features of a clip (eval mode, no gradients, then the model's training mode is restored),
and compares the Val loss with and without the cached output for the one-time self-check. The model is an opaque object (a Hugging Face Whisper model).
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch


def _encoder(model: Any) -> Any:
    return model.model.encoder


def encoder_weights_fingerprint(model: Any) -> str:
    """SHA-256 over the names and values of every parameter and buffer of the encoder, in sorted name order (about 2.5 GB for the turbo model: a few seconds)."""
    digest = hashlib.sha256()
    state = _encoder(model).state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        if tensor.dtype in (torch.bfloat16,):
            tensor = tensor.to(torch.float32)
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def encoder_output_shape(model: Any) -> tuple[int, int]:
    """(frames, width) of one clip's encoder output: `max_source_positions` and `d_model` of the model."""
    config = model.config
    return int(config.max_source_positions), int(config.d_model)


def encode(model: Any, features: np.ndarray) -> np.ndarray:
    """The encoder output of one clip, float32 (frames, width). Eval mode and no gradients; the model's own mode is restored."""
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            batch = torch.from_numpy(np.asarray(features, dtype=np.float32))[None].to(model.device)
            return np.asarray(_encoder(model)(batch).last_hidden_state[0].detach().cpu().numpy(), dtype=np.float32)
    finally:
        model.train(was_training)


def val_losses(model: Any, batch: dict[str, Any], cached: np.ndarray) -> tuple[float, float]:
    """(Val loss computed through the encoder, Val loss computed from `cached`) for one collated batch of one clip."""
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            labels = batch["labels"].to(model.device)
            plain = model(input_features=batch["input_features"].to(model.device), labels=labels).loss
            cached_tensor = torch.from_numpy(np.asarray(cached, dtype=np.float32))[None].to(model.device)
            through_cache = model(encoder_outputs=(cached_tensor,), labels=labels).loss
            return float(plain), float(through_cache)
    finally:
        model.train(was_training)
