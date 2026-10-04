"""FeatureCollator: turns a list of FeatureExample into one padded batch for the Hugging Face trainer.

How to run: this module is not a script; `FeaturePipeline.build_collator()` creates it. Example (PowerShell, project root;
`examples` is a list of FeatureExample from a FeatureResult):
    $env:PYTHONPATH = "src"
    python -c "from stt.featurepipeline.collator import FeatureCollator; c = FeatureCollator(decoder_start_token_id=50258)"
Output of `collator(examples)`: a dict `{"input_features": float32 tensor (batch, mel bins, frames), "labels": int64 tensor
(batch, longest label)}`. The labels are padded with -100 (the value the loss ignores) and the leading decoder start
token is removed, because the model puts it back when it shifts the labels to make the decoder input. The order of the
batch is the order of the examples. This is the ONLY module of the component that imports torch.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from stt.featurepipeline.types import FeatureExample

IGNORE_INDEX = -100


class FeatureCollator:
    """Batch maker for the Hugging Face `Seq2SeqTrainer` (padding value -100, start token removed)."""

    def __init__(self, decoder_start_token_id: int | None) -> None:
        self._start = decoder_start_token_id

    def __call__(self, examples: Sequence[FeatureExample]) -> dict[str, torch.Tensor]:
        """Return the padded batch. Errors: ValueError for an empty list or features of different shapes."""
        if not examples:
            raise ValueError("cannot make a batch from zero examples")
        shapes = {e.input_features.shape for e in examples}
        if len(shapes) != 1:
            raise ValueError(f"the features of one batch must have the same shape, got {sorted(shapes)}")
        features = torch.from_numpy(np.stack([e.input_features for e in examples]).astype(np.float32, copy=False))
        sequences = [e.labels[1:] if self._start is not None and e.labels and e.labels[0] == self._start else list(e.labels) for e in examples]
        width = max(len(s) for s in sequences)
        labels = torch.full((len(sequences), width), IGNORE_INDEX, dtype=torch.long)
        for row, ids in enumerate(sequences):
            labels[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return {"input_features": features, "labels": labels}
