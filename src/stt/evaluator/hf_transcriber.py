"""Hugging Face adapter of the Evaluator: transcribes a feature dataset with `model.generate` (the ONLY module of the component that imports torch).

How to run: this module is not a script; the composition root creates `HfTranscriber()` and passes it to the Evaluator (the full working
example is in docs/COMPONENTS.md, C7). It does the following and nothing else:
  - runs the model in evaluation mode without gradients, in batches of `settings.batch_size`, and restores the previous mode afterwards;
  - uses the language and task that the ModelLoader wrote into the model (no timestamps are generated);
  - applies the token cap of each file (`settings.token_caps`, else `settings.max_new_tokens`, clamped to what the model can hold) and flags
    an output that used its whole cap without ending as `hit_length_cap` (the text is cut at the cap);
  - applies an optional thread limit (`threads`, wired by the composition root, for example from `train.cpu_threads`) during the call only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from stt.evaluator.errors import EvaluatorError
from stt.evaluator.types import DecodeSettings, RawHypothesis
from stt.featurepipeline import FeatureDataset
from stt.modelloader import LoadedModel

_OOM_SIGNS = ("out of memory", "defaultcpuallocator", "can't allocate memory", "not enough memory")
_PROMPT_TOKENS = 5  # the decoder prefix (start, language, task, no timestamps) and one token of margin


class HfTranscriber:
    """Transcriber over a Hugging Face Whisper model. `threads` None leaves the torch thread count as it is."""

    def __init__(self, threads: int | None = None) -> None:
        self._threads = threads

    def transcribe(self, model: LoadedModel, dataset: FeatureDataset, settings: DecodeSettings) -> list[RawHypothesis]:
        """One hypothesis per example, in dataset order.

        Errors: EvaluatorError when the model runs out of memory (lower eval.batch_size).
        """
        native: Any = model.model
        room = int(native.config.max_target_positions) - _PROMPT_TOKENS
        eos = int(native.config.eos_token_id)
        previous_threads = torch.get_num_threads()
        was_training = bool(native.training)
        if self._threads is not None:
            torch.set_num_threads(self._threads)
        native.eval()
        results: list[RawHypothesis] = []
        try:
            with torch.no_grad():
                for start in range(0, len(dataset), settings.batch_size):
                    examples = [dataset[i] for i in range(start, min(start + settings.batch_size, len(dataset)))]
                    caps = [min(room, (settings.token_caps or {}).get(e.sample_id, settings.max_new_tokens)) for e in examples]
                    features = torch.from_numpy(np.stack([e.input_features for e in examples]).astype(np.float32, copy=False))
                    try:
                        out = native.generate(input_features=features, num_beams=settings.num_beams, max_new_tokens=max(caps), do_sample=False)
                    except (MemoryError, RuntimeError) as exc:
                        if not (isinstance(exc, MemoryError) or any(s in str(exc).lower() for s in _OOM_SIGNS)):
                            raise
                        raise EvaluatorError(f"Out of memory while transcribing ({exc}). Lower eval.batch_size; there is no automatic retry.") from exc
                    for example, cap, row in zip(examples, caps, out.tolist(), strict=True):
                        first_eos = row.index(eos) if eos in row else len(row)
                        hit = first_eos >= cap
                        text = model.tokenizer.decode(row[: min(first_eos, cap)], skip_special_tokens=True)
                        results.append(RawHypothesis(example.sample_id, text, hit))
        finally:
            native.train(was_training)
            torch.set_num_threads(previous_threads)
        return results
