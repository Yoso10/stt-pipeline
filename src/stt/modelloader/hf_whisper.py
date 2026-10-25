"""HfWhisperLoader: loads a Hugging Face Whisper folder (the ONLY module of the component that imports torch and transformers).

How to run: this module is not a script; the registry creates the loader on demand. Example
(PowerShell, project root; the full working example is in docs/COMPONENTS.md, C5):
    $env:PYTHONPATH = "src"
    python -c "from pathlib import Path; from stt.modelloader import ModelConfig, create_loader; l = create_loader(ModelConfig(path=Path('models/whisper-tiny'))); print(l.load().info)"
Rules: local files only, even when runtime.online is true; `trust_remote_code` is never enabled; the folder is
validated first and never modified; language and task come from Config and are written into the generation
settings and the tokenizer prefix; the number of mel bins is read from the feature extractor; no caching, so two
`load()` calls return independent objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration

from stt.modelloader.config import ModelConfig
from stt.modelloader.errors import ModelLoaderError
from stt.modelloader.types import LoadedModel, ModelInfo
from stt.modelloader.whisper_folder import check_model_folder, info_file_ref

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


class HfFeatureExtractor:
    """FeatureExtractor Protocol over a Hugging Face WhisperFeatureExtractor."""

    def __init__(self, native: WhisperFeatureExtractor) -> None:
        self._native = native
        self.sampling_rate: int = int(native.sampling_rate)
        self.window_seconds: float = float(native.chunk_length)
        self.num_mel_bins: int = int(native.feature_size)

    def extract(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Return float32 (num_mel_bins, frames) features; see the Protocol for the errors."""
        if sampling_rate != self.sampling_rate:
            raise ValueError(f"audio is at {sampling_rate} Hz but the model needs {self.sampling_rate} Hz: convert it first")
        if len(audio) > self.window_seconds * self.sampling_rate + 1e-6:
            raise ValueError(f"audio of {len(audio) / self.sampling_rate:.2f} s is longer than the model window of {self.window_seconds:g} s")
        features = self._native(audio, sampling_rate=sampling_rate, return_tensors="np").input_features[0]
        return np.asarray(features, dtype=np.float32)


class HfTokenizer:
    """LabelTokenizer Protocol over a Hugging Face Whisper tokenizer with the language and task prefix set."""

    def __init__(self, native: Any, max_label_tokens: int) -> None:
        self._native = native
        self.max_label_tokens = max_label_tokens

    def encode_label(self, text: str) -> list[int]:
        """Prefix tokens + text tokens + end token."""
        return [int(i) for i in self._native(text).input_ids]

    def count_tokens(self, text: str) -> int:
        """Number of text tokens, without special tokens."""
        return len(self._native(text, add_special_tokens=False).input_ids)

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Text of `ids`."""
        return str(self._native.decode(list(ids), skip_special_tokens=skip_special_tokens))


class HfWhisperLoader:
    """Loads a model, its feature extractor and its tokenizer from the local folder `config.path`.

    Errors: ModelLoaderError for everything the brief lists (missing folder or files, old weights, not Whisper,
    English-only model or unsupported language, unknown dtype is rejected by the Config schema, out of memory).
    """

    def __init__(self, config: ModelConfig, device: str = "cpu") -> None:
        self._config = config
        self._device = device  # "cpu" (the model stays where it is loaded) or "cuda" (the model is moved to the GPU after loading; resolved from runtime.device)

    def load(self) -> LoadedModel:
        """Validate the folder, then load. Every call returns new, independent objects."""
        config = self._config
        folder = config.path
        check_model_folder(folder, config.language)
        try:
            model = WhisperForConditionalGeneration.from_pretrained(
                str(folder), local_files_only=True, trust_remote_code=False, dtype=_DTYPES[config.dtype]
            )
            extractor = WhisperFeatureExtractor.from_pretrained(str(folder), local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(
                str(folder), local_files_only=True, trust_remote_code=False, clean_up_tokenization_spaces=False
            )
        except MemoryError as exc:
            raise ModelLoaderError(
                f"Not enough memory to load {folder}. Use a smaller model or close other programs; no automatic retry."
            ) from exc
        except Exception as exc:  # damaged or incompatible files raise many different types (OSError, ValueError, SafetensorError, ...)
            raise ModelLoaderError(f"Cannot load the model in {folder}: {type(exc).__name__}: {exc}. Check that the folder is complete and not damaged.") from exc
        language_token = tokenizer.convert_tokens_to_ids(f"<|{config.language}|>")
        if language_token is None or language_token == tokenizer.unk_token_id:
            raise ModelLoaderError(
                f"The tokenizer of {folder.name} does not know the language '{config.language}' (token <|{config.language}|>). "
                "Check model.language (use a language code such as he) or use a multilingual model."
            )
        tokenizer.set_prefix_tokens(language=config.language, task=config.task, predict_timestamps=False)
        generation = model.generation_config
        generation.language = config.language
        generation.task = config.task
        generation.forced_decoder_ids = None  # the language and task are given by the two settings above
        model.eval()
        if self._device != "cpu":
            try:
                model.to(self._device)
            except Exception as exc:  # noqa: BLE001  a CUDA out-of-memory error and a missing device are different exception types
                raise ModelLoaderError(f"Cannot move the model in {folder} to the device '{self._device}': {type(exc).__name__}: {exc}. A GPU with less memory than the model, or no usable GPU, is the usual cause; see runtime.device.") from exc
        wrapped_extractor = HfFeatureExtractor(extractor)
        info = ModelInfo(
            name=folder.name,
            path=folder,
            num_parameters=int(sum(p.numel() for p in model.parameters())),
            dtype=config.dtype,
            num_mel_bins=wrapped_extractor.num_mel_bins,
            language=config.language,
            task=config.task,
            source_config_ref=info_file_ref(Path(folder)),
        )
        return LoadedModel(
            model=model,
            feature_extractor=wrapped_extractor,
            tokenizer=HfTokenizer(tokenizer, int(model.config.max_target_positions)),
            info=info,
        )
