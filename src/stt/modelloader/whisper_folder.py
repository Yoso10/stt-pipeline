"""Checks a saved Whisper model folder BEFORE anything is loaded (standard library only).

How to run: this module is not a script; the loader calls it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from pathlib import Path; from stt.modelloader.whisper_folder import check_model_folder; check_model_folder(Path('models/whisper-tiny'))"
It prints nothing when the folder is valid and raises ModelLoaderError listing exactly what is wrong otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

from stt.modelloader.errors import ModelLoaderError

ENGLISH_ONLY_VOCAB_SIZE = 51864  # the Whisper ".en" models have this vocabulary; multilingual ones have 51865
OLD_WEIGHT_FILES = ("pytorch_model.bin", "pytorch_model.bin.index.json", "tf_model.h5", "flax_model.msgpack")


def info_file_ref(folder: Path) -> str | None:
    """`source_config_ref` from `pipeline_model_info.json` when the file exists, else None."""
    path = folder / "pipeline_model_info.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ModelLoaderError(f"{path} is not valid JSON ({exc}). The folder is damaged: copy it again.") from exc
    ref = data.get("source_config_ref") if isinstance(data, dict) else None
    return None if ref is None else str(ref)


def check_model_folder(folder: Path, language: str | None = None) -> None:
    """Fail with a clear message unless `folder` is a complete multilingual Whisper folder in safetensors format.

    Errors: ModelLoaderError for a missing folder, missing files (all listed), weights that exist only in an
    old format, a model that is not Whisper, or an English-only model when a language is requested.
    """
    if not folder.exists() or not folder.is_dir():
        raise ModelLoaderError(
            f"Model folder not found: {folder}. Copy the model folder there, or run tools/fetch_model.py at home. "
            "Do not enable online mode as a fix."
        )
    names = {p.name for p in folder.iterdir() if p.is_file()}
    missing = [n for n in ("config.json", "preprocessor_config.json") if n not in names]
    has_weights = "model.safetensors" in names or "model.safetensors.index.json" in names
    old = [n for n in OLD_WEIGHT_FILES if n in names]
    if not has_weights:
        if old:
            raise ModelLoaderError(
                f"The weights in {folder} are only in an old format ({', '.join(old)}); safetensors is required. "
                "Convert the model once at home (load it and save it again with safe serialization), then copy the folder."
            )
        missing.append("model.safetensors (the weights, in safetensors format)")
    if not ("tokenizer.json" in names or {"vocab.json", "merges.txt"} <= names):
        missing.append("tokenizer files (tokenizer.json, or vocab.json and merges.txt)")
    if "tokenizer_config.json" not in names:
        missing.append("tokenizer_config.json")
    if missing:
        raise ModelLoaderError(
            f"The model folder {folder} is incomplete. Missing: {', '.join(missing)}. "
            "Copy the full folder from home, or run tools/fetch_model.py there."
        )
    try:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ModelLoaderError(f"{folder / 'config.json'} is not valid JSON ({exc}). Copy the folder again.") from exc
    if config.get("model_type") != "whisper":
        raise ModelLoaderError(
            f"{folder} is not a Whisper model (model_type is '{config.get('model_type')}'). "
            "The loader needs a Whisper model: set model.path to a Whisper folder."
        )
    if language is not None and config.get("vocab_size") == ENGLISH_ONLY_VOCAB_SIZE:
        raise ModelLoaderError(
            f"The model {folder.name} is English-only and cannot be used with model.language '{language}'. "
            "Set model.path to a multilingual Whisper model."
        )
