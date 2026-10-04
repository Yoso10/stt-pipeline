"""Builds the returned model folder from a checkpoint (standard library only).

How to run: this module is not a script; TrainerCore calls it at the end of a run. The folder `<output_dir>/model` gets: the
weights and the model settings copied from the chosen checkpoint (safetensors), the tokenizer and feature-extractor files copied
unchanged from the base model folder, and `pipeline_model_info.json`. The ModelLoader loads the folder like any other model and reads
`source_config_ref` from the info file. Example (PowerShell, project root; after a run, `data/run_x` is its output folder):
    $env:PYTHONPATH = "src"
    python -c "from pathlib import Path; from stt.modelloader import ModelConfig, create_loader; print(create_loader(ModelConfig(path=Path('data/run_x/model'))).load().info)"
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from stt.trainercore.errors import TrainerCoreError

INFO_NAME = "pipeline_model_info.json"
# Copied from the checkpoint: the weights (also when sharded) and the model settings.
CHECKPOINT_FILES = ("config.json", "generation_config.json", "model.safetensors", "model.safetensors.index.json")
# Copied unchanged from the base model folder (a fixed list: nothing else is copied, so nothing unexpected ends up in the folder).
SUPPORT_FILES = (
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
    "added_tokens.json",
    "special_tokens_map.json",
)


def build_model_dir(checkpoint: Path, base_model_dir: Path, target: Path, info: dict[str, Any]) -> Path:
    """Create `target` (replacing an older one of the same run) and return it.

    Errors: TrainerCoreError when the checkpoint holds no weights or the base model folder is gone (the checkpoints stay).
    """
    if not base_model_dir.is_dir():
        raise TrainerCoreError(
            f"The base model folder {base_model_dir} is gone, so the tokenizer files cannot be copied into the result. "
            f"The weights are safe in {checkpoint}. Restore the base model folder and run again with train.resume true."
        )
    temporary = target.with_name(target.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    copied = [name for name in CHECKPOINT_FILES if (checkpoint / name).is_file()]
    shards = sorted(p.name for p in checkpoint.glob("model-*.safetensors"))
    if "model.safetensors" not in copied and not shards:
        raise TrainerCoreError(f"The checkpoint {checkpoint} holds no safetensors weights, so no model can be returned.")
    for name in [*copied, *shards]:
        shutil.copy2(checkpoint / name, temporary / name)
    for name in SUPPORT_FILES:
        if (base_model_dir / name).is_file():
            shutil.copy2(base_model_dir / name, temporary / name)
    (temporary / INFO_NAME).write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(target, ignore_errors=True)
    temporary.rename(target)
    return target
