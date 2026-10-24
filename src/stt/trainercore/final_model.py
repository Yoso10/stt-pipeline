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
import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stt.trainercore.errors import TrainerCoreError

logger = logging.getLogger("stt.trainercore")
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


def _put_in_place(temporary: Path, target: Path, checkpoint: Path, retries: int, seconds: float, sleep: Callable[[float], None]) -> None:
    """Rename the finished `temporary` folder to `target`, trying again after a failure (O122).

    On Windows another program (the antivirus, the search indexer, a sync tool) can hold a new folder for a moment, so the rename fails with `PermissionError`
    ("Access is denied"), an `OSError`. The rename is tried `retries` more times, waiting `seconds` before each new attempt. Before every attempt an older folder
    `target` (the model of an earlier part of a resumed run) is removed, as it always was. When every attempt fails NOTHING is deleted: the `temporary` folder
    (the complete model) and the finished `checkpoint` stay, and the error says where they are and how to continue.
    Errors: TrainerCoreError after the last failed attempt.
    """
    attempts = retries + 1
    last: OSError | None = None
    for attempt in range(1, attempts + 1):
        try:
            shutil.rmtree(target, ignore_errors=True)
            temporary.rename(target)
        except OSError as exc:  # PermissionError is an OSError
            last = exc
            logger.warning(
                "Attempt %d of %d to rename %s to %s failed (%s: %s).%s",
                attempt, attempts, temporary.name, target.name, type(exc).__name__, exc, f" Waiting {seconds:g} s before the next attempt." if attempt < attempts else "",
            )
            if attempt < attempts:
                sleep(seconds)
            continue
        if attempt > 1:
            logger.warning("The model folder %s was put in place at attempt %d of %d.", target, attempt, attempts)
        return
    raise TrainerCoreError(
        f"The model of the finished training could not be put in place: renaming {temporary} to {target} failed {attempts} time(s) "
        f"(last error: {type(last).__name__}: {last}); {seconds:g} s were waited between the attempts. NOTHING WAS DELETED: the complete model is in {temporary} "
        f"and the finished checkpoint is in {checkpoint}. Usually another program (the antivirus, the search indexer or a sync tool) holds the new folder for a moment. "
        f"To continue: close or pause that program, or exclude the project folder from it; then EITHER rename the folder by hand "
        f"(PowerShell: Rename-Item -LiteralPath '{temporary}' -NewName '{target.name}'), OR run the same command again with train.resume true "
        f"(it continues from the finished checkpoint and builds the model folder again); raise train.rename_retries or train.rename_retry_seconds if this repeats."
    )


def build_model_dir(
    checkpoint: Path,
    base_model_dir: Path,
    target: Path,
    info: dict[str, Any],
    *,
    rename_retries: int = 0,
    rename_retry_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Create `target` (replacing an older one of the same run) and return it.

    `rename_retries` and `rename_retry_seconds` (the keys `train.rename_retries` and `train.rename_retry_seconds`, which the TrainerCore passes) say how often and how long
    to wait before trying again when the final rename fails; the defaults here (no retry) are the behaviour of a direct call. `sleep` is injectable for tests.
    Errors: TrainerCoreError when the checkpoint holds no weights or the base model folder is gone (the checkpoints stay), or when the final rename fails
    on every attempt (the `.tmp` folder and the checkpoint stay).
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
    _put_in_place(temporary, target, checkpoint, rename_retries, rename_retry_seconds, sleep)
    return target
