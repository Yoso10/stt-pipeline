"""Build the tiny random-weight Whisper-like test model with the REAL tokenizer and feature extractor (home-side).

How to run (from the project root, with the project virtual environment active):
    python tools/make_test_model.py --tokenizer-from models/whisper-tiny --out tests/assets/tiny_whisper
Options: --seed 0 (the random weights are the same for the same seed), --force (replace an existing output folder).
Example output:
    Created: tests/assets/tiny_whisper  (8 files, 3.9 MB, 0.2 million parameters)
    Verified: loads through the ModelLoader and runs a forward pass (loss 10.9)
    Next: the tests and `smoke` use this folder as model.path.
What it does: copies the tokenizer and feature-extractor files from the local model folder (no network), creates a
very small Whisper (a few thousand weights besides the embedding of the real vocabulary), saves it in safetensors
format, then loads it through the ModelLoader and runs one forward pass. The folder stays under 5 MB so it can be
stored with the tests. Exit code 0 = success, 1 = failure (the message names the missing files).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from _common import FetchError  # noqa: E402

COPY_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "generation_config.json",
)
REQUIRED_SOURCE_FILES = ("config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json")
MAX_BYTES = 5_000_000


def check_source(source: Path) -> None:
    """Fail naming the missing files of the source model folder (raises FetchError)."""
    if not source.is_dir():
        raise FetchError(f"Source model folder not found: {source}. Run tools/fetch_model.py first (for example for openai/whisper-tiny).")
    missing = [n for n in REQUIRED_SOURCE_FILES if not (source / n).is_file()]
    if missing:
        raise FetchError(f"The source folder {source} lacks: {', '.join(missing)}. Run tools/fetch_model.py again.")


def build_tiny_model(source: Path, out: Path, seed: int) -> int:
    """Write the tiny model to `out` (a new folder). Returns the parameter count."""
    import torch
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    config = WhisperConfig.from_pretrained(str(source), local_files_only=True)
    config.d_model = 4
    config.encoder_layers = 1
    config.decoder_layers = 1
    config.encoder_attention_heads = 2
    config.decoder_attention_heads = 2
    config.encoder_ffn_dim = 8
    config.decoder_ffn_dim = 8
    config.num_hidden_layers = 1
    torch.manual_seed(seed)
    model = WhisperForConditionalGeneration(config)
    out.mkdir(parents=True)
    model.save_pretrained(str(out), safe_serialization=True)
    for name in COPY_FILES:
        if (source / name).is_file():
            shutil.copyfile(source / name, out / name)
    return int(sum(p.numel() for p in model.parameters()))


def verify(out: Path) -> float:
    """Load `out` through the ModelLoader and run one forward pass. Returns the loss."""
    import numpy as np
    import torch

    from stt.modelloader import ModelConfig, create_loader

    loaded = create_loader(ModelConfig(path=out)).load()
    features = loaded.feature_extractor.extract(np.zeros(16000, dtype=np.float32), 16000)
    labels = torch.tensor([loaded.tokenizer.encode_label("שלום עולם")])
    with torch.no_grad():
        result = loaded.model(input_features=torch.from_numpy(features)[None], labels=labels)  # type: ignore[operator]
    return float(result.loss)


def make_test_model(source: Path, out: Path, seed: int = 0, force: bool = False) -> dict[str, object]:
    """Create and verify the tiny test model; return a summary. Raises FetchError on failure."""
    check_source(source)
    if out.exists():
        if not force:
            raise FetchError(f"{out} already exists. Use --force to replace it (it is derived data, nothing else is deleted).")
        shutil.rmtree(out)
    try:
        parameters = build_tiny_model(source, out, seed)
        total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        if total > MAX_BYTES:
            raise FetchError(f"The test model is {total / 1e6:.1f} MB, above the {MAX_BYTES / 1e6:.0f} MB limit for files stored with the tests.")
        loss = verify(out)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return {"out": out, "files": len(list(out.iterdir())), "bytes": total, "parameters": parameters, "loss": loss}


def main(argv: list[str] | None = None) -> int:
    """Command-line entry. Returns the exit code."""
    parser = argparse.ArgumentParser(description="Build the tiny random-weight Whisper test model.")
    parser.add_argument("--tokenizer-from", required=True, type=Path, help="local model folder with the tokenizer and feature extractor")
    parser.add_argument("--out", required=True, type=Path, help="destination folder")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = make_test_model(args.tokenizer_from, args.out, args.seed, args.force)
    except FetchError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"Created: {result['out']}  ({result['files']} files, {int(result['bytes']) / 1e6:.1f} MB, {int(result['parameters']) / 1e6:.1f} million parameters)")  # type: ignore[call-overload]
    print(f"Verified: loads through the ModelLoader and runs a forward pass (loss {float(result['loss']):.1f})")  # type: ignore[arg-type]
    print("Next: the tests and `smoke` use this folder as model.path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
