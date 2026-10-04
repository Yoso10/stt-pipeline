"""Download a model from the Hugging Face Hub into a local folder (home-side, needs internet).

How to run (from the project root, with the project virtual environment active):
    python tools/fetch_model.py --repo openai/whisper-tiny --out models/whisper-tiny
Example output:
    Created: models/whisper-tiny  (12 files, 151.1 MB)
    Next: set  model.path: models/whisper-tiny  in your config YAML.
Running it again reuses the finished folder. Only safetensors weights and the small config and
tokenizer files are downloaded (no .bin, TensorFlow or Flax copies). Exit code 0 = success.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    Downloader,
    FetchError,
    FetchResult,
    fetch_repo,
    hub_downloader,
)

MODEL_PATTERNS = ["*.json", "*.txt", "*.safetensors", "*.model"]


def missing_model_files(folder: Path) -> list[str]:
    """Return the files the ModelLoader requires that are absent from `folder`."""
    names = {p.name for p in folder.iterdir() if p.is_file()}
    missing = [n for n in ("config.json", "preprocessor_config.json") if n not in names]
    if "model.safetensors" not in names and "model.safetensors.index.json" not in names:
        missing.append("model.safetensors (weights in safetensors format)")
    has_tokenizer = "tokenizer.json" in names or {"vocab.json", "merges.txt"} <= names
    if not has_tokenizer:
        missing.append("tokenizer files (tokenizer.json, or vocab.json and merges.txt)")
    return missing


def fetch_model(repo: str, out: Path, downloader: Downloader = hub_downloader) -> FetchResult:
    """Download model `repo` into `out`; return FetchResult. Raises FetchError on failure."""
    return fetch_repo(repo, "model", out, MODEL_PATTERNS, downloader, missing_model_files)


def main(argv: list[str] | None = None, downloader: Downloader = hub_downloader) -> int:
    """Command-line entry. Returns the exit code. `downloader` is replaced in tests."""
    parser = argparse.ArgumentParser(description="Download a Hub model into a local folder.")
    parser.add_argument("--repo", required=True, help="Hub name, for example openai/whisper-tiny")
    parser.add_argument("--out", required=True, type=Path, help="destination folder")
    args = parser.parse_args(argv)
    try:
        result = fetch_model(args.repo, args.out, downloader)
    except FetchError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    verb = "Reused (already complete)" if result.reused else "Created"
    n = len(result.info["files"])  # type: ignore[arg-type]
    mb = int(str(result.info["total_bytes"])) / 1e6
    print(f"{verb}: {result.out}  ({n} files, {mb:.1f} MB)")
    print(f"Next: set  model.path: {result.out.as_posix()}  in your config YAML.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
