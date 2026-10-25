"""The encoder-output cache: one file per clip with the output of the FROZEN audio encoder (DECISIONS 89). No torch in this module.

How to run: this module is not a script; the FeaturePipeline uses it when `features.encoder_cache.enabled` is true. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from pathlib import Path; from stt.featurepipeline.encoder_cache import EncoderCache; print(EncoderCache(Path('cache'), 'abc', 'float16').folder)"
Example output:
    cache\\enc_abc_float16
Why: with a frozen encoder the encoder output of a clip never changes, and the encoder is about 96% of the time of a training step, of a Val-loss pass and about 70% of a
decoding (Task 3 measurements). Files: `<folder>/<key>.npy`, written to a temporary name and renamed (a crash never leaves a half file under a final name; a stopped build
continues because finished files are kept), and `selfcheck.json` (written once the cached and uncached losses agreed). The folder name holds the first 16 characters of a
fingerprint of the base encoder weights, the feature extractor and the dtype; the file key holds the sample id, the audio path, the size and modification time of the audio
file, the dtype and the same fingerprint, so a changed encoder, extractor, dtype or audio file never reuses an old file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np

from stt.config import ConfigError

log = logging.getLogger("stt.featurepipeline")

DTYPES = {"float16": np.float16, "float32": np.float32}
# The largest relative difference between the Val loss with and without the cache that the one-time self-check accepts. float32 storage is exact (the check on the
# turbo model showed a bit-identical loss); float16 rounds each value (measured: a relative difference of the encoder output of 2e-4 and of the loss below 1e-4).
SELF_CHECK_TOLERANCE = {"float32": 1e-5, "float16": 2e-3}
SELF_CHECK_CLIPS = 3


class EncoderCache:
    """Files in `<directory>/enc_<fingerprint[:16]>_<dtype>/`: `<key>.npy` per clip and `selfcheck.json`."""

    def __init__(self, directory: Path, fingerprint: str, dtype: str) -> None:
        if dtype not in DTYPES:
            raise ConfigError(f"features.encoder_cache.dtype '{dtype}' is not valid. Use one of: {', '.join(DTYPES)}.")
        self.directory = directory
        self.fingerprint = fingerprint
        self.dtype = dtype
        self.folder = directory / f"enc_{fingerprint[:16]}_{dtype}"

    def key(self, sample_id: str, audio_path: Path) -> str:
        """The key of a clip: id, audio path, size and modification time of the audio file, dtype and the fingerprint of the encoder."""
        stat = audio_path.stat()
        text = f"{sample_id}|{audio_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{self.dtype}|{self.fingerprint}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def has(self, key: str) -> bool:
        """True when a finished file exists for `key`."""
        return (self.folder / f"{key}.npy").is_file()

    def get(self, key: str) -> np.ndarray | None:
        """The cached encoder output as float32, or None when it is missing or damaged (the caller rebuilds)."""
        path = self.folder / f"{key}.npy"
        if not path.is_file():
            return None
        try:
            return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        except (OSError, ValueError):
            log.warning("damaged encoder-cache file %s is ignored and rebuilt", path)
            return None

    def put(self, key: str, output: np.ndarray) -> None:
        """Store `output` (converted to the cache dtype) atomically: a temporary file, then a rename. Errors: ConfigError when the folder cannot be written."""
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            temp = self.folder / f"{key}.{os.getpid()}.tmp"
            with temp.open("wb") as handle:
                np.save(handle, output.astype(DTYPES[self.dtype], copy=False), allow_pickle=False)
            temp.replace(self.folder / f"{key}.npy")
        except OSError as exc:
            raise ConfigError(
                f"The encoder cache folder {self.folder} cannot be written ({exc}). Fix: set features.encoder_cache.dir to a folder you can write to, with enough free space, "
                "or set features.encoder_cache.enabled to false."
            ) from exc

    def self_check_done(self) -> bool:
        """True when the one-time check of the cached against the uncached loss already passed for this folder."""
        return (self.folder / "selfcheck.json").is_file()

    def mark_self_check(self, relative_differences: list[float]) -> None:
        """Remember that the self-check passed (written after the numbers are known)."""
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / "selfcheck.json").write_text(json.dumps({"relative_loss_differences": relative_differences, "dtype": self.dtype}), encoding="utf-8")


def bytes_per_clip(frames: int, width: int, dtype: str) -> int:
    """The size of one cached encoder output."""
    return frames * width * int(np.dtype(DTYPES[dtype]).itemsize)


def check_disk(directory: Path, needed_bytes: int, free_bytes: int | None = None) -> None:
    """Refuse, with the numbers, when the folder's drive has less free space than the missing files need.

    `free_bytes` is injectable for tests. Errors: ConfigError (exit code 2).
    """
    if free_bytes is None:
        probe = directory
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        free_bytes = shutil.disk_usage(probe).free
    if free_bytes < needed_bytes:
        gb = 1e9
        raise ConfigError(
            f"Not enough free disk for the encoder cache in {directory}: it needs {needed_bytes / gb:.2f} GB and {free_bytes / gb:.2f} GB are free (short by {(needed_bytes - free_bytes) / gb:.2f} GB). "
            "Fix: free disk space, set features.encoder_cache.dtype to float16 (half the size) if it is float32, point features.encoder_cache.dir to a bigger drive, "
            "or set features.encoder_cache.enabled to false."
        )


class Progress:
    """Logs `clips done / total` and an ETA about every 5 percent (count based, so there is no hard-coded time)."""

    def __init__(self, total: int, clock: Callable[[], float]) -> None:
        self._total = total
        self._clock = clock
        self._start = clock()
        self._step = max(1, total // 20)

    def update(self, done: int) -> None:
        """Call after each finished clip."""
        if done % self._step and done != self._total:
            return
        elapsed = self._clock() - self._start
        remaining = elapsed / done * (self._total - done) if done else 0.0
        log.info("Encoder cache: %d of %d clips done, elapsed %.0f s, remaining about %.0f s.", done, self._total, elapsed, remaining)
