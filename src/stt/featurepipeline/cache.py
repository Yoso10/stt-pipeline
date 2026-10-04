"""The optional feature cache: one `.npy` file per sample, never reused when anything it depends on changed.

How to run: this module is not a script; the dataset uses it when `features.cache.enabled` is true. Example
(PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.featurepipeline.cache import FeatureCache; from pathlib import Path; print(FeatureCache(Path('cache'), 'abc').folder)"
Example output:
    cache\abc
A cache entry is found by a key made of the sample id, the audio path and the size and modification time of the
audio file, inside a folder named after the signature of the extractor (its numbers plus a hash of the features of a
fixed test signal). A changed extractor gives a new folder, a changed audio file a new key; a damaged file is rebuilt.
Writing is crash-safe (temporary file, then rename); a folder that cannot be written is skipped, not an error.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from stt.modelloader import FeatureExtractor

log = logging.getLogger(__name__)


def extractor_signature(extractor: FeatureExtractor, downmix: str) -> str:
    """A hash that changes when the extractor or the audio conversion changes.

    It covers the sampling rate, the window, the mel bins, the downmix mode and the features of a fixed 1 s tone.
    """
    rate = extractor.sampling_rate
    tone = (0.3 * np.sin(2 * np.pi * 440.0 * np.arange(rate) / rate)).astype(np.float32)
    probe = np.round(extractor.extract(tone, rate), 4).astype(np.float32)
    meta = json.dumps(
        {"rate": rate, "window": extractor.window_seconds, "mel": extractor.num_mel_bins, "downmix": downmix, "shape": list(probe.shape)}
    )
    return hashlib.sha256(meta.encode() + probe.tobytes()).hexdigest()


class FeatureCache:
    """Files in `<directory>/<signature[:16]>/`: `<key>.npy` per sample and `signature.json`."""

    def __init__(self, directory: Path, signature: str) -> None:
        self.folder = directory / signature[:16]
        self._signature = signature
        self._writable = True

    def key(self, sample_id: str, audio_path: Path) -> str:
        """The key of a sample: id, path, and the size and modification time of the audio file."""
        stat = audio_path.stat()
        text = f"{sample_id}|{audio_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def has(self, key: str) -> bool:
        """True when a file exists for `key`."""
        return (self.folder / f"{key}.npy").is_file()

    def get(self, key: str) -> np.ndarray | None:
        """The cached features, or None when missing or damaged (the caller rebuilds)."""
        path = self.folder / f"{key}.npy"
        if not path.is_file():
            return None
        try:
            return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        except (OSError, ValueError):
            log.warning("damaged cache file %s is ignored and rebuilt", path)
            return None

    def put(self, key: str, features: np.ndarray) -> bool:
        """Store `features`; returns False (and stops trying) when the folder cannot be written."""
        if not self._writable:
            return False
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            marker = self.folder / "signature.json"
            if not marker.exists():
                marker.write_text(json.dumps({"signature": self._signature}), encoding="utf-8")
            temp = self.folder / f"{key}.tmp.npy"
            np.save(temp, features.astype(np.float32, copy=False), allow_pickle=False)
            temp.replace(self.folder / f"{key}.npy")
            return True
        except OSError as exc:
            log.warning("the cache folder %s cannot be written (%s); features are computed without the cache", self.folder, exc)
            self._writable = False
            return False
