"""Synthetic smoke data: short sine-tone clips with random Hebrew transcripts (standard library only).

How to run: this module is not a script; the `smoke` command calls it. Example (PowerShell, project root; writes 3 clips into a temporary folder):
    $env:PYTHONPATH = "src"
    python -c "import tempfile; from pathlib import Path; from stt.orchestrator import SmokeConfig, write_smoke_data; d = Path(tempfile.mkdtemp()); print(write_smoke_data(SmokeConfig(n_samples=3), 42, d)); print((d / 'manifest.csv').read_text(encoding='utf-8-sig'))"
Example output (the folder name varies): the folder, then `audio,text,group` and one row per clip, for example `audio/smoke_000.wav,...,g000`.
The clips are 16 kHz mono 16-bit wav files (the standard `wave` module), one tone per clip; every clip is its own group, so the Splitter can
split them. The same seed gives the same files. The point is to exercise the mechanics of the whole chain, not the quality of the model.
"""

from __future__ import annotations

import csv
import math
import random
import struct
import wave
from pathlib import Path

from stt.orchestrator.config import SmokeConfig

SAMPLE_RATE = 16000
WORDS = ["שלום", "עולם", "בית", "ספר", "מים", "אור", "יום", "לילה", "דרך", "שמש", "ים", "הר", "עיר", "גן", "שיר", "חלום", "לב", "קול"]


def write_smoke_data(config: SmokeConfig, seed: int, folder: Path) -> Path:
    """Write `config.n_samples` wav clips and `manifest.csv` (columns audio, text, group) into `folder` and return it.

    Existing files of the same names are overwritten (the content is deterministic for the seed).
    """
    audio_dir = folder / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = []
    frames = int(config.duration_sec * SAMPLE_RATE)
    for i in range(config.n_samples):
        frequency = 220.0 + 40.0 * (i % 9)
        samples = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * frequency * n / SAMPLE_RATE))) for n in range(frames))
        name = f"audio/smoke_{i:03d}.wav"
        with wave.open(str(folder / name), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(samples)
        rows.append([name, " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 6))), f"g{i:03d}"])
    with (folder / "manifest.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["audio", "text", "group"])
        writer.writerows(rows)
    return folder
