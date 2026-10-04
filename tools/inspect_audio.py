"""Read-only look at a downloaded Hub parquet dataset: schema, metadata, real audio, label texts.

How to run (from the project root, with the project virtual environment active):
    python tools/inspect_audio.py --data data/daf_yomi_raw
Options: --seed 0 (fixed seed, same rows every time), --audio-samples 50, --text-samples 30
Example output (shortened):
    == A. Parquet files (metadata only) ==
    train-00000-of-00002.parquet: 4143 rows, 5 columns ...
    == C. Audio (50 rows, seed 0) ==
    sample rate: {16000: 50}  channels: {1: 50}  duration s: min 1.2 median 18.4 max 29.9
It changes nothing: no file is written or normalized. Audio is read one row group at a time
(never a whole file). Exit code 0 = success, 1 = the folder has no parquet files.
"""

from __future__ import annotations

import argparse
import io
import random
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import scipy.signal
import soundfile

WHISPER_WINDOW_S = 30.0
TARGET_SR = 16000

PUNCTUATION = set(".,;:!?-־–—\"'")
GERESH = set("׳״")  # geresh and gershayim, abbreviation marks (for example in "gmara-geresh")
BRACKETS = set("()[]{}<>")
LATIN = re.compile(r"[A-Za-z]")
DIGIT = re.compile(r"[0-9]")
NIKUD = re.compile(r"[֑-ׇ]")  # Hebrew points and cantillation marks
TIMESTAMP_LIKE = re.compile(r"<\|[^|]*\|>|\b\d{1,2}:\d{2}\b|\[\d+(\.\d+)?\]")
TIMESTAMP_TOKEN = re.compile(r"<\|[^|]*\|>")  # Whisper-style <|12.34|>
HEBREW_LETTER = re.compile(r"[א-ת]")


def find_parquet_files(data: Path) -> list[Path]:
    """Return the parquet files under `data`, sorted by name (train before eval is not assumed)."""
    return sorted(data.rglob("*.parquet"))


def schema_report(files: list[Path]) -> list[str]:
    """Part A: per file row count, row groups and columns from the parquet metadata only."""
    lines: list[str] = []
    total = 0
    for i, path in enumerate(files):
        parquet = pq.ParquetFile(path)
        total += parquet.metadata.num_rows
        lines.append(
            f"{path.name}: {parquet.metadata.num_rows} rows, "
            f"{parquet.metadata.num_row_groups} row groups, {parquet.metadata.num_columns} leaf columns"
        )
        if i == 0:
            for field in parquet.schema_arrow:
                lines.append(f"    column {field.name}: {field.type}")
    lines.append(f"TOTAL rows: {total}")
    return lines


def _describe(values: list[float]) -> str:
    if not values:
        return "no values"
    return f"min {min(values):.3g} median {statistics.median(values):.3g} max {max(values):.3g}"


def metadata_report(path: Path, top: int = 8) -> list[str]:
    """Part B: distributions of the metadata columns of one file (audio is NOT read)."""
    split = path.name.split("-")[0]
    table = pq.read_table(path, columns=["metadata", "has_prev", "has_timestamps"])
    meta = table.column("metadata").to_pylist()
    lines = [f"file: {path.name}  (the split is only in the file name: '{split}'; there is no split column)"]
    lines.append(f"rows: {len(meta)}")
    for key in ("source", "creator", "entry_id"):
        counts = Counter(str(m.get(key)) for m in meta)
        lines.append(f"{key}: {len(counts)} distinct; top: {counts.most_common(top)}")
    for key in ("duration", "quality_score", "number_of_segments", "bad_segments", "seek"):
        lines.append(f"{key}: {_describe([float(m[key]) for m in meta if m.get(key) is not None])}")
    lines.append(f"golden_segments: {dict(Counter(m.get('golden_segments') for m in meta))}")
    lines.append(f"has_prev: {dict(Counter(table.column('has_prev').to_pylist()))}")
    lines.append(f"has_timestamps: {dict(Counter(table.column('has_timestamps').to_pylist()))}")
    return lines


def choose_rows(counts: list[int], n: int, seed: int) -> list[tuple[int, int]]:
    """Pick `n` (file index, row index) pairs at random over all files with a fixed seed."""
    universe = [(f, r) for f, c in enumerate(counts) for r in range(c)]
    rng = random.Random(seed)
    return sorted(rng.sample(universe, min(n, len(universe))))


def audio_report(files: list[Path], n: int, seed: int) -> list[str]:
    """Part C: decode about `n` real rows (one row group at a time) and describe them."""
    parquets = [pq.ParquetFile(p) for p in files]
    picks = choose_rows([p.metadata.num_rows for p in parquets], n, seed)
    rates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    formats: Counter[str] = Counter()
    durations: list[tuple[float, str]] = []
    failures: list[str] = []
    by_group: dict[tuple[int, int], list[int]] = {}
    for f, row in picks:
        offset = 0
        for g in range(parquets[f].metadata.num_row_groups):
            size = parquets[f].metadata.row_group(g).num_rows
            if row < offset + size:
                by_group.setdefault((f, g), []).append(row - offset)
                break
            offset += size
    first_clip: tuple[np.ndarray, int] | None = None
    for (f, g), rows in sorted(by_group.items()):
        audio = parquets[f].read_row_group(g, columns=["audio"]).column("audio").to_pylist()
        for r in rows:
            label = f"{files[f].name}#group{g}:row{r}"
            data = audio[r].get("bytes") if isinstance(audio[r], dict) else None
            try:
                if not data:
                    raise ValueError("empty audio bytes")
                info = soundfile.info(io.BytesIO(data))
                samples, sr = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
            except Exception as exc:  # report every decode failure with its reason
                failures.append(f"{label}: {type(exc).__name__}: {exc}")
                continue
            rates[sr] += 1
            channels[samples.shape[1]] += 1
            formats[f"{info.format}/{info.subtype}"] += 1
            durations.append((samples.shape[0] / sr, label))
            if first_clip is None:
                first_clip = (samples.mean(axis=1), sr)
    lines = [f"rows requested {len(picks)}, decoded {len(durations)}, failed {len(failures)}"]
    lines.append(f"container/codec: {dict(formats)}")
    lines.append(f"sample rate: {dict(rates)}  channels: {dict(channels)}")
    values = [d for d, _ in durations]
    lines.append(f"duration s: {_describe(values)}")
    long_clips = sorted((d for d in durations if d[0] > WHISPER_WINDOW_S), reverse=True)
    lines.append(f"longer than {WHISPER_WINDOW_S:.0f} s: {len(long_clips)} of {len(durations)}")
    for d, label in long_clips[:5]:
        lines.append(f"    {d:.2f} s  {label}")
    top = sorted(durations, reverse=True)[:3]
    lines.append("longest 3: " + ", ".join(f"{d:.2f} s" for d, _ in top))
    for failure in failures:
        lines.append(f"DECODE FAILURE {failure}")
    if first_clip is not None:
        clip, sr = first_clip
        source = "native" if sr != TARGET_SR else "synthetic 44.1 kHz clip of the same length (native is already 16 kHz)"
        if sr == TARGET_SR:
            clip = np.resize(clip, int(len(clip) * 44100 / sr))
            sr = 44100
        start = time.perf_counter()
        out = scipy.signal.resample_poly(clip, TARGET_SR, sr)
        elapsed = time.perf_counter() - start
        lines.append(
            f"resample_poly {sr} -> {TARGET_SR} Hz of one {len(out) / TARGET_SR:.1f} s clip ({source}): {elapsed * 1000:.1f} ms"
        )
    return lines


def text_flags(text: str) -> dict[str, bool]:
    """Simple yes/no properties of one label text (nothing is changed or stored).

    Whisper-style timestamp tokens such as <|0.44|> are masked out FIRST for every property
    except `timestamp_like`, because they contain digits and angle brackets themselves.
    """
    core = TIMESTAMP_TOKEN.sub("", text)
    stripped = core.strip()
    return {
        "empty_after_masking_timestamps": stripped == "",
        "very_short(<5 chars)": 0 < len(stripped) < 5,
        "punctuation": any(c in PUNCTUATION for c in core),
        "geresh_or_gershayim": any(c in GERESH for c in core),
        "latin_letters": bool(LATIN.search(core)),
        "digits": bool(DIGIT.search(core)),
        "brackets": any(c in BRACKETS for c in core),
        "timestamp_like": bool(TIMESTAMP_LIKE.search(text)),
        "nikud_or_cantillation": bool(NIKUD.search(core)),
    }


def count_flags(texts: list[str]) -> dict[str, int]:
    """Count how many of `texts` have each property of text_flags."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(k for k, v in text_flags(text).items() if v)
    return dict(counts)


def text_report(files: list[Path], n: int, seed: int) -> list[str]:
    """Part D: print `n` raw labels (fixed seed) and count simple properties.

    Counts are given for the printed sample and, because the transcript column is small and
    read without audio, for ALL rows of all files.
    """
    all_texts: list[str] = []
    for path in files:
        all_texts.extend(pq.read_table(path, columns=["transcript"]).column("transcript").to_pylist())
    all_texts = [t if t is not None else "" for t in all_texts]
    rng = random.Random(seed)
    picks = sorted(rng.sample(range(len(all_texts)), min(n, len(all_texts))))
    lines = [f"-- {len(picks)} raw labels (seed {seed}), exactly as stored --"]
    lines += [f"[{i}] {all_texts[i]!r}" for i in picks]
    lines.append(f"-- property counts in these {len(picks)} labels --")
    lines.append(str(count_flags([all_texts[i] for i in picks])))
    lens = [len(t) for t in all_texts]
    lines.append(f"-- property counts in ALL {len(all_texts)} labels (text column only) --")
    lines.append(str(count_flags(all_texts)))
    lines.append(f"label length in characters (all): {_describe([float(x) for x in lens])}")
    stamps = [len(TIMESTAMP_TOKEN.findall(t)) for t in all_texts]
    lines.append(f"timestamp tokens per label (all): {_describe([float(x) for x in stamps])}")
    others: Counter[str] = Counter()
    for t in all_texts:
        others.update(c for c in TIMESTAMP_TOKEN.sub("", t) if not HEBREW_LETTER.match(c) and not c.isspace())
    lines.append(f"non-Hebrew-letter characters after masking timestamps (all labels), top 25: {others.most_common(25)}")
    return lines


def overlap_report(files: list[Path]) -> list[str]:
    """Part B2: do the same `entry_id` values appear in different splits? (metadata only)

    The split name is the part of the file name before the first '-', for example 'train'.
    """
    ids: dict[str, set[str]] = {}
    for path in files:
        meta = pq.read_table(path, columns=["metadata"]).column("metadata").to_pylist()
        ids.setdefault(path.name.split("-")[0], set()).update(str(m.get("entry_id")) for m in meta)
    lines = [f"{name}: {len(values)} distinct entry_id" for name, values in sorted(ids.items())]
    names = sorted(ids)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = ids[a] & ids[b]
            lines.append(
                f"entry_id shared by {a} and {b}: {len(shared)} ({100 * len(shared) / max(1, len(ids[b])):.1f}% of {b})"
            )
    return lines


def build_report(data: Path, seed: int, audio_samples: int, text_samples: int) -> str:
    """Return the full text report. Raises FileNotFoundError when there are no parquet files."""
    files = find_parquet_files(data)
    if not files:
        raise FileNotFoundError(f"No .parquet files under {data}. Run tools/fetch_data.py first.")
    sections = [
        ("A. Parquet files (metadata only)", schema_report(files)),
        ("B. Metadata columns of the first file", metadata_report(files[0])),
        ("B2. entry_id overlap between the splits (O7, O42)", overlap_report(files)),
        (f"C. Audio ({audio_samples} real rows, seed {seed})", audio_report(files, audio_samples, seed)),
        ("D. Label texts", text_report(files, text_samples, seed)),
    ]
    return "\n".join(f"== {title} ==\n" + "\n".join(lines) + "\n" for title, lines in sections)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry. Returns the exit code."""
    parser = argparse.ArgumentParser(description="Read-only inspection of a parquet dataset.")
    parser.add_argument("--data", type=Path, required=True, help="folder with the parquet files")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--audio-samples", type=int, default=50)
    parser.add_argument("--text-samples", type=int, default=30)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Hebrew in the Windows console
    try:
        print(build_report(args.data, args.seed, args.audio_samples, args.text_samples))
    except FileNotFoundError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
