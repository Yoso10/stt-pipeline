r"""make_sample_pack: copy a small, reproducible slice of an extracted dataset (manifest.csv + audio) into a pack folder.

How to run (PowerShell, project root; the full data is only READ, the pack folder is written):
    $env:PYTHONPATH = "src"
    .\.venv\Scripts\python.exe tools\make_sample_pack.py --source data\extracted\daf_yomi --group-column meta_entry_id --groups 80 --per-group 5 --seed 42 --out data1 --report-only
    .\.venv\Scripts\python.exe tools\make_sample_pack.py --source data\extracted\daf_yomi --group-column meta_entry_id --groups 80 --per-group 5 --seed 42 --out data1 --date 2026-11-09
Options: --rejected FILE (the rejected.csv of the full-data prepare; its ids are excluded), --max-seconds 30 (longest clip), --id-column id, --audio-column audio, --date YYYY-MM-DD (written into pack_info.json; default: today).
Example output (shortened):
    rows 8723, eligible 8723, groups 710, groups with >= 5 eligible clips 705
    selected 400 clips in 80 groups -> data1 (xx.x MB)
What it does: keeps only ELIGIBLE rows (audio file exists, transcript not empty, duration at most --max-seconds read from the audio header with the pipeline's own reader, id not in --rejected);
picks --groups groups by a stable hash of (seed, group id) among the groups with at least --per-group eligible clips, and from each group --per-group clips by a stable hash of (seed, group id, clip id).
The result does not depend on the order of the rows, and the same inputs give byte-identical files (pack_info.json differs only in its date unless --date is given).
It writes <out>/manifest.csv (UTF-8, every original column and original id; the audio column holds a path relative to the pack folder), <out>/audio/<ascii id>.<ext>,
<out>/SHA256SUMS.txt (the AUDIO files only: `<hash>  audio/<name>`, so a line-ending conversion of the text files cannot break it) and <out>/pack_info.json.
It refuses to write into a folder that already has a manifest.csv (nothing is overwritten or deleted). Exit code 0 = done, 2 = bad input or too few groups, 3 = the output folder is not empty.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

Row = dict[str, str]
_SAFE = re.compile(r"[^A-Za-z0-9_-]")


def stable_key(*parts: object) -> str:
    """The hash that orders groups and clips: SHA-256 of the parts joined with `|` (independent of the row order)."""
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def select_clips(eligible: Sequence[Row], id_column: str, group_column: str, groups: int, per_group: int, seed: int) -> list[Row]:
    """The chosen rows, sorted by id.

    Only groups with at least `per_group` eligible rows can be chosen; the `groups` groups with the smallest stable key are taken, and from each the `per_group` rows
    with the smallest stable key. Errors: ValueError when fewer than `groups` groups qualify (the numbers are named).
    """
    by_group: dict[str, list[Row]] = {}
    for row in eligible:
        by_group.setdefault(row[group_column], []).append(row)
    usable = sorted((g for g, rows in by_group.items() if len(rows) >= per_group), key=lambda g: (stable_key(seed, g), g))
    if len(usable) < groups:
        raise ValueError(f"only {len(usable)} groups have at least {per_group} eligible clips, but --groups is {groups}. Lower --groups or --per-group.")
    chosen: list[Row] = []
    for g in usable[:groups]:
        rows = sorted(by_group[g], key=lambda r: (stable_key(seed, g, r[id_column]), r[id_column]))
        chosen.extend(rows[:per_group])
    return sorted(chosen, key=lambda r: r[id_column])


def ascii_name(sample_id: str) -> str:
    """A file name stem that is ASCII only: every other character becomes `_`."""
    return _SAFE.sub("_", sample_id)


def is_eligible(row: Row, source: Path, audio_column: str, text_column: str, id_column: str, rejected: set[str], max_seconds: float, duration_of: Callable[[Path], float]) -> bool:
    """True when the audio file exists, the transcript is not empty, the id is not rejected and the duration (from the header) is at most `max_seconds`."""
    if row[id_column].strip() in rejected or not row[text_column].strip():
        return False
    path = source / row[audio_column]
    if not path.is_file():
        return False
    try:
        return duration_of(path) <= max_seconds
    except ValueError:
        return False


def read_rejected(path: Path | None) -> set[str]:
    """The ids listed in a rejected.csv (column `id`); an empty set when no file is given."""
    if path is None:
        return set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {r["id"].strip() for r in csv.DictReader(handle)}


def describe(values: Sequence[float]) -> str:
    """min / median / max of a list of numbers as text."""
    return "none" if not values else f"min {min(values):g} / median {statistics.median(values):g} / max {max(values):g}"


def write_pack(chosen: Sequence[Row], header: Sequence[str], source: Path, out: Path, id_column: str, audio_column: str, info: dict[str, object]) -> int:
    """Write the pack folder; returns the total size in bytes. Errors: ValueError for two ids that give the same ASCII name."""
    names: dict[str, str] = {}
    for row in chosen:
        name = ascii_name(row[id_column]) + Path(row[audio_column]).suffix.lower()
        if name in names:
            raise ValueError(f"the ids {names[name]!r} and {row[id_column]!r} give the same file name {name}.")
        names[name] = row[id_column]
    (out / "audio").mkdir(parents=True)
    rows_out: list[Row] = []
    sums: list[str] = []
    total = 0
    for row in chosen:
        name = ascii_name(row[id_column]) + Path(row[audio_column]).suffix.lower()
        target = out / "audio" / name
        shutil.copyfile(source / row[audio_column], target)
        data = target.read_bytes()
        total += len(data)
        sums.append(f"{hashlib.sha256(data).hexdigest()}  audio/{name}")
        rows_out.append({**row, audio_column: f"audio/{name}"})
    with (out / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)
    (out / "SHA256SUMS.txt").write_text("\n".join(sorted(sums, key=lambda s: s.split("  ", 1)[1])) + "\n", encoding="utf-8", newline="\n")
    info = {**info, "audio_bytes_total": total}
    (out / "pack_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return total + sum(len(p.read_bytes()) for p in (out / "manifest.csv", out / "SHA256SUMS.txt", out / "pack_info.json"))


def report(rows: Sequence[Row], eligible: Sequence[Row], group_column: str, durations: Sequence[float], per_group: int) -> list[str]:
    """The numbers to look at before building (printed by --report-only and before every build)."""
    all_groups = Counter(r[group_column] for r in rows)
    ok_groups = Counter(r[group_column] for r in eligible)
    return [
        f"rows {len(rows)}, eligible {len(eligible)}, groups {len(all_groups)}, groups with eligible clips {len(ok_groups)}",
        f"clips per group (all rows): {describe(list(all_groups.values()))}",
        f"clips per group (eligible): {describe(list(ok_groups.values()))}",
        f"duration s (eligible): {describe(list(durations))}",
        f"groups with at least {per_group} eligible clips: {sum(1 for n in ok_groups.values() if n >= per_group)}",
    ]


def main(argv: Iterable[str] | None = None) -> int:
    """Run the tool; returns the exit code (see the module docstring)."""
    parser = argparse.ArgumentParser(description="Copy a small reproducible slice of an extracted dataset into a pack folder.")
    parser.add_argument("--source", type=Path, required=True, help="the extracted folder that holds manifest.csv and the audio")
    parser.add_argument("--group-column", required=True, help="the manifest column with the group (lesson) id")
    parser.add_argument("--groups", type=int, required=True, help="G: how many groups to take")
    parser.add_argument("--per-group", type=int, required=True, help="K: how many clips from each group")
    parser.add_argument("--seed", type=int, default=42, help="the seed of the stable hash (default 42)")
    parser.add_argument("--out", type=Path, required=True, help="the pack folder to create, for example data1")
    parser.add_argument("--rejected", type=Path, default=None, help="the rejected.csv of the full-data prepare (its ids are excluded)")
    parser.add_argument("--max-seconds", type=float, default=30.0, help="the longest clip allowed (default 30)")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="transcript")
    parser.add_argument("--date", default=None, help="written into pack_info.json (default: today)")
    parser.add_argument("--report-only", action="store_true", help="print the numbers and write nothing")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from stt.datasource.audio_header import read_duration_seconds  # the pipeline's own reader (soundfile)

    manifest = args.source / "manifest.csv"
    if not manifest.is_file():
        print(f"ERROR: {manifest} does not exist. Check --source.", file=sys.stderr)
        return 2
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    missing = [c for c in (args.id_column, args.audio_column, args.text_column, args.group_column) if c not in header]
    if missing:
        print(f"ERROR: the manifest has no column {missing}. Columns: {', '.join(header)}.", file=sys.stderr)
        return 2
    rejected = read_rejected(args.rejected)
    durations: dict[str, float] = {}

    def duration_of(path: Path) -> float:
        durations[str(path)] = read_duration_seconds(path)
        return durations[str(path)]

    eligible = [r for r in rows if is_eligible(r, args.source, args.audio_column, args.text_column, args.id_column, rejected, args.max_seconds, duration_of)]
    seconds = [durations[str(args.source / r[args.audio_column])] for r in eligible]
    for line in report(rows, eligible, args.group_column, seconds, args.per_group):
        print(line)
    if args.report_only:
        return 0
    try:
        chosen = select_clips(eligible, args.id_column, args.group_column, args.groups, args.per_group, args.seed)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if (args.out / "manifest.csv").exists():
        print(f"ERROR: {args.out / 'manifest.csv'} already exists; nothing is overwritten. Choose another --out or move the old folder away.", file=sys.stderr)
        return 3
    split_column = "source_split" if "source_split" in header else None
    info: dict[str, object] = {
        "source_manifest": str(manifest.as_posix()),
        "source_rows": len(rows),
        "eligible_rows": len(eligible),
        "seed": args.seed,
        "groups": args.groups,
        "per_group": args.per_group,
        "clips": len(chosen),
        "group_column": args.group_column,
        "max_seconds": args.max_seconds,
        "rejected_file": None if args.rejected is None else args.rejected.name,
        "source_split_counts": dict(sorted(Counter(r[split_column] for r in chosen).items())) if split_column else None,
        "created": args.date or date.today().isoformat(),
    }
    try:
        size = write_pack(chosen, header, args.source, args.out, args.id_column, args.audio_column, info)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"selected {len(chosen)} clips in {args.groups} groups -> {args.out} ({size / 1e6:.1f} MB)")
    if split_column:
        print(f"source_split of the chosen clips: {info['source_split_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
