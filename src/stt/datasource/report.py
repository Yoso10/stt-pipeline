"""Build the LoadReport and write rejected.csv.

How to run: this module is not a script; CsvAudioSource calls it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.datasource.report import distribution; print(distribution([1.0, 2.0, 3.0]))"
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from stt.datasource.types import AudioSample, Distribution, LoadReport, Rejection

REJECTED_FILE = "rejected.csv"


def distribution(values: Sequence[float]) -> Distribution | None:
    """Return min, max, mean and percentiles (5, 25, 50, 75, 95) of `values`, or None when empty."""
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    p05, p25, p50, p75, p95 = (float(x) for x in np.percentile(array, [5, 25, 50, 75, 95]))
    return Distribution(
        count=len(values),
        minimum=float(array.min()),
        maximum=float(array.max()),
        mean=float(array.mean()),
        p05=p05,
        p25=p25,
        p50=p50,
        p75=p75,
        p95=p95,
    )


def build_load_report(
    samples: Sequence[AudioSample],
    rejections: Sequence[Rejection],
    rejected_splits: Mapping[str, str | None],
    missing_extra_values: Mapping[str, int],
) -> LoadReport:
    """Summarize a load. `rejected_splits` maps a rejected id to its source split (or None)."""
    reasons = Counter(r.reason for r in rejections)
    loaded_per_split = Counter(s.source_split or "(none)" for s in samples)
    rejected_per_split = Counter(rejected_splits.get(r.id) or "(none)" for r in rejections)
    numeric_names = sorted(
        {
            name
            for s in samples
            for name, value in s.extra.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    extras = {}
    for name in numeric_names:
        found = distribution(
            [
                float(s.extra[name])  # type: ignore[arg-type]
                for s in samples
                if name in s.extra and isinstance(s.extra[name], (int, float)) and not isinstance(s.extra[name], bool)
            ]
        )
        if found is not None:
            extras[name] = found
    splits_of_group: dict[str, set[str]] = {}
    for s in samples:
        splits_of_group.setdefault(s.group_id, set()).add(s.source_split or "(none)")
    return LoadReport(
        seen=len(samples) + len(rejections),
        loaded=len(samples),
        rejected=len(rejections),
        rejection_reasons=dict(sorted(reasons.items())),
        loaded_per_split=dict(sorted(loaded_per_split.items())),
        rejected_per_split=dict(sorted(rejected_per_split.items())),
        duration=distribution([s.duration_sec for s in samples]),
        extra_distributions=extras,
        missing_extra_values=dict(sorted(missing_extra_values.items())),
        distinct_groups=len(splits_of_group),
        groups_in_several_splits=sum(1 for splits in splits_of_group.values() if len(splits) > 1),
    )


def write_rejected_csv(rejections: Sequence[Rejection], out_dir: Path) -> Path:
    """Write `rejected.csv` (id, reason, detail; UTF-8) into `out_dir` and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REJECTED_FILE
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "reason", "detail"])
        for r in rejections:
            writer.writerow([r.id, r.reason, r.detail])
    return path
