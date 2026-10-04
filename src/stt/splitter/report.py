"""Fingerprints and the SplitReport/SplitResult assembly (no splitting logic, no file access).

How to run: this module is not a script; the splitters and the SplitStore call it. Example
(PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.splitter.report import data_fingerprint; print(data_fingerprint([]))"
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence

from stt.datasource import AudioSample
from stt.splitter.keys import field_value
from stt.splitter.types import GroupShare, SplitParams, SplitReport, SplitResult, SplitStats

TOP_GROUPS = 5


def data_fingerprint(samples: Sequence[AudioSample]) -> str:
    """sha256 over the sorted lines `id|group_id|duration` of the input samples."""
    lines = sorted(f"{s.id}|{s.group_id}|{s.duration_sec!r}" for s in samples)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def split_fingerprint(data_fp: str, assignments: Mapping[str, str]) -> str:
    """sha256 over the data fingerprint and the sorted lines `id|split` (dropped ids use the split `~dropped`)."""
    lines = sorted(f"{i}|{name}" for i, name in assignments.items())
    return hashlib.sha256((data_fp + "\n" + "\n".join(lines)).encode("utf-8")).hexdigest()


def _weight(sample: AudioSample, unit: str) -> float:
    return sample.duration_sec if unit == "duration" else 1.0


def build_result(
    inputs: Sequence[AudioSample],
    split_of: Mapping[str, str],
    dropped_guard: set[str],
    dropped_overlap: set[str],
    params: SplitParams,
    keys: Mapping[str, str],
    notes: Sequence[str] = (),
    overlap_check: str = "ok: no group appears in more than one split",
) -> SplitResult:
    """Assemble a SplitResult (lists sorted by id) and its report from an assignment.

    `split_of` maps a kept sample id to its split name. `keys` maps sample id to group key.
    """
    ordered = sorted(inputs, key=lambda s: s.id)
    splits: dict[str, list[AudioSample]] = {name: [] for name in params.names}
    guard: list[AudioSample] = []
    overlap: list[AudioSample] = []
    for sample in ordered:
        if sample.id in dropped_guard:
            guard.append(sample)
        elif sample.id in dropped_overlap:
            overlap.append(sample)
        else:
            splits[split_of[sample.id]].append(sample)
    data_fp = data_fingerprint(ordered)
    assignments = {s.id: name for name, group in splits.items() for s in group}
    assignments.update({s.id: "~dropped_guard" for s in guard})
    assignments.update({s.id: "~dropped_overlap" for s in overlap})
    fingerprint = split_fingerprint(data_fp, assignments)

    kept_weight = sum(_weight(s, params.ratio_unit) for g in splits.values() for s in g) or 1.0
    stats = tuple(
        SplitStats(
            name=name,
            samples=len(group),
            groups=len({keys[s.id] for s in group}),
            duration_sec=sum(s.duration_sec for s in group),
            requested_ratio=ratio,
            actual_ratio=sum(_weight(s, params.ratio_unit) for s in group) / kept_weight,
        )
        for (name, group), ratio in zip(splits.items(), params.ratios)
    )
    total_weight = sum(_weight(s, params.ratio_unit) for s in ordered) or 1.0
    per_group: dict[str, list[AudioSample]] = {}
    for s in ordered:
        per_group.setdefault(keys[s.id], []).append(s)
    shares = sorted(
        (
            GroupShare(g, len(members), sum(m.duration_sec for m in members), sum(_weight(m, params.ratio_unit) for m in members) / total_weight)
            for g, members in per_group.items()
        ),
        key=lambda x: (-x.share, x.group),
    )
    table: dict[str, Counter[str]] = {}
    for name, group in splits.items():
        for s in group:
            table.setdefault(s.source_split or "(none)", Counter())[name] += 1
    report = SplitReport(
        params=params,
        total_samples=len(ordered),
        total_groups=len(per_group),
        stats=stats,
        largest_groups=tuple(shares[:TOP_GROUPS]),
        largest_group_share=shares[0].share if shares else 0.0,
        group_overlap_check=overlap_check,
        source_split_table={k: dict(sorted(v.items())) for k, v in sorted(table.items())},
        speaker_summary=_speaker_summary(splits, params.speaker_column),
        dropped_guard=len(guard),
        dropped_overlap=len(overlap),
        data_fingerprint=data_fp,
        fingerprint=fingerprint,
        notes=tuple(notes),
    )
    return SplitResult(splits=splits, dropped_guard=guard, dropped_overlap=overlap, report=report, fingerprint=fingerprint, data_fingerprint=data_fp)


def _speaker_summary(splits: Mapping[str, Sequence[AudioSample]], column: str | None) -> str:
    if column is None:
        return (
            "speakers: not checked (set split.speaker_column to a speaker or creator field). Splits are made by group only, "
            "so the Test may share speakers with Train."
        )
    per_split = {name: {field_value(s, column) for s in group} - {None} for name, group in splits.items()}
    everyone = set().union(*per_split.values()) if per_split else set()
    shared = {x for x in everyone if sum(x in v for v in per_split.values()) > 1}
    text = f"speakers ({column}): {len(everyone)} distinct; {len(shared)} appear in more than one split; " + ", ".join(
        f"{name} {len(v)}" for name, v in per_split.items()
    )
    if shared:
        text += ". The Test does NOT measure generalization to unseen speakers (speakers are shared with Train)."
    return text
