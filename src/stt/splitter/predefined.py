"""PredefinedSplitter: uses the split that came with the data (`AudioSample.source_split`).

How to run: this module is not a script; the composition root creates it through the registry.
Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.splitter import PredefinedSplitter"
Rules: each source split name is mapped to a split name (`split.predefined.source_map`, default: the
same name). A name in `split.names` that no sample uses (for example `val`) is carved out of the FIRST
name's groups, by group, in proportion to the ratios. A group present in several source splits FAILS
(`split.on_overlap: fail`) or is removed from the first split (`drop_from_train`; the removed samples are
listed in the result and the report).
"""

from __future__ import annotations

from collections.abc import Sequence

from stt.splitter.base import Assignment, Group, GroupSplitter
from stt.splitter.errors import SplitError
from stt.splitter.grouped_ratio import assign_by_deficit


class PredefinedSplitter(GroupSplitter):
    """Use `source_split` as the split (see the module docstring). Errors: SplitError."""

    def assign(self, groups: Sequence[Group]) -> Assignment:
        """Map source splits to split names, resolve overlaps, and carve missing splits from the first one."""
        names = self.config.names
        mapping = self.config.predefined.source_map
        assignment = Assignment()
        sample_split: dict[str, str] = {}
        unmapped: set[str] = set()
        for group in groups:
            for sample in group.samples:
                if sample.source_split is None:
                    raise SplitError(
                        f"Sample {sample.id} has no source_split, so mode 'predefined' cannot be used. "
                        "Map data.columns.split in the DataSource or use another split.mode."
                    )
                target = mapping.get(sample.source_split, sample.source_split)
                if target not in names:
                    unmapped.add(sample.source_split)
                sample_split[sample.id] = target
        if unmapped:
            raise SplitError(
                f"Source split name(s) {sorted(unmapped)} are not in split.names {list(names)}. "
                "Add them to split.predefined.source_map (for example {eval: test}) or to split.names."
            )
        first = names[0]
        group_targets: dict[str, set[str]] = {g.key: {sample_split[s.id] for s in g.samples} for g in groups}
        overlapping = sorted(k for k, t in group_targets.items() if len(t) > 1)
        if overlapping:
            fixable = all(first in group_targets[k] for k in overlapping)
            if self.config.on_overlap == "fail" or not fixable:
                reason = "" if fixable else " (they are also in several splits other than the first, which cannot be fixed by dropping)"
                raise SplitError(
                    f"{len(overlapping)} group(s) appear in more than one source split{reason}, for example "
                    f"{', '.join(overlapping[:3])}. Set split.on_overlap to 'drop_from_train' to remove them from '{first}', "
                    "or split by group with split.mode grouped_ratio."
                )
            for group in groups:
                if group.key in overlapping:
                    for sample in group.samples:
                        if sample_split[sample.id] == first:
                            assignment.dropped_overlap.add(sample.id)
            assignment.overlap_check = f"{len(overlapping)} overlapping group(s): their samples were removed from '{first}' (listed in dropped_overlap)"
        used = {sample_split[s.id] for g in groups for s in g.samples if s.id not in assignment.dropped_overlap}
        missing = [n for n in names if n not in used]
        carve_names = [first, *missing] if missing else []
        carved: dict[str, str] = {}
        if missing:
            first_groups = [g for g in groups if group_targets[g.key] == {first}]
            ratios = [r for n, r in zip(names, self.config.ratios) if n in carve_names]
            order = [n for n in names if n in carve_names]
            scale = sum(ratios)
            carved = assign_by_deficit(first_groups, order, [r / scale for r in ratios], self.seed)
            assignment.notes.append(f"{', '.join(missing)} carved out of '{first}' by group (no source samples for them).")
        for group in groups:
            for sample in group.samples:
                if sample.id in assignment.dropped_overlap:
                    continue
                name = sample_split[sample.id]
                if name == first and group.key in carved:
                    name = carved[group.key]
                assignment.split_of[sample.id] = name
        return assignment
