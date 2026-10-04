"""TimeBlockSplitter: for few long recordings; contiguous time blocks with a guard gap between splits.

How to run: this module is not a script; the composition root creates it through the registry.
Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.splitter import TimeBlockSplitter"
Method: inside every group (a recording) the clips are ordered by `split.time_block.order_column`
(default `meta_seek`, the start time in seconds); the ordered clips are cut into contiguous blocks, one
per split, by the ratios. Then every clip that starts less than `guard_gap_sec` after the end of the
previous block is removed from all splits and listed in `dropped_guard`. A group therefore appears in
several splits BY DESIGN (the hard check "no group in two splits" is replaced by the guard gap).
"""

from __future__ import annotations

from collections.abc import Sequence

from stt.datasource import AudioSample
from stt.splitter.base import Assignment, Group, GroupSplitter
from stt.splitter.errors import SplitError


class TimeBlockSplitter(GroupSplitter):
    """Contiguous time blocks per recording with a guard gap (see the module docstring)."""

    groups_may_span_splits = True

    def assign(self, groups: Sequence[Group]) -> Assignment:
        """Cut each group into blocks and drop the clips inside the guard gap. Errors: SplitError."""
        column = self.config.time_block.order_column
        gap = self.config.time_block.guard_gap_sec
        starts: dict[str, float] = {}
        missing: list[str] = []
        for group in groups:
            for sample in group.samples:
                value = sample.extra.get(column)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    missing.append(sample.id)
                else:
                    starts[sample.id] = float(value)
        if missing:
            raise SplitError(
                f"{len(missing)} sample(s) have no numeric '{column}' (for example {', '.join(missing[:5])}). "
                "Set split.time_block.order_column to a numeric field of the samples "
                "(and list it in data.extra_types as float) or use another split.mode."
            )
        assignment = Assignment()
        names = self.config.names
        for group in groups:
            ordered = sorted(group.samples, key=lambda s: (starts[s.id], s.id))
            blocks = self._cut(ordered, names)
            previous_end: float | None = None
            for name in names:
                kept: list[AudioSample] = []
                for sample in blocks[name]:
                    if previous_end is not None and starts[sample.id] < previous_end + gap:
                        assignment.dropped_guard.add(sample.id)
                    else:
                        kept.append(sample)
                        assignment.split_of[sample.id] = name
                if kept:
                    previous_end = max(starts[s.id] + s.duration_sec for s in kept)
        assignment.overlap_check = (
            f"time_block: groups span splits by design; blocks are separated by a guard gap of {gap} s "
            f"({len(assignment.dropped_guard)} sample(s) dropped)"
        )
        return assignment

    def _cut(self, ordered: Sequence[AudioSample], names: Sequence[str]) -> dict[str, list[AudioSample]]:
        total = sum(self.weight(s) for s in ordered) or 1.0
        edges: list[float] = []
        running = 0.0
        for ratio in self.config.ratios:
            running += ratio
            edges.append(running)
        blocks: dict[str, list[AudioSample]] = {n: [] for n in names}
        done = 0.0
        for sample in ordered:
            middle = (done + self.weight(sample) / 2) / total
            index = next((i for i, edge in enumerate(edges) if middle < edge), len(edges) - 1)
            blocks[names[index]].append(sample)
            done += self.weight(sample)
        return blocks
