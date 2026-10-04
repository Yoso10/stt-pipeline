"""HashGroupSplitter: stable assignment of a group by sha256(seed|group); new groups never move old ones.

How to run: this module is not a script; the composition root creates it through the registry.
Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.splitter import HashGroupSplitter"
Method: each group gets a number in [0, 1) from its hash; the cumulative ratios cut that range into the
splits. A group's split depends only on the seed, the ratios and the group itself, so data that grows
over time keeps every old group where it was. With few groups the shares can be uneven: the minimum
gates then FAIL instead of continuing silently (the ratio tolerance is only reported for this mode).
"""

from __future__ import annotations

from collections.abc import Sequence

from stt.splitter.base import Assignment, Group, GroupSplitter
from stt.splitter.keys import unit_hash


class HashGroupSplitter(GroupSplitter):
    """Stable hash-based assignment of whole groups (see the module docstring)."""

    def assign(self, groups: Sequence[Group]) -> Assignment:
        """Assign each group by its hash; notes the deviation from the requested ratios."""
        assignment = Assignment()
        self.check_oversized(groups, assignment)
        edges: list[float] = []
        running = 0.0
        for ratio in self.config.ratios:
            running += ratio
            edges.append(running)
        totals = dict.fromkeys(self.config.names, 0.0)
        for group in groups:
            u = unit_hash(self.seed, group.key)
            index = next((i for i, edge in enumerate(edges) if u < edge), len(edges) - 1)
            name = self.config.names[index]
            totals[name] += group.weight
            for sample in group.samples:
                assignment.split_of[sample.id] = name
        whole = sum(totals.values()) or 1.0
        for name, ratio in zip(self.config.names, self.config.ratios):
            actual = totals[name] / whole
            if abs(actual - ratio) > self.config.ratio_tolerance:
                assignment.notes.append(
                    f"WARNING: hash_group split '{name}' has ratio {actual:.3f} vs requested {ratio:.3f} "
                    f"(outside the tolerance {self.config.ratio_tolerance}; normal with few groups)."
                )
        return assignment
