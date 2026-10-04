"""GroupedRatioSplitter: whole groups assigned to splits so that the shares match the requested ratios.

How to run: this module is not a script; the composition root creates it through the registry.
Example (PowerShell, project root; `config` is a SplitConfig, `samples` a list of AudioSample):
    $env:PYTHONPATH = "src"
    python -c "from stt.splitter import GroupedRatioSplitter"
Method: the groups are put in a seeded, stable order (sha256 of seed and group); each group goes to
the split that is furthest below its target. Then the deviation from every requested ratio must be
within `split.ratio_tolerance`, otherwise the split FAILS. A group that is too large FAILS
(`split.on_oversized_group`).
"""

from __future__ import annotations

from collections.abc import Sequence

from stt.splitter.base import Assignment, Group, GroupSplitter
from stt.splitter.errors import SplitError
from stt.splitter.keys import order_hash


def assign_by_deficit(groups: Sequence[Group], names: Sequence[str], ratios: Sequence[float], seed: int) -> dict[str, str]:
    """Assign every group to a name: seeded order, each group to the split furthest below its target.

    Returns {group key: name}. Ties go to the earlier name.
    """
    total = sum(g.weight for g in groups)
    targets = [r * total for r in ratios]
    current = [0.0] * len(names)
    chosen: dict[str, str] = {}
    for group in sorted(groups, key=lambda g: (order_hash(seed, g.key), g.key)):
        index = max(range(len(names)), key=lambda i: (targets[i] - current[i], -i))
        chosen[group.key] = names[index]
        current[index] += group.weight
    return chosen


class GroupedRatioSplitter(GroupSplitter):
    """Exact, balanced assignment of whole groups by the requested ratios (see the module docstring)."""

    def assign(self, groups: Sequence[Group]) -> Assignment:
        """Assign the groups; fails when a group is oversized or a ratio is missed by more than the tolerance."""
        assignment = Assignment()
        self.check_oversized(groups, assignment)
        chosen = assign_by_deficit(groups, self.config.names, self.config.ratios, self.seed)
        totals = dict.fromkeys(self.config.names, 0.0)
        for group in groups:
            totals[chosen[group.key]] += group.weight
            for sample in group.samples:
                assignment.split_of[sample.id] = chosen[group.key]
        whole = sum(totals.values()) or 1.0
        tolerance = self.config.ratio_tolerance
        misses = [
            f"'{name}' {totals[name] / whole:.3f} vs requested {ratio:.3f}"
            for name, ratio in zip(self.config.names, self.config.ratios)
            if abs(totals[name] / whole - ratio) > tolerance
        ]
        if misses:
            raise SplitError(
                f"The ratios cannot be met within split.ratio_tolerance {tolerance}: {'; '.join(misses)} "
                f"({len(groups)} groups). Use a finer group key, or raise the tolerance only consciously."
            )
        return assignment
