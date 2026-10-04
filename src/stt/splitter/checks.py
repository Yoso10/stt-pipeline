"""The HARD CHECKS and the minimum gates every split must pass (rule 1 of the Splitter brief).

How to run: this module is not a script; every splitter calls `verify_result` after splitting.
Example (PowerShell, project root; `result`, `samples`, `keys` come from a splitter):
    $env:PYTHONPATH = "src"
    python -c "from stt.splitter.checks import verify_result"
A violation raises SplitError with the group counts and the options for a fix.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from stt.datasource import AudioSample
from stt.splitter.config import SplitConfig
from stt.splitter.errors import SplitError
from stt.splitter.types import SplitResult


def check_input(samples: Sequence[AudioSample]) -> None:
    """Fail on empty input or duplicate ids (naming the first duplicates)."""
    if not samples:
        raise SplitError("There are no samples to split. Check that the DataSource loaded data.")
    counts = Counter(s.id for s in samples)
    duplicates = [i for i, c in counts.items() if c > 1]
    if duplicates:
        raise SplitError(f"Duplicate sample ids: {', '.join(sorted(duplicates)[:5])}. Ids must be unique (check data.columns.id).")


def verify_result(
    result: SplitResult,
    inputs: Sequence[AudioSample],
    keys: Mapping[str, str],
    config: SplitConfig | None,
    groups_may_span_splits: bool = False,
) -> None:
    """Run the hard checks and the minimum gates on `result`.

    Hard checks: no group in two splits (skipped for time_block, where blocks of one recording
    are separated by a guard gap instead); the union of the splits plus the dropped samples equals
    the input and no sample is listed twice; no duplicate ids; no empty split.
    Gates: `min_samples_per_split` and `min_groups_per_split` (skipped when `config` is None, as when a
    saved split is loaded).
    Errors: SplitError with the numbers and the options to fix it.
    """
    seen: Counter[str] = Counter()
    for group in result.splits.values():
        seen.update(s.id for s in group)
    seen.update(s.id for s in result.dropped_guard)
    seen.update(s.id for s in result.dropped_overlap)
    if any(c > 1 for c in seen.values()):
        raise SplitError("Hard check failed: a sample appears more than once in the result.")
    if set(seen) != {s.id for s in inputs}:
        raise SplitError("Hard check failed: the splits plus the dropped samples do not equal the input.")
    if not groups_may_span_splits:
        where: dict[str, set[str]] = {}
        for name, group in result.splits.items():
            for s in group:
                where.setdefault(keys[s.id], set()).add(name)
        leaked = sorted(g for g, names in where.items() if len(names) > 1)
        if leaked:
            raise SplitError(
                f"Hard check failed: {len(leaked)} group(s) appear in more than one split, for example {', '.join(leaked[:3])}. "
                "This would leak content between Train and Test."
            )
    for stat in result.report.stats:
        if stat.samples == 0:
            raise SplitError(
                f"Split '{stat.name}' is empty. There are {result.report.total_groups} groups for {len(result.splits)} splits. "
                "Use a finer group key (split.group_key) or another mode, or fewer splits."
            )
    if config is None:
        return
    problems = []
    for stat in result.report.stats:
        if stat.samples < config.min_samples_per_split:
            problems.append(f"'{stat.name}' has {stat.samples} samples (minimum {config.min_samples_per_split})")
        if stat.groups < config.min_groups_per_split:
            problems.append(f"'{stat.name}' has {stat.groups} groups (minimum {config.min_groups_per_split})")
    if problems:
        raise SplitError(
            f"Minimum gate failed: {'; '.join(problems)}. Total {result.report.total_groups} groups. "
            "Use a finer group key (for example one lesson or recording) or the time_block mode; "
            "do not just lower the minimums."
        )
