"""Rank correlation and deterministic thinning (pure functions, standard library only).

How to run: this module is not a script; the report builder uses it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.reportbuilder.correlation import spearman; print(spearman([1, 2, 3, 4], [1, 3, 2, 4]))"
Example output:
    0.8
`spearman` is Spearman's rank correlation: the Pearson correlation of the average ranks (so ties are handled the usual way). It is None for fewer than two points or when one of the two series is constant
(a constant series has no ranks to correlate). A correlation is not a cause: the report always says so next to the numbers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeVar

from stt.reportbuilder.types import CorrelationRow

T = TypeVar("T")


def ranks(values: Sequence[float]) -> list[float]:
    """The ranks (1 = smallest) of the values; equal values share the average of their ranks."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            result[order[k]] = average
        i = j + 1
    return result


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman's rank correlation of two series of the same length, or None (fewer than 2 points, or a constant series)."""
    if len(x) != len(y):
        raise ValueError(f"The two series have different lengths ({len(x)} and {len(y)}).")
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy)


def correlation_row(attribute: str, x: Sequence[float], y: Sequence[float], min_points: int) -> CorrelationRow:
    """The row of one attribute: pairs with a missing or non-finite value are dropped first; `n` is the number of pairs used.

    Fewer than `min_points` pairs: `too few points`, no number. A constant series: `constant`, no number.
    """
    pairs = [(a, b) for a, b in zip(x, y, strict=True) if math.isfinite(a) and math.isfinite(b)]
    n = len(pairs)
    if n < min_points:
        return CorrelationRow(attribute, n, None, f"too few points ({n} < {min_points})")
    xs, ys = [a for a, _ in pairs], [b for _, b in pairs]
    if len(set(xs)) < 2:
        return CorrelationRow(attribute, n, None, "constant (the attribute has one value)")
    if len(set(ys)) < 2:
        return CorrelationRow(attribute, n, None, "constant (the WER has one value)")
    return CorrelationRow(attribute, n, spearman(xs, ys), "")


def thin(items: Sequence[T], cap: int) -> tuple[list[T], bool]:
    """At most `cap` items, chosen by position at equal steps (the first and the last are kept): the same input always gives the same output. Returns (items, was_thinned)."""
    if len(items) <= cap:
        return list(items), False
    if cap < 2:
        return [items[0]], True
    picked = sorted({round(i * (len(items) - 1) / (cap - 1)) for i in range(cap)})
    return [items[i] for i in picked], True
