"""The `grid` strategy: every combination in a fixed order (own code, no dependency).

How to run: this module is not a script; the registry creates it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.searcher.grid import grid_points; from stt.searcher.types import ParamRange; print(grid_points(ParamRange('log_float', 1e-5, 1e-3, points=3)))"
Example output:
    [1e-05, 0.0001, 0.001]
The values of each parameter: `log_float` geometric, `float` evenly spaced (`points` of them, ends included), `int` every whole number from low to high, `categorical`
the choices in the order given. The combinations are the product in the order of the parameters (the last one varies fastest). A resumed study skips the
combinations that were already tried (finished or failed). When the combinations are used up the strategy returns None (`space_exhausted`).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping

from stt.searcher.types import ParamRange, ParamValue, SearchSpec, Suggestion, TrialOutcome, TrialRecord


def grid_points(r: ParamRange) -> list[ParamValue]:
    """The values of one range for the grid."""
    if r.kind == "categorical":
        return list(r.choices)
    low, high = float(r.low or 0.0), float(r.high or 0.0)
    if r.kind == "int":
        return list(range(int(low), int(high) + 1))
    if r.kind == "log_float":
        a, b = math.log(low), math.log(high)
        return [round(math.exp(a + (b - a) * i / (r.points - 1)), 12) for i in range(r.points)]
    return [round(low + (high - low) * i / (r.points - 1), 12) for i in range(r.points)]


def _key(params: Mapping[str, ParamValue]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((k, repr(float(v)) if isinstance(v, int | float) else str(v)) for k, v in params.items()))


class GridStrategy:
    """Enumerates the product of the grid values in a fixed order."""

    name = "grid"

    def __init__(self, spec: SearchSpec) -> None:
        names = list(spec.parameters)
        self._combos = [dict(zip(names, values, strict=True)) for values in itertools.product(*(grid_points(spec.parameters[n]) for n in names))]
        self._done: set[tuple[tuple[str, str], ...]] = set()

    def size(self) -> int:
        """The number of combinations."""
        return len(self._combos)

    def ask(self, trial_id: int) -> Suggestion | None:
        """The next combination that was not tried yet, or None."""
        for combo in self._combos:
            if _key(combo) not in self._done:
                self._done.add(_key(combo))  # reserved now: a combination is never handed out twice
                return Suggestion(dict(combo))
        return None

    def record_eval(self, trial_id: int, step: int, val_loss: float) -> None:
        """The grid ignores intermediate values."""

    def tell(self, trial_id: int, params: Mapping[str, ParamValue], outcome: TrialOutcome) -> None:
        """The grid learns nothing from outcomes."""

    def replay(self, records: list[TrialRecord]) -> None:
        """Mark the combinations of the earlier trials as done."""
        self._done.update(_key(r.trial.params) for r in records)
