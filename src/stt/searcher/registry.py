"""Select a SearchStrategy by name (registry); the Optuna adapter is imported only when it is used.

How to run: this module is not a script; the searcher calls it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.searcher import default_registry; print(default_registry().names())"
Example output:
    ['grid', 'optuna_tpe']
Importing this module does not import Optuna (the adapter is imported inside the factory). When Optuna is not installed, choosing `optuna_tpe` fails with a
message that names the package and the alternative `grid`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from stt.searcher.errors import SearchConfigError
from stt.searcher.grid import GridStrategy
from stt.searcher.types import SearchSpec, SearchStrategy

StrategyFactory = Callable[[SearchSpec], SearchStrategy]


def _optuna(spec: SearchSpec) -> SearchStrategy:
    try:
        from stt.searcher.optuna_strategy import OptunaTpeStrategy  # imported only when this strategy is chosen
    except ImportError as exc:
        raise SearchConfigError(
            "search.strategy 'optuna_tpe' needs the package optuna, which is not installed in this environment. "
            "Install it in the virtual environment (from the offline wheels), or set search.strategy to 'grid' (no dependency)."
        ) from exc
    return OptunaTpeStrategy(spec)


class StrategyRegistry:
    """A read-only name -> strategy factory mapping."""

    def __init__(self, factories: Mapping[str, StrategyFactory]) -> None:
        self._factories: Mapping[str, StrategyFactory] = MappingProxyType(dict(factories))

    def names(self) -> list[str]:
        """The registered names, sorted."""
        return sorted(self._factories)

    def create(self, spec: SearchSpec) -> SearchStrategy:
        """Create the strategy `spec.strategy`. Errors: SearchConfigError listing the registered names (or the missing package)."""
        factory = self._factories.get(spec.strategy)
        if factory is None:
            raise SearchConfigError(f"Unknown search.strategy '{spec.strategy}'. Registered strategies: {', '.join(self.names())}.")
        return factory(spec)


def default_registry() -> StrategyRegistry:
    """The registry with `grid` and `optuna_tpe`."""
    return StrategyRegistry({"grid": GridStrategy, "optuna_tpe": _optuna})
