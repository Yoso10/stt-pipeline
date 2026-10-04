"""Strategies (the unit of selection) and their registry.

How to run: this module is not a script; the Subsampler uses it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.subsampler import default_registry; print(default_registry().names())"
Example output:
    ['by_group', 'by_sample']
`by_sample`: the unit is the sample (its id). `by_group`: the unit is the whole group (for example a whole lesson), so a group is never split.
A new strategy is a new class plus one entry in `default_registry`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from stt.subsampler.errors import SubsampleConfigError
from stt.subsampler.types import SubsampleStrategy
from stt.textnormalizer import NormalizedSample


class BySample:
    """The unit is the single sample."""

    def key(self, sample: NormalizedSample) -> str:
        """The sample id."""
        return sample.sample.id


class ByGroup:
    """The unit is the whole group."""

    def key(self, sample: NormalizedSample) -> str:
        """The group id."""
        return sample.sample.group_id


class StrategyRegistry:
    """A read-only name -> strategy factory mapping."""

    def __init__(self, factories: Mapping[str, Callable[[], SubsampleStrategy]]) -> None:
        self._factories: Mapping[str, Callable[[], SubsampleStrategy]] = MappingProxyType(dict(factories))

    def names(self) -> list[str]:
        """The registered names, sorted."""
        return sorted(self._factories)

    def create(self, name: str) -> SubsampleStrategy:
        """Create the strategy `name`. Errors: SubsampleConfigError listing the registered names."""
        factory = self._factories.get(name)
        if factory is None:
            raise SubsampleConfigError(f"Unknown subsample.strategy '{name}'. Registered strategies: {', '.join(self.names())}.")
        return factory()


def default_registry() -> StrategyRegistry:
    """The registry with `by_sample` and `by_group`."""
    return StrategyRegistry({"by_sample": BySample, "by_group": ByGroup})
