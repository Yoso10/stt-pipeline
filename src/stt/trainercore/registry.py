"""Registries of the TrainerCore: freeze policies and metrics sinks, selected by name (`train.freeze_policy`, `train.sinks`).

How to run: this module is not a script; the composition root creates the registries. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.trainercore import default_freeze_registry, default_sink_registry; print(default_freeze_registry().names(), default_sink_registry().names())"
Example output:
    ['freeze_encoder', 'none'] ['console', 'csv']
A new policy or sink is a new class plus one entry here or in a registry built by the caller; no other code changes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType

from stt.trainercore.errors import TrainerConfigError
from stt.trainercore.freeze import FreezeEncoder, NoFreeze
from stt.trainercore.sinks import ConsoleMetricsSink, CsvMetricsSink
from stt.trainercore.types import FreezePolicy, MetricsSink, SinkContext

FreezeFactory = Callable[[], FreezePolicy]
SinkFactory = Callable[[SinkContext], MetricsSink]


class FreezeRegistry:
    """A read-only name -> freeze policy factory mapping."""

    def __init__(self, factories: Mapping[str, FreezeFactory]) -> None:
        self._factories: Mapping[str, FreezeFactory] = MappingProxyType(dict(factories))

    def names(self) -> list[str]:
        """The registered names, sorted."""
        return sorted(self._factories)

    def create(self, name: str) -> FreezePolicy:
        """Create the policy `name`. Errors: TrainerConfigError listing the registered names."""
        factory = self._factories.get(name)
        if factory is None:
            raise TrainerConfigError(f"Unknown train.freeze_policy '{name}'. Registered policies: {', '.join(self.names())}.")
        return factory()


class SinkRegistry:
    """A read-only name -> metrics sink factory mapping."""

    def __init__(self, factories: Mapping[str, SinkFactory]) -> None:
        self._factories: Mapping[str, SinkFactory] = MappingProxyType(dict(factories))

    def names(self) -> list[str]:
        """The registered names, sorted."""
        return sorted(self._factories)

    def require(self, names: Sequence[str]) -> None:
        """Fail unless every name is registered. Errors: TrainerConfigError listing the registered names."""
        for name in names:
            if name not in self._factories:
                raise TrainerConfigError(f"Unknown train.sinks entry '{name}'. Registered sinks: {', '.join(self.names())}.")

    def create(self, name: str, context: SinkContext) -> MetricsSink:
        """Create the sink `name`. Errors: TrainerConfigError listing the registered names."""
        factory = self._factories.get(name)
        if factory is None:
            raise TrainerConfigError(f"Unknown train.sinks entry '{name}'. Registered sinks: {', '.join(self.names())}.")
        return factory(context)


def default_freeze_registry() -> FreezeRegistry:
    """The registry with `none` and `freeze_encoder`."""
    return FreezeRegistry({"none": NoFreeze, "freeze_encoder": FreezeEncoder})


def default_sink_registry() -> SinkRegistry:
    """The registry with `csv` and `console`."""
    return SinkRegistry({"csv": CsvMetricsSink, "console": ConsoleMetricsSink})
