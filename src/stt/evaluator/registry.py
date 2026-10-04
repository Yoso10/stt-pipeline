"""Registries of the Evaluator: metrics and significance tests, selected by name (`eval.metrics`, `eval.significance.method`).

How to run: this module is not a script; the composition root creates the registries. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.evaluator import default_metric_registry, default_significance_registry; print(default_metric_registry().names(), default_significance_registry().names())"
Example output:
    ['cer', 'wer'] ['paired_group_bootstrap']
A new metric or test is a new class plus one entry here (or in a registry the caller builds); no other code changes. A second significance
method that needs a new library (for example Wilcoxon with scipy) needs the owner's approval first.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from stt.evaluator.bootstrap import PairedGroupBootstrap
from stt.evaluator.config import SignificanceConfig
from stt.evaluator.errors import EvaluatorConfigError
from stt.evaluator.metrics import CharErrorRate, WordErrorRate
from stt.evaluator.types import Metric, SignificanceTest

SignificanceFactory = Callable[[SignificanceConfig], SignificanceTest]


class MetricRegistry:
    """A read-only name -> metric class mapping."""

    def __init__(self, metrics: Mapping[str, Callable[[], Metric]]) -> None:
        self._metrics: Mapping[str, Callable[[], Metric]] = MappingProxyType(dict(metrics))

    def names(self) -> list[str]:
        """The registered names, sorted."""
        return sorted(self._metrics)

    def create(self, name: str) -> Metric:
        """Create the metric `name`. Errors: EvaluatorConfigError listing the registered names."""
        factory = self._metrics.get(name)
        if factory is None:
            raise EvaluatorConfigError(f"Unknown eval.metrics entry '{name}'. Registered metrics: {', '.join(self.names())}.")
        return factory()


class SignificanceRegistry:
    """A read-only name -> significance test factory mapping."""

    def __init__(self, factories: Mapping[str, SignificanceFactory]) -> None:
        self._factories: Mapping[str, SignificanceFactory] = MappingProxyType(dict(factories))

    def names(self) -> list[str]:
        """The registered names, sorted."""
        return sorted(self._factories)

    def create(self, config: SignificanceConfig) -> SignificanceTest:
        """Create the test named by `config.method`. Errors: EvaluatorConfigError listing the registered names."""
        factory = self._factories.get(config.method)
        if factory is None:
            raise EvaluatorConfigError(f"Unknown eval.significance.method '{config.method}'. Registered methods: {', '.join(self.names())}.")
        return factory(config)


def default_metric_registry() -> MetricRegistry:
    """The registry with `wer` and `cer`."""
    return MetricRegistry({"wer": WordErrorRate, "cer": CharErrorRate})


def default_significance_registry() -> SignificanceRegistry:
    """The registry with `paired_group_bootstrap`."""
    return SignificanceRegistry({"paired_group_bootstrap": lambda c: PairedGroupBootstrap(c.confidence, c.one_sided, c.n_resamples)})
