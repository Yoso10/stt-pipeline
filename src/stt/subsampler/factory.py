"""resolve_subsample: the one call the Orchestrator needs (get-or-create a stored subset).

How to run: this module is not a script; the composition root calls it. Example (PowerShell, project root; `splits` maps split names to NormalizedSample lists):
    $env:PYTHONPATH = "src"
    python -c "from stt.subsampler import SubsampleConfig, build_spec; print(build_spec(SubsampleConfig(preset='tiny'), 42, 'train', 'val').size)"
Example output:
    SubsampleSize(kind='minutes', value=30.0)
`enabled: false` returns the input unchanged. Without `subsample.dir` nothing is saved. With `subsample.dir`: a stored file of the same spec is loaded
(FAILS when stale) and otherwise the subset is made and saved.
"""

from __future__ import annotations

from dataclasses import replace

from stt.subsampler.config import SubsampleConfig, build_spec
from stt.subsampler.errors import SubsampleError
from stt.subsampler.strategies import StrategyRegistry
from stt.subsampler.store import SubsampleStore
from stt.subsampler.subsampler import Splits, Subsampler
from stt.subsampler.types import SubsampleResult


def resolve_subsample(
    config: SubsampleConfig,
    runtime_seed: int,
    splits: Splits,
    train_split: str,
    val_split: str | None,
    registry: StrategyRegistry | None = None,
    store: SubsampleStore | None = None,
) -> SubsampleResult:
    """The subsampled splits for `config` (see the module docstring). Errors: SubsampleConfigError and SubsampleError."""
    spec = build_spec(config, runtime_seed, train_split, val_split)
    subsampler = Subsampler(registry)
    if not config.enabled:
        return subsampler.disabled(splits, spec)
    if config.dir is None:
        return subsampler.subsample(splits, spec)
    store = store or SubsampleStore()
    if store.exists(config.dir, spec):
        loaded = store.load(config.dir, spec, splits)
        fresh = subsampler.subsample(splits, spec)  # the report of the stored subset: made again from the same rule, then checked against the stored ids
        if {n: [s.sample.id for s in v] for n, v in loaded.items()} != {n: [s.sample.id for s in v] for n, v in fresh.splits.items()}:
            raise SubsampleError(f"The stored subset {store.path(config.dir, spec)} differs from what the rule selects now. Delete the file to make a new subset.")
        return replace(fresh, report=replace(fresh.report, ids_location=str(store.path(config.dir, spec))))
    result = subsampler.subsample(splits, spec)
    path = store.save(config.dir, spec, splits, {name: [n.sample.id for n in samples] for name, samples in result.splits.items()})
    return replace(result, report=replace(result.report, ids_location=str(path)))

