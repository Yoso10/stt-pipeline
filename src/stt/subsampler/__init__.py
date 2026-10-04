"""stt.subsampler: select a smaller, deterministic, nested subset of the samples (C3b Subsampler).

How to run: from Python (PowerShell, project root; the working example is in docs/COMPONENTS.md, C3b):
    $env:PYTHONPATH = "src"
    python -c "from stt.subsampler import default_registry; print(default_registry().names())"
Example output:
    ['by_group', 'by_sample']
Importing this package imports no torch, no transformers and no audio library.
"""

from stt.subsampler.config import BUILT_IN_PRESETS, PresetConfig, SizeConfig, SubsampleConfig, build_spec, preset_names
from stt.subsampler.errors import SubsampleConfigError, SubsampleError
from stt.subsampler.factory import resolve_subsample
from stt.subsampler.store import SubsampleStore
from stt.subsampler.strategies import ByGroup, BySample, StrategyRegistry, default_registry
from stt.subsampler.subsampler import Subsampler, data_fingerprint, rank
from stt.subsampler.types import (
    SplitSizes,
    SubsampleReport,
    SubsampleResult,
    SubsampleSize,
    SubsampleSpec,
    SubsampleStrategy,
)

__all__ = [
    "BUILT_IN_PRESETS",
    "ByGroup",
    "BySample",
    "PresetConfig",
    "SizeConfig",
    "SplitSizes",
    "StrategyRegistry",
    "SubsampleConfig",
    "SubsampleConfigError",
    "SubsampleError",
    "SubsampleReport",
    "SubsampleResult",
    "SubsampleSize",
    "SubsampleSpec",
    "SubsampleStore",
    "SubsampleStrategy",
    "Subsampler",
    "build_spec",
    "data_fingerprint",
    "default_registry",
    "preset_names",
    "rank",
    "resolve_subsample",
]
