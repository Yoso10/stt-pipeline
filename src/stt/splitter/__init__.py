"""stt.splitter: divide samples into named splits BY GROUP, deterministically (C3 Splitter).

How to run: from Python (PowerShell, project root; the working example is in docs/COMPONENTS.md, C3):
    $env:PYTHONPATH = "src"
    python -c "from stt.splitter import default_registry; print(default_registry().names())"
Example output:
    ['grouped_ratio', 'hash_group', 'predefined', 'time_block']
"""

from stt.splitter.config import SplitConfig, resolve_seed
from stt.splitter.errors import SplitError
from stt.splitter.factory import SplitterRegistry, default_registry, resolve_split
from stt.splitter.grouped_ratio import GroupedRatioSplitter
from stt.splitter.hash_group import HashGroupSplitter
from stt.splitter.predefined import PredefinedSplitter
from stt.splitter.store import SplitStore
from stt.splitter.time_block import TimeBlockSplitter
from stt.splitter.types import GroupShare, SplitParams, SplitReport, SplitResult, SplitStats, Splitter

__all__ = [
    "GroupShare",
    "GroupedRatioSplitter",
    "HashGroupSplitter",
    "PredefinedSplitter",
    "SplitConfig",
    "SplitError",
    "SplitParams",
    "SplitReport",
    "SplitResult",
    "SplitStats",
    "SplitStore",
    "Splitter",
    "SplitterRegistry",
    "TimeBlockSplitter",
    "default_registry",
    "resolve_seed",
    "resolve_split",
]
