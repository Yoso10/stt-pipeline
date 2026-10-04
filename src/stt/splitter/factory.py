"""Select a Splitter by name (registry) and get-or-create a saved split.

How to run: this module is not a script; the composition root calls it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.splitter import default_registry; print(default_registry().names())"
Example output:
    ['grouped_ratio', 'hash_group', 'predefined', 'time_block']
`resolve_split(config, runtime_seed, samples)` is the one call the Orchestrator needs: it reuses the
saved split of `split.version` when there is one, otherwise it splits and saves it (when `split.dir` is set).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType

from stt.datasource import AudioSample
from stt.splitter.base import raise_unknown_mode
from stt.splitter.config import SplitConfig, resolve_seed
from stt.splitter.errors import SplitError
from stt.splitter.grouped_ratio import GroupedRatioSplitter
from stt.splitter.hash_group import HashGroupSplitter
from stt.splitter.predefined import PredefinedSplitter
from stt.splitter.store import CSV_FILE, SplitStore
from stt.splitter.time_block import TimeBlockSplitter
from stt.splitter.types import SplitResult, Splitter

SplitterFactory = Callable[[SplitConfig, int], Splitter]


class SplitterRegistry:
    """A read-only name -> factory mapping. `factory(config, seed)` returns a Splitter."""

    def __init__(self, factories: Mapping[str, SplitterFactory]) -> None:
        self._factories: Mapping[str, SplitterFactory] = MappingProxyType(dict(factories))

    def names(self) -> list[str]:
        """The registered mode names, sorted."""
        return sorted(self._factories)

    def create(self, config: SplitConfig, seed: int) -> Splitter:
        """Create the splitter named by `config.mode`. Errors: SplitError for an unknown name (the names are listed)."""
        factory = self._factories.get(config.mode)
        if factory is None:
            raise raise_unknown_mode(config.mode, self.names())
        return factory(config, seed)


def default_registry() -> SplitterRegistry:
    """The registry with the four built-in modes."""
    return SplitterRegistry(
        {
            "grouped_ratio": GroupedRatioSplitter,
            "hash_group": HashGroupSplitter,
            "predefined": PredefinedSplitter,
            "time_block": TimeBlockSplitter,
        }
    )


def resolve_split(
    config: SplitConfig,
    runtime_seed: int,
    samples: Sequence[AudioSample],
    registry: SplitterRegistry | None = None,
    store: SplitStore | None = None,
) -> SplitResult:
    """Return the split for `samples`: the saved one of `split.version` (reuse), or a new one.

    Without `split.dir` nothing is saved. With `split.dir`: if `<dir>/<version>/split.json` exists and
    `split.reuse` is true it is loaded (a data fingerprint mismatch FAILS: create a new version); if it
    exists and `reuse` is false it FAILS (a saved split is never overwritten); otherwise the new split is
    saved (and `split.csv` written when `split.export_csv` is true).
    """
    store = store or SplitStore()
    if config.dir is None:
        return (registry or default_registry()).create(config, resolve_seed(config, runtime_seed)).split(samples)
    directory = config.dir / config.version
    if store.exists(directory):
        if not config.reuse:
            raise SplitError(
                f"A split of version '{config.version}' already exists in {directory} and split.reuse is false. "
                "A saved split is never overwritten: set a new split.version to create a new one."
            )
        return store.load(directory, samples)
    result = (registry or default_registry()).create(config, resolve_seed(config, runtime_seed)).split(samples)
    store.save(result, directory)
    if config.export_csv:
        store.export_csv(result, directory / CSV_FILE)
    return result
