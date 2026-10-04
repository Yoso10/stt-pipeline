"""Select a DataSource by name (registry) without if/else on type names.

How to run: this module is not a script; the composition root calls `create_data_source`. Example
(PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.datasource import default_registry; print(sorted(default_registry().names()))"
Example output:
    ['csv', 'hf_parquet']
A new source type is added by registering a factory; no existing code changes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType

from stt.datasource.config import DataConfig
from stt.datasource.csv_source import CsvAudioSource
from stt.datasource.errors import DataSourceError
from stt.datasource.parquet_source import HfParquetSource
from stt.datasource.types import DataSource

SourceFactory = Callable[[DataConfig, Path], DataSource]


class DataSourceRegistry:
    """A read-only name -> factory mapping. `factory(config, out_dir)` returns a DataSource."""

    def __init__(self, factories: Mapping[str, SourceFactory]) -> None:
        self._factories: Mapping[str, SourceFactory] = MappingProxyType(dict(factories))

    def names(self) -> list[str]:
        """The registered source names, sorted."""
        return sorted(self._factories)

    def create(self, config: DataConfig, out_dir: Path) -> DataSource:
        """Create the source named by `config.source.type`. Errors: DataSourceError for an unknown name."""
        name = config.source.type
        factory = self._factories.get(name)
        if factory is None:
            raise DataSourceError(
                f"Unknown data.source.type '{name}'. Registered source types: {', '.join(self.names())}."
            )
        return factory(config, out_dir)


def default_registry() -> DataSourceRegistry:
    """The registry with the two built-in sources: `csv` and `hf_parquet`."""
    return DataSourceRegistry(
        {
            "csv": lambda config, out_dir: CsvAudioSource(config, out_dir),
            "hf_parquet": lambda config, out_dir: HfParquetSource(config, out_dir),
        }
    )


def create_data_source(config: DataConfig, out_dir: Path, registry: DataSourceRegistry | None = None) -> DataSource:
    """Create the DataSource selected by `data.source.type` (default registry unless one is given)."""
    return (registry or default_registry()).create(config, out_dir)
