"""stt.datasource: load labeled audio + text from a local folder into AudioSample objects (C1 DataSource).

How to run: from Python (PowerShell, project root; the working example is in docs/COMPONENTS.md, C1):
    $env:PYTHONPATH = "src"
    python -c "from stt.datasource import default_registry; print(default_registry().names())"
Example output:
    ['csv', 'hf_parquet']
"""

from stt.datasource.config import ColumnsConfig, DataConfig, ExtractConfig, SourceConfig
from stt.datasource.csv_source import CsvAudioSource
from stt.datasource.errors import DataSourceError
from stt.datasource.factory import DataSourceRegistry, create_data_source, default_registry
from stt.datasource.parquet_source import HfParquetSource
from stt.datasource.types import AudioSample, DataSource, Distribution, LoadReport, LoadResult, Rejection

__all__ = [
    "AudioSample",
    "ColumnsConfig",
    "CsvAudioSource",
    "DataConfig",
    "DataSource",
    "DataSourceError",
    "DataSourceRegistry",
    "Distribution",
    "ExtractConfig",
    "HfParquetSource",
    "LoadReport",
    "LoadResult",
    "Rejection",
    "SourceConfig",
    "create_data_source",
    "default_registry",
]
