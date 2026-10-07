"""stt.modelexporter: convert a registered model into a target format, verify it, and record the export (C11 ModelExporter).

How to run: from Python (PowerShell, project root; the working example is in docs/COMPONENTS.md, C11):
    $env:PYTHONPATH = "src"
    python -c "from stt.modelexporter import default_exporter_registry; print(default_exporter_registry().names())"
Example output:
    ['hf_safetensors']
Importing this package imports no torch and no transformers (the `hf_safetensors` adapter imports the ModelLoader, which loads them only when a model is loaded).
"""

from stt.modelexporter.compare import wer_difference
from stt.modelexporter.config import ExportConfig
from stt.modelexporter.errors import ExportConfigError, ExportDependencyError, ExportError
from stt.modelexporter.registry import ExporterRegistry, default_exporter_registry, missing_packages, require_installed
from stt.modelexporter.service import EXPORTS_FILE, EXPORTS_FOLDER, EntryAccess, ModelExportService, append_record, read_exports
from stt.modelexporter.types import ExportRecord, ExportResult, ExportStatus, ModelExporter, VerifyInput, VerifyLevel, VerifyResult

__all__ = [
    "EXPORTS_FILE",
    "EXPORTS_FOLDER",
    "EntryAccess",
    "ExportConfig",
    "ExportConfigError",
    "ExportDependencyError",
    "ExportError",
    "ExportRecord",
    "ExportResult",
    "ExportStatus",
    "ExporterRegistry",
    "ModelExportService",
    "ModelExporter",
    "VerifyInput",
    "VerifyLevel",
    "VerifyResult",
    "append_record",
    "default_exporter_registry",
    "missing_packages",
    "read_exports",
    "require_installed",
    "wer_difference",
]
