"""The registry of exporters: name -> exporter factory (standard library only).

How to run: this module is not a script; the service uses it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.modelexporter import default_exporter_registry; print(default_exporter_registry().names())"
Example output:
    ['hf_safetensors']
v1 has one exporter, `hf_safetensors` (the format the registry already stores). A new target format is a new class registered here by name; each new format needs its own approval for its
dependency (owner rule). An unknown name fails and lists the registered names.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType

from stt.modelexporter.errors import ExportConfigError, ExportDependencyError
from stt.modelexporter.types import ModelExporter


class ExporterRegistry:
    """A read-only name -> exporter factory mapping."""

    def __init__(self, factories: Mapping[str, Callable[[], ModelExporter]]) -> None:
        self._factories: Mapping[str, Callable[[], ModelExporter]] = MappingProxyType(dict(factories))

    def names(self) -> list[str]:
        """The registered names, sorted."""
        return sorted(self._factories)

    def create(self, name: str) -> ModelExporter:
        """The exporter `name`. Errors: ExportConfigError listing the registered names."""
        factory = self._factories.get(name)
        if factory is None:
            raise ExportConfigError(f"Unknown exporter '{name}' in export.formats. Registered exporters: {', '.join(self.names())}.")
        return factory()


def missing_packages(requires: Sequence[str]) -> list[str]:
    """The packages of `requires` that are not installed (looked up without importing them)."""
    return [name for name in requires if importlib.util.find_spec(name) is None]


def require_installed(exporter: ModelExporter) -> None:
    """Fail unless the packages the exporter needs are installed. Errors: ExportDependencyError naming the packages and saying that the machine is offline."""
    missing = missing_packages(exporter.requires)
    if missing:
        raise ExportDependencyError(
            f"The exporter '{exporter.name}' needs the package(s) {', '.join(missing)}, which are not installed. This machine is offline: bring the package(s) in advance with the other "
            "wheels (tools/fetch_wheels.py at home); do not enable online mode as a fix."
        )


def default_exporter_registry() -> ExporterRegistry:
    """The registry with the v1 exporter."""
    from stt.modelexporter.hf_safetensors import HfSafetensorsExporter  # imported here: the adapter module is the only one that touches the model loader

    return ExporterRegistry({HfSafetensorsExporter.name: HfSafetensorsExporter})
