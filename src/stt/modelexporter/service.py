"""ModelExportService: export a registered model, verify the export, and record it in `exports.json` of the registry entry (C11).

How to run: from Python (PowerShell, project root; the working example is in docs/COMPONENTS.md, C11):
    $env:PYTHONPATH = "src"
    python -c "from stt.modelexporter import ExportConfig, default_exporter_registry; print(ExportConfig().formats, default_exporter_registry().names())"
Example output:
    ['hf_safetensors'] ['hf_safetensors']
`ModelExportService(config, entries, exporters).export(entry_id, formats, check, exported_at, explicit) -> list[ExportRecord]`. Order of one call, and the failure rules of the brief: (1) an unknown
format name FAILS listing the registered names (before anything is touched); (2) an entry that does not exist FAILS; an entry of kind `baseline_retained` has nothing to export: explicitly asked it FAILS
with a plain explanation, in the automatic step (`explicit=False`) it is skipped and logged; (3) the entry's files are checked against its checksums and a mismatch FAILS before anything is exported
(an entry registered without checksums is exported with a warning: there is nothing to check against); (4) an exporter whose package is missing FAILS naming the package and saying that the machine
is offline; (5) a target folder `exports/<exporter>/` that already exists and is not empty FAILS (never overwrite); (6) the exporter runs, then, with `export.verify`, its verification; (7) the record is
appended to `exports.json` (the entry's model files and its card are never touched); a failed verification keeps the export, marks it `failed` in `exports.json` and, after every format was done, raises
ExportError (exit code 5). The component reads no clock (the time of the record is passed in) and never evaluates a model.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from stt.modelexporter.config import ExportConfig
from stt.modelexporter.errors import ExportConfigError, ExportError
from stt.modelexporter.registry import ExporterRegistry, default_exporter_registry, require_installed
from stt.modelexporter.types import ExportRecord, ExportStatus, VerifyInput, VerifyResult
from stt.registry import ModelCard, RegistryEntry, RegistryError, VerifyReport

logger = logging.getLogger("stt.modelexporter")
EXPORTS_FILE = "exports.json"
EXPORTS_FOLDER = "exports"
SCHEMA_VERSION = 1


class EntryAccess(Protocol):
    """What the service needs from the registry (the real `ModelRegistry` fits)."""

    def get(self, entry_id: str) -> RegistryEntry:
        """The entry. Raises RegistryError when it does not exist."""
        ...

    def verify(self, entry_id: str) -> VerifyReport:
        """Check the entry's files against its checksums. Raises RegistryVerifyError on a mismatch and RegistryError when there are no checksums."""
        ...

    def card(self, entry_id: str) -> ModelCard:
        """The card of the entry."""
        ...


def _record_to_json(r: ExportRecord) -> dict[str, Any]:
    verify = None if r.verify is None else {"level": r.verify.level, "ok": r.verify.ok, "detail": r.verify.detail, "wer_diff": r.verify.wer_diff}
    return {"entry_id": r.entry_id, "exporter": r.exporter, "target_dir": r.target_dir, "files": list(r.files), "verify": verify, "status": r.status, "seconds": r.seconds, "exported_at": r.exported_at}


def _record_from_json(raw: dict[str, Any]) -> ExportRecord:
    verify = None if raw["verify"] is None else VerifyResult(raw["verify"]["level"], raw["verify"]["ok"], raw["verify"]["detail"], raw["verify"]["wer_diff"])
    return ExportRecord(raw["entry_id"], raw["exporter"], raw["target_dir"], tuple(raw["files"]), verify, raw["status"], raw["seconds"], raw.get("exported_at", ""))


def read_exports(entry_dir: Path) -> list[ExportRecord]:
    """The records of `exports.json` in an entry folder (empty when there is none). Errors: ExportError when the file is damaged (it is never repaired)."""
    path = entry_dir / EXPORTS_FILE
    if not path.is_file():
        return []
    try:
        return [_record_from_json(r) for r in json.loads(path.read_text(encoding="utf-8"))["exports"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ExportError(f"The export record {path} cannot be read ({type(exc).__name__}: {exc}). Look at it by hand; it is never repaired or replaced.") from exc


def append_record(entry_dir: Path, record: ExportRecord) -> Path:
    """Append one record to `exports.json` (a temporary file and a rename; the records already there are kept)."""
    records = [*read_exports(entry_dir), record]
    path = entry_dir / EXPORTS_FILE
    body = {"schema_version": SCHEMA_VERSION, "exports": [_record_to_json(r) for r in records]}
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(body, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    except OSError as exc:
        raise ExportError(f"Cannot write {path} ({exc}). Check that the registry folder is writable and the disk is not full.") from exc
    return path


class ModelExportService:
    """Exports registered models. The exporters come from a registry (by name); `clock` is injectable for tests."""

    def __init__(self, config: ExportConfig, entries: EntryAccess, exporters: ExporterRegistry | None = None, clock: Callable[[], float] | None = None) -> None:
        self._config = config
        self._entries = entries
        self._exporters = exporters or default_exporter_registry()
        self._clock = clock

    def verify_input(self, audio_paths: Sequence[Path], expected_texts: Sequence[str], base_model_dir: Path | None = None) -> VerifyInput:
        """The input of a `transcribe`-level verification: the first `export.verify_samples` files, the allowed difference `export.max_wer_diff`."""
        n = self._config.verify_samples
        return VerifyInput(list(audio_paths)[:n], list(expected_texts)[:n], self._config.max_wer_diff, base_model_dir)

    def records(self, entry_id: str) -> list[ExportRecord]:
        """The exports of an entry, in the order they were made. Errors: ExportError (the entry does not exist, or `exports.json` is damaged)."""
        return read_exports(self._entry(entry_id).card_path.parent)

    def _entry(self, entry_id: str) -> RegistryEntry:
        try:
            return self._entries.get(entry_id)
        except RegistryError as exc:
            raise ExportError(str(exc)) from exc

    def export(self, entry_id: str, formats: Sequence[str] | None = None, check: VerifyInput | None = None, exported_at: str = "", explicit: bool = True) -> list[ExportRecord]:
        """Export the entry in `formats` (default `export.formats`) and record each export (see the module docstring for the order and the failure rules).

        `explicit` is True for a command that names the entry (a `baseline_retained` entry then FAILS) and False for the automatic step after a registration (it is skipped and logged).
        Errors: ExportConfigError (unknown format, exit code 2); ExportError and its subclass ExportDependencyError (exit code 5), also after the records were written when a verification failed.
        """
        names = list(formats) if formats is not None else list(self._config.formats)
        exporters = [self._exporters.create(name) for name in names]  # an unknown name fails here, before anything is touched
        entry = self._entry(entry_id)
        if entry.kind != "model" or entry.model_dir is None:
            if explicit:
                raise ExportError(f"The registry entry {entry_id} kept the baseline (kind {entry.kind}): there is no model to export. Export an entry of kind `model`; `python -m stt registry list` shows them.")
            logger.info("The registry entry %s kept the baseline: nothing to export, the automatic export is skipped.", entry_id)
            return []
        entry_dir = entry.card_path.parent
        try:
            self._entries.verify(entry_id)
        except RegistryError as exc:
            if "has no SHA256SUMS" in str(exc):
                logger.warning("The entry %s has no checksums: its files cannot be checked before the export.", entry_id)
            else:
                raise ExportError(f"The files of the registry entry {entry_id} do not match its checksums, so nothing was exported: {exc}") from exc
        for exporter in exporters:
            require_installed(exporter)
        for exporter in exporters:
            target = entry_dir / EXPORTS_FOLDER / exporter.name
            if target.exists() and any(target.iterdir()):
                raise ExportError(f"The export folder {target} already exists and is not empty: an export never overwrites. Export to a new target, or look at the folder first (nothing is deleted for you).")
        base = self._base_model_dir(entry_id)
        base_check = check if check is not None else VerifyInput([], [], self._config.max_wer_diff)
        base_check = replace(base_check, base_model_dir=base_check.base_model_dir or base)
        records: list[ExportRecord] = []
        failed: list[str] = []
        for exporter in exporters:
            target = entry_dir / EXPORTS_FOLDER / exporter.name
            result = exporter.export(entry.model_dir, target)
            verification = exporter.verify(result, base_check) if self._config.verify else None
            status: ExportStatus = "unverified" if verification is None else "ok" if verification.ok else "failed"
            relative = tuple(_relative(p, entry_dir) for p in result.files)
            record = ExportRecord(entry_id, exporter.name, _relative(result.target_dir, entry_dir), relative, verification, status, round(result.seconds, 3), exported_at)
            append_record(entry_dir, record)
            records.append(record)
            if status == "failed":
                failed.append(f"{exporter.name}: {verification.detail if verification else ''}")
            logger.info("Exported %s with %s: %s.", entry_id, exporter.name, status)
        if failed:
            raise ExportError(f"The verification of the export of {entry_id} failed ({'; '.join(failed)}). The export is kept and marked failed in {entry_dir / EXPORTS_FILE}.")
        return records

    def _base_model_dir(self, entry_id: str) -> Path | None:
        try:
            path = Path(self._entries.card(entry_id).base_model_path)
        except (RegistryError, OSError):
            return None
        return path if (path / "config.json").is_file() else None


def _relative(path: Path, folder: Path) -> str:
    try:
        return path.relative_to(folder).as_posix()
    except ValueError:
        return path.as_posix()  # a path outside the entry (never expected) is kept as it is
