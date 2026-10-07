"""Typed data objects of the ModelExporter component and the ModelExporter Protocol.

How to run: this module is not a script; the other modules and the callers import it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.modelexporter.types import ExportResult; print(list(ExportResult.__dataclass_fields__))"
Example output:
    ['exporter', 'target_dir', 'files', 'seconds']
No file access and no model imports. The fields are those of the C11 brief; the additions are marked: `VerifyInput.base_model_dir` (the base model's folder, so that a load check can compare the
architecture) and `ExportRecord.status`, `.files`, `.exported_at`, `.entry_id` (an `exports.json` record needs them).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

VerifyLevel = Literal["load", "transcribe"]
ExportStatus = Literal["ok", "failed", "unverified"]


@dataclass(frozen=True)
class ExportResult:
    """What an exporter made. `target_dir` is the folder that holds the exported model (the `hf_safetensors` exporter does not copy: it is the registry entry's own model folder); `files` are the
    exported files; `seconds` is how long the export took."""

    exporter: str
    target_dir: Path
    files: list[Path]
    seconds: float


@dataclass(frozen=True)
class VerifyInput:
    """What a verification may use. `audio_paths` and `expected_texts` are the check files and the transcripts of the REGISTERED model for them (for exporters that verify at the `transcribe`
    level); `max_wer_diff` is the allowed absolute WER difference. `base_model_dir` (an addition to the brief) is the folder of the base model, when it exists on this machine: a load check
    compares the architecture with it."""

    audio_paths: list[Path]
    expected_texts: list[str]
    max_wer_diff: float
    base_model_dir: Path | None = None


@dataclass(frozen=True)
class VerifyResult:
    """The outcome of a verification. `level` says how deep it looked (`load` or `transcribe`), `ok` whether it passed, `detail` what was checked and what differed, `wer_diff` the WER difference at the
    `transcribe` level (None at the `load` level)."""

    level: VerifyLevel
    ok: bool
    detail: str
    wer_diff: float | None = None


@dataclass(frozen=True)
class ExportRecord:
    """One line of `exports.json` in the registry entry: the exporter, the target folder (relative to the entry), the verification result, and (additions) the status, the files, the time (passed in: the
    component never reads the clock) and the entry. `status` is `ok`, `failed` (the verification failed: the export is kept and marked) or `unverified` (`export.verify` is false)."""

    entry_id: str
    exporter: str
    target_dir: str
    files: tuple[str, ...]
    verify: VerifyResult | None
    status: ExportStatus
    seconds: float
    exported_at: str = field(default="")


class ModelExporter(Protocol):
    """An exporter: a target format. A new format is a new class in the registry; its format-specific libraries are imported only inside its own adapter module."""

    name: str
    requires: tuple[str, ...]

    def export(self, source: Path, target_dir: Path) -> ExportResult:
        """Convert the model in the folder `source` (never modified) into `target_dir`. Errors: ExportError."""
        ...

    def verify(self, result: ExportResult, check: VerifyInput) -> VerifyResult:
        """Verify an export (never raises for a failed check: it returns `ok` false with the detail)."""
        ...
