"""CsvAudioSource: a folder of audio files plus a `manifest.csv`, loaded into AudioSample objects.

How to run: this module is not a script; the composition root creates the source. Example
(PowerShell, project root; `config` is an AppConfig loaded with the `data` section registered):
    $env:PYTHONPATH = "src"
    python -c "from stt.datasource import create_data_source; ..."   # see docs/COMPONENTS.md, C1
Layout: <data.source.path>/manifest.csv (UTF-8, a BOM is tolerated, one row per clip) and the audio
files; relative audio paths are resolved relative to the manifest. The mapped columns come from
`data.columns.*`; every other column is kept in `extra`. Reads headers only, never audio arrays.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType

from stt.datasource.audio_header import read_duration_seconds
from stt.datasource.config import ColumnsConfig, DataConfig
from stt.datasource.errors import DataSourceError
from stt.datasource.report import build_load_report, write_rejected_csv
from stt.datasource.types import AudioSample, ExtraValue, LoadResult, Rejection

MANIFEST_NAME = "manifest.csv"
HeaderReader = Callable[[Path], float]

_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}


def convert_extra(value: str, kind: str) -> ExtraValue | None:
    """Convert one manifest text value to its configured type.

    Returns None for an empty value of a non-text type (the key is then left out of `extra`).
    Errors: ValueError when the text does not fit the type.
    """
    if kind == "str":
        return value
    text = value.strip()
    if text == "":
        return None
    if kind == "int":
        return int(text)
    if kind == "float":
        return float(text)
    if kind == "bool":
        lowered = text.lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise ValueError(f"'{value}' is not a boolean (use true/false)")
    raise ValueError(f"unknown type '{kind}'")


class CsvAudioSource:
    """Load a manifest.csv plus audio files.

    Inputs: the `data` config, `out_dir` (where rejected.csv is written; the run folder), an optional
    header reader (default: soundfile), and for composition by HfParquetSource an explicit manifest
    path and column mapping.
    Output: `load()` returns a LoadResult (samples sorted by id, report, rejections).
    Errors: DataSourceError (exit code 3) for a missing manifest or column, a missing group column
    with `group_fallback: fail`, an empty or duplicate id, a rejected ratio above the gate, or zero
    samples. With `data.on_invalid: fail` the first rejected sample fails the load.
    """

    def __init__(
        self,
        config: DataConfig,
        out_dir: Path,
        *,
        header_reader: HeaderReader = read_duration_seconds,
        manifest_path: Path | None = None,
        columns: ColumnsConfig | None = None,
    ) -> None:
        self._config = config
        self._out_dir = out_dir
        self._header_reader = header_reader
        self._manifest = manifest_path or (config.source.path / MANIFEST_NAME)
        self._columns = columns or config.columns

    def load(self) -> LoadResult:
        """Read the manifest and return the samples and the report (see the class docstring)."""
        manifest = self._manifest
        if not manifest.is_file():
            raise DataSourceError(
                f"Manifest not found: {manifest}. A csv source needs a file named {MANIFEST_NAME} in data.source.path "
                "(UTF-8, one row per clip). Check data.source.path."
            )
        samples: list[AudioSample] = []
        rejections: list[Rejection] = []
        rejected_splits: dict[str, str | None] = {}
        missing_extra: dict[str, int] = {}
        seen_ids: set[str] = set()
        duplicates: list[str] = []
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, restkey="__too_many_cells__")
            header = list(reader.fieldnames or [])
            self._check_columns(header)
            mapped = {c for c in (self._columns.id, self._columns.audio, self._columns.text, self._columns.group, self._columns.split) if c}
            for row in reader:
                line = reader.line_num
                row_id = self._row_id(row, line)
                if row_id in seen_ids:
                    if len(duplicates) < 5:
                        duplicates.append(row_id)
                    continue
                seen_ids.add(row_id)
                split = (row.get(self._columns.split) or "").strip() or None if self._columns.split else None
                outcome = self._build_sample(row, row_id, split, mapped, manifest.parent, missing_extra)
                if isinstance(outcome, Rejection):
                    rejections.append(outcome)
                    rejected_splits[outcome.id] = split
                    if self._config.on_invalid == "fail":
                        write_rejected_csv(rejections, self._out_dir)
                        raise DataSourceError(
                            f"Invalid sample {outcome.id} (line {line}): {outcome.reason}. {outcome.detail} "
                            "data.on_invalid is 'fail'; set it to 'skip' to continue and review rejected.csv."
                        )
                else:
                    samples.append(outcome)
        if duplicates:
            raise DataSourceError(
                f"Duplicate sample ids in {manifest}: {', '.join(duplicates)}. "
                "Ids must be unique: the id column (or audio path) mapping is wrong. Check data.columns.id."
            )
        write_rejected_csv(rejections, self._out_dir)
        report = build_load_report(samples, rejections, rejected_splits, missing_extra)
        seen = report.seen
        if seen == 0 or not samples:
            raise DataSourceError(
                f"No samples were loaded from {manifest} ({seen} rows seen, {len(rejections)} rejected). "
                f"Check the mapping in data.columns and {self._out_dir / 'rejected.csv'}."
            )
        ratio = len(rejections) / seen
        if ratio > self._config.max_rejected_ratio:
            raise DataSourceError(
                f"Rejected ratio {ratio:.3f} is above data.max_rejected_ratio {self._config.max_rejected_ratio}: "
                f"{len(rejections)} of {seen} rows rejected, reasons {dict(report.rejection_reasons)}. "
                f"Read {self._out_dir / 'rejected.csv'}, then fix data.source.path or data.columns. "
                "Do not raise the ratio just to pass."
            )
        samples.sort(key=lambda s: s.id)
        return LoadResult(samples=samples, report=report, rejections=tuple(rejections))

    def _check_columns(self, header: list[str]) -> None:
        cols = self._columns
        wanted = {"data.columns.audio": cols.audio, "data.columns.text": cols.text}
        for key, name in (("data.columns.id", cols.id), ("data.columns.group", cols.group), ("data.columns.split", cols.split)):
            if name:
                wanted[key] = name
        for name in self._config.extra_types:
            wanted[f"data.extra_types.{name}"] = name
        for key, name in wanted.items():
            if name not in header:
                raise DataSourceError(
                    f"Column '{name}' (from {key}) does not exist in {self._manifest}. Existing columns: {', '.join(header)}. "
                    "Fix the name in the YAML configuration."
                )
        if cols.group is None and self._config.group_fallback == "fail":
            raise DataSourceError(
                "No group column is mapped (data.columns.group) and data.group_fallback is 'fail'. A wrong group causes "
                "train/test leakage, so the run stops. Fix: (1) set data.columns.group to a column that identifies the "
                "recording or lesson, (2) or set data.group_fallback to 'parent_dir' or 'file_stem' knowingly, "
                "(3) or add such a column to the manifest."
            )

    def _row_id(self, row: dict[str, str | None], line: int) -> str:
        cols = self._columns
        value = row.get(cols.id) if cols.id else row.get(cols.audio)
        if value is None or value.strip() == "":
            which = f"id column '{cols.id}'" if cols.id else f"audio column '{cols.audio}'"
            raise DataSourceError(
                f"Empty sample id at line {line} of {self._manifest} (taken from the {which}). Every row needs a unique id."
            )
        return value.strip()

    def _build_sample(
        self,
        row: dict[str, str | None],
        row_id: str,
        split: str | None,
        mapped: set[str],
        base: Path,
        missing_extra: dict[str, int],
    ) -> AudioSample | Rejection:
        cols = self._columns
        if "__too_many_cells__" in row or any(v is None for v in row.values()):
            return Rejection(row_id, "malformed_row", "the row has a different number of cells than the header")
        audio_cell = (row[cols.audio] or "").strip()
        audio_path = Path(audio_cell) if audio_cell else None
        if audio_path is not None and not audio_path.is_absolute():
            audio_path = base / audio_path
        if audio_path is None or not audio_path.is_file():
            return Rejection(row_id, "missing_audio", f"audio file not found: {audio_path if audio_path else '(empty path)'}")
        text = row[cols.text] or ""
        if text.strip() == "":
            return Rejection(row_id, "empty_text", "the transcript is empty or only whitespace")
        try:
            duration = self._header_reader(audio_path)
        except ValueError as exc:
            return Rejection(row_id, "unreadable_audio", f"{audio_path}: {exc}")
        group = self._group(row, audio_path)
        if group is None:
            return Rejection(row_id, "missing_group", f"column '{cols.group}' is empty")
        extra: dict[str, ExtraValue] = {}
        for name, value in row.items():
            if name in mapped or value is None:
                continue
            kind = self._config.extra_types.get(name, "str")
            try:
                converted = convert_extra(value, kind)
            except ValueError as exc:
                return Rejection(row_id, "bad_extra_value", f"column '{name}' (type {kind}): {exc}")
            if converted is None:
                missing_extra[name] = missing_extra.get(name, 0) + 1
            else:
                extra[name] = converted
        return AudioSample(
            id=row_id,
            audio_path=audio_path,
            text=text,
            group_id=group,
            duration_sec=duration,
            source_split=split,
            extra=MappingProxyType(extra),
        )

    def _group(self, row: dict[str, str | None], audio_path: Path) -> str | None:
        if self._columns.group:
            value = (row[self._columns.group] or "").strip()
            return value or None
        if self._config.group_fallback == "parent_dir":
            return audio_path.parent.name or None
        return audio_path.stem or None
