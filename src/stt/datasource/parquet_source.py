"""HfParquetSource: a LOCAL folder of Hugging Face parquet files, extracted to audio files + manifest.csv.

How to run: this module is not a script; the composition root creates the source. Example
(PowerShell, project root; see docs/COMPONENTS.md, C1, for the full working example):
    $env:PYTHONPATH = "src"
    python -c "from stt.datasource import create_data_source; ..."
What it does: (1) extracts the audio bytes of every row to `<data.extract.dir>/audio/<split>/<id>.<ext>`,
(2) writes `manifest.csv` LAST (temp file + rename; a manifest means the extraction is complete) and
`extract_info.json`, (3) loads the manifest through CsvAudioSource (no duplicated logic).
Ids are `<source_split>_<row_index:06d>` in sorted file order (the source `audio.path` is not unique).
Nested `metadata` fields become columns with the prefix `meta_`. The source files are never changed.
pyarrow is imported only in this module.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from stt.datasource.config import DataConfig
from stt.datasource.csv_source import MANIFEST_NAME, CsvAudioSource, HeaderReader
from stt.datasource.audio_header import read_duration_seconds
from stt.datasource.errors import DataSourceError
from stt.datasource.types import LoadResult

INFO_NAME = "extract_info.json"
ID_COLUMN = "id"
SPLIT_COLUMN = "source_split"


def guess_extension(data: bytes) -> str:
    """Guess a file extension from the first bytes (mp3, wav, flac, ogg); `.bin` when unknown."""
    if data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return ".mp3"
    if data[:4] == b"RIFF":
        return ".wav"
    if data[:4] == b"fLaC":
        return ".flac"
    if data[:4] == b"OggS":
        return ".ogg"
    return ".bin"


def flatten(prefix: str, value: Any, out: dict[str, str]) -> None:
    """Flatten a nested dict into `out` as `<prefix>_<key>` text values (None becomes empty text)."""
    if isinstance(value, dict):
        for key, inner in value.items():
            flatten(f"{prefix}_{key}", inner, out)
    else:
        out[prefix] = "" if value is None else str(value)


def split_name(path: Path) -> str:
    """The source split of a parquet file: the part of its name before the first '-' (`train-0000...` -> `train`)."""
    return path.name.split("-")[0] if "-" in path.name else path.stem


class HfParquetSource:
    """Extract Hugging Face parquet files, then load the manifest with CsvAudioSource.

    Inputs: the `data` config (`source.type: hf_parquet`, `source.path` = folder with *.parquet files,
    `extract.dir`, `extract.overwrite`, `columns.*` named as in the manifest: audio and transcript
    columns keep their parquet names, `metadata` fields are `meta_<name>`, the split is `source_split`),
    and `out_dir` for rejected.csv.
    Output: `load()` returns a LoadResult.
    Errors: DataSourceError for a folder without parquet files, a missing audio or transcript column,
    or an existing extraction that does not match the current source files (set
    `data.extract.overwrite: true`, or delete the extract folder, to re-extract).
    """

    def __init__(self, config: DataConfig, out_dir: Path, *, header_reader: HeaderReader = read_duration_seconds) -> None:
        if config.extract.dir is None:
            raise DataSourceError("data.extract.dir is required for hf_parquet sources.")
        self._config = config
        self._out_dir = out_dir
        self._header_reader = header_reader
        self._extract_dir: Path = config.extract.dir

    def load(self) -> LoadResult:
        """Extract if needed (idempotent), then load the manifest. See the class docstring."""
        files = sorted(self._config.source.path.rglob("*.parquet"))
        if not files:
            raise DataSourceError(
                f"No .parquet files found under {self._config.source.path}. Check data.source.path "
                "(download the data first with tools/fetch_data.py at home)."
            )
        self.extract(files)
        columns = self._config.columns.model_copy(update={"id": ID_COLUMN})
        csv_source = CsvAudioSource(
            self._config,
            self._out_dir,
            header_reader=self._header_reader,
            manifest_path=self._extract_dir / MANIFEST_NAME,
            columns=columns,
        )
        return csv_source.load()

    def extract(self, files: list[Path]) -> Path:
        """Write the audio files and manifest.csv under `data.extract.dir`; return the manifest path.

        A complete extraction (manifest present) of the same source files is reused unless
        `data.extract.overwrite` is true. An incomplete one (no manifest) is redone.
        """
        manifest = self._extract_dir / MANIFEST_NAME
        sources = [{"file": f.name, "bytes": f.stat().st_size} for f in files]
        if manifest.is_file() and not self._config.extract.overwrite:
            self._check_matches_sources(sources)
            return manifest
        if self._extract_dir.exists():
            shutil.rmtree(self._extract_dir)  # our own derived data: partial or to be overwritten
        (self._extract_dir / "audio").mkdir(parents=True)

        audio_col, text_col = self._config.columns.audio, self._config.columns.text
        counters: dict[str, int] = {}
        rows_per_file: dict[str, int] = {}
        header: list[str] | None = None
        temp = self._extract_dir / (MANIFEST_NAME + ".tmp")
        with temp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for path in files:
                parquet = pq.ParquetFile(path)
                names = parquet.schema_arrow.names
                for needed, key in ((audio_col, "data.columns.audio"), (text_col, "data.columns.text")):
                    if needed not in names:
                        raise DataSourceError(
                            f"Column '{needed}' (from {key}) does not exist in {path.name}. "
                            f"Existing columns: {', '.join(names)}. Fix the name in the YAML configuration."
                        )
                split = split_name(path)
                rows_per_file[path.name] = parquet.metadata.num_rows
                for group_index in range(parquet.metadata.num_row_groups):
                    table = parquet.read_row_group(group_index)
                    for record in table.to_pylist():
                        index = counters.get(split, 0)
                        counters[split] = index + 1
                        row_id = f"{split}_{index:06d}"
                        audio = record.get(audio_col) or {}
                        blob = audio.get("bytes") if isinstance(audio, dict) else None
                        rel = ""
                        if blob:
                            rel_path = Path("audio") / split / f"{row_id}{guess_extension(blob)}"
                            target = self._extract_dir / rel_path
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(blob)
                            rel = rel_path.as_posix()
                        flat: dict[str, str] = {}
                        for key, value in record.items():
                            if key == audio_col:
                                continue
                            if key == "metadata" or isinstance(value, dict):
                                flatten("meta" if key == "metadata" else key, value, flat)
                            else:
                                flat[key] = "" if value is None else str(value)
                        row = {ID_COLUMN: row_id, audio_col: rel, SPLIT_COLUMN: split, **flat}
                        if header is None:
                            header = list(row)
                            writer.writerow(header)
                        writer.writerow([row.get(h, "") for h in header])
        if header is None:
            raise DataSourceError(f"The parquet files under {self._config.source.path} contain no rows.")
        info = {"source_files": sources, "rows_per_file": rows_per_file, "rows_per_split": counters}
        (self._extract_dir / INFO_NAME).write_text(json.dumps(info, indent=2), encoding="utf-8")
        temp.replace(manifest)  # the manifest is written LAST: its presence means "complete"
        return manifest

    def _check_matches_sources(self, sources: list[dict[str, object]]) -> None:
        info_path = self._extract_dir / INFO_NAME
        try:
            recorded = json.loads(info_path.read_text(encoding="utf-8"))["source_files"]
        except (OSError, ValueError, KeyError) as exc:
            raise DataSourceError(
                f"The extraction in {self._extract_dir} has no readable {INFO_NAME}. "
                "Delete that folder (or set data.extract.overwrite: true) to extract again."
            ) from exc
        if recorded != sources:
            raise DataSourceError(
                f"The extraction in {self._extract_dir} was made from other source files than the current ones "
                f"({[s['file'] for s in recorded]} vs {[s['file'] for s in sources]}). "
                "Set data.extract.overwrite: true, or delete that folder, to extract again."
            )
