"""SplitStore: saves, loads and exports a split (the splitters never touch files).

How to run: this module is not a script; `resolve_split` (stt.splitter.factory) uses it. Example
(PowerShell, project root; `result` is a SplitResult, `samples` the same input list):
    $env:PYTHONPATH = "src"
    python -c "from stt.splitter import SplitStore; store = SplitStore(); print(store.path(__import__('pathlib').Path('splits/v1')))"
Example output:
    splits/v1/split.json   (printed with the separator of your system)
File format (JSON, UTF-8): schema_version, parameters (mode, seed, names, ratios, ratio unit, group key,
version), data_fingerprint, fingerprint, and the sample ids of each split and of the dropped samples.
`export_csv` writes `id,split` rows that match the JSON.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

from stt.datasource import AudioSample
from stt.splitter.checks import verify_result
from stt.splitter.errors import SplitError
from stt.splitter.keys import group_keys
from stt.splitter.report import build_result, data_fingerprint
from stt.splitter.types import SplitParams, SplitResult

SCHEMA_VERSION = 1
SPLIT_FILE = "split.json"
CSV_FILE = "split.csv"
DROPPED_GUARD = "dropped_guard"
DROPPED_OVERLAP = "dropped_overlap"


class SplitStore:
    """Persistence of a SplitResult: `save(result, dir)`, `load(dir, samples)`, `export_csv(result, path)`."""

    def path(self, directory: Path) -> Path:
        """The split file inside `directory`."""
        return directory / SPLIT_FILE

    def exists(self, directory: Path) -> bool:
        """True when `directory` holds a saved split."""
        return self.path(directory).is_file()

    def save(self, result: SplitResult, directory: Path) -> Path:
        """Write the split file (temp file + rename). A saved split is never overwritten: FAILS if one exists."""
        target = self.path(directory)
        if target.exists():
            raise SplitError(
                f"A split is already saved in {directory}. A saved split is never overwritten: "
                "set a new split.version to create a new one."
            )
        params = result.report.params
        document = {
            "schema_version": SCHEMA_VERSION,
            "parameters": {
                "mode": params.mode,
                "seed": params.seed,
                "names": list(params.names),
                "ratios": list(params.ratios),
                "ratio_unit": params.ratio_unit,
                "group_column": params.group_column,
                "group_regex": params.group_regex,
                "speaker_column": params.speaker_column,
                "version": params.version,
            },
            "data_fingerprint": result.data_fingerprint,
            "fingerprint": result.fingerprint,
            "splits": {name: [s.id for s in group] for name, group in result.splits.items()},
            DROPPED_GUARD: [s.id for s in result.dropped_guard],
            DROPPED_OVERLAP: [s.id for s in result.dropped_overlap],
        }
        directory.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".json.tmp")
        temp.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
        temp.replace(target)
        return target

    def load(self, directory: Path, samples: Sequence[AudioSample]) -> SplitResult:
        """Rebuild the SplitResult saved in `directory` for the current `samples`.

        Errors: SplitError when there is no split file, the schema version is unknown, the data
        fingerprint does not match the current data (create a new split.version), or an id is unknown.
        """
        target = self.path(directory)
        if not target.is_file():
            raise SplitError(f"No saved split in {directory} (expected {SPLIT_FILE}).")
        try:
            doc = json.loads(target.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise SplitError(f"The split file {target} is not valid JSON: {exc}.") from exc
        if doc.get("schema_version") != SCHEMA_VERSION:
            raise SplitError(f"The split file {target} has schema_version {doc.get('schema_version')}, expected {SCHEMA_VERSION}.")
        ordered = sorted(samples, key=lambda s: s.id)
        if data_fingerprint(ordered) != doc["data_fingerprint"]:
            raise SplitError(
                f"The data changed since the split in {directory} was saved (fingerprint mismatch). "
                "Create a new split.version; keep the old one so past experiments stay interpretable."
            )
        by_id = {s.id: s for s in ordered}
        saved = doc["parameters"]
        params = SplitParams(
            mode=saved["mode"],
            seed=int(saved["seed"]),
            names=tuple(saved["names"]),
            ratios=tuple(float(r) for r in saved["ratios"]),
            ratio_unit=saved["ratio_unit"],
            group_column=saved["group_column"],
            group_regex=saved["group_regex"],
            speaker_column=saved["speaker_column"],
            version=saved["version"],
        )
        try:
            split_of = {i: name for name, ids in doc["splits"].items() for i in ids}
            guard = set(doc[DROPPED_GUARD])
            overlap = set(doc[DROPPED_OVERLAP])
            unknown = [i for i in [*split_of, *guard, *overlap] if i not in by_id]
        except (KeyError, TypeError) as exc:
            raise SplitError(f"The split file {target} is incomplete or damaged ({exc!r}).") from exc
        if unknown:
            raise SplitError(f"The split file {target} lists unknown sample ids, for example {unknown[:3]}.")
        keys = group_keys(ordered, params.group_column, params.group_regex)
        result = build_result(ordered, split_of, guard, overlap, params, keys, notes=("loaded from a saved split",))
        verify_result(result, ordered, keys, None, groups_may_span_splits=params.mode == "time_block")
        if result.fingerprint != doc["fingerprint"]:
            raise SplitError(f"The split file {target} does not match its own fingerprint: it was changed by hand.")
        return result

    def export_csv(self, result: SplitResult, path: Path) -> Path:
        """Write `id,split` rows (dropped samples get the split `dropped_guard` or `dropped_overlap`)."""
        rows = [(s.id, name) for name, group in result.splits.items() for s in group]
        rows += [(s.id, DROPPED_GUARD) for s in result.dropped_guard]
        rows += [(s.id, DROPPED_OVERLAP) for s in result.dropped_overlap]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "split"])
            writer.writerows(sorted(rows))
        return path
