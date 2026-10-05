"""index.csv: one row per registry entry with its metrics (standard library only).

How to run: this module is not a script; the registry uses it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.registry.index import COLUMNS; print(COLUMNS[:5])"
Example output:
    ('entry_id', 'experiment_id', 'kind', 'verdict', 'eval_version')
The index is DERIVED from the entry cards (`ModelRegistry.rebuild_index()` makes it again). It is written to a temporary file and renamed, so a reader never sees a
half-written file. It is never rebuilt silently: a missing or damaged index fails with the fix. Open it in Excel (UTF-8 with a byte order mark).
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from stt.registry.errors import RegistryError
from stt.registry.types import ModelCard

INDEX_NAME = "index.csv"
COLUMNS = (
    "entry_id",
    "experiment_id",
    "kind",
    "verdict",
    "eval_version",
    "test_fingerprint",
    "test_evaluations",
    "val_wer",
    "val_baseline_wer",
    "val_delta_wer",
    "val_verdict",
    "test_status",
    "test_wer",
    "test_baseline_wer",
    "test_verdict",
    "trial_id",
    "params",
    "model_dir",
    "code_version",
    "generated_at",
)


def _cell(value: object) -> str:
    return "" if value is None else repr(value) if isinstance(value, float) else str(value)


def row_from_card(card: ModelCard, entry_id: str, experiment_id: str) -> dict[str, str]:
    """The index row of an entry (all values as text)."""
    val, test = card.val, card.test
    values = {
        "entry_id": entry_id,
        "experiment_id": experiment_id,
        "kind": card.kind,
        "verdict": card.verdict,
        "eval_version": card.eval_version,
        "test_fingerprint": card.test_fingerprint,
        "test_evaluations": card.test_evaluations,
        "val_wer": val.wer if val else None,
        "val_baseline_wer": val.baseline_wer if val else None,
        "val_delta_wer": val.delta_wer_vs_baseline if val else None,
        "val_verdict": val.verdict if val else None,
        "test_status": card.test_status,
        "test_wer": test.wer if test else None,
        "test_baseline_wer": test.baseline_wer if test else None,
        "test_verdict": test.verdict if test else None,
        "trial_id": card.trial_id,
        "params": json.dumps(dict(card.searched_params), ensure_ascii=False, sort_keys=True) if card.searched_params else "",
        "model_dir": f"{entry_id}/model" if card.kind == "model" else "",
        "code_version": card.code_version,
        "generated_at": card.generated_at,
    }
    return {name: _cell(values[name]) for name in COLUMNS}


class IndexStore:
    """Reads and writes `index.csv`."""

    def path(self, root: Path) -> Path:
        """The index file of a registry folder."""
        return root / INDEX_NAME

    def read(self, root: Path) -> list[dict[str, str]]:
        """The rows. Errors: RegistryError (with the fix `rebuild_index()`) when the file is missing, has another header, or a row has the wrong number of cells."""
        path = self.path(root)
        fix = "Run ModelRegistry.rebuild_index() to make it again from the entry cards; it is never rebuilt silently."
        if not path.is_file():
            raise RegistryError(f"The registry index {path} is missing. {fix}")
        try:
            rows = [r for r in csv.reader(io.StringIO(path.read_text(encoding="utf-8-sig"), newline="")) if r]  # blank lines (an editor added them) are skipped
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            raise RegistryError(f"The registry index {path} cannot be read ({type(exc).__name__}: {exc}). {fix}") from exc
        if not rows or tuple(rows[0]) != COLUMNS:
            raise RegistryError(f"The registry index {path} has an unexpected header (expected {', '.join(COLUMNS[:3])}, ...). {fix}")
        for number, row in enumerate(rows[1:], 2):
            if len(row) != len(COLUMNS):
                raise RegistryError(f"Line {number} of the registry index {path} has {len(row)} cells instead of {len(COLUMNS)}. {fix}")
        return [dict(zip(COLUMNS, row, strict=True)) for row in rows[1:]]

    def write(self, root: Path, rows: list[dict[str, str]]) -> Path:
        """Write the whole index (temporary file and rename)."""
        target = self.path(root)
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\r\n")
        writer.writerow(COLUMNS)
        writer.writerows([[row[c] for c in COLUMNS] for row in rows])
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text("﻿" + buffer.getvalue(), encoding="utf-8", newline="")
        temporary.replace(target)
        return target
