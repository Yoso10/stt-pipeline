"""StudyStore: the stored study of a search, as one JSON file (standard library only).

How to run: this module is not a script; the searcher uses it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from pathlib import Path; from stt.searcher.study import StudyStore; print(StudyStore().path(Path('experiments/exp_0001/search')))"
Example output:
    experiments/exp_0001/search/study.json   (printed with the separator of your system)
The file lives in the experiment's `search/` folder and holds the search SPACE (the parameters, the strategy and the seed), and every finished trial
(parameters, status, Val loss, steps, seconds, model folder, error, the Val losses it reported). The parent process writes it after every finished trial
(temporary file and rename), so a stopped search can be resumed: a rerun with the same space continues and the counts add up. A changed space or seed on an
existing study FAILS and names both. The budget (`n_trials`, `time_budget_minutes`) is not part of the space: a resumed search may have a larger budget.
"""

from __future__ import annotations

import json
from pathlib import Path

from stt.searcher.errors import SearchConfigError, SearchError
from stt.searcher.types import SearchSpec, Trial, TrialOutcome, TrialRecord

SCHEMA_VERSION = 1
STUDY_FILE = "study.json"


def space_document(spec: SearchSpec) -> dict[str, object]:
    """The part of the spec that identifies a study (space, strategy, seed)."""
    return {
        "parameters": {n: {"kind": r.kind, "low": r.low, "high": r.high, "choices": list(r.choices), "points": r.points} for n, r in spec.parameters.items()},
        "strategy": spec.strategy,
        "seed": spec.seed,
    }


def _describe(doc: object) -> str:
    return json.dumps(doc, sort_keys=True, ensure_ascii=False)


class StudyStore:
    """Reads and writes `study.json`."""

    def path(self, directory: Path) -> Path:
        """The study file of a folder."""
        return directory / STUDY_FILE

    def load(self, directory: Path, spec: SearchSpec) -> list[TrialRecord]:
        """The stored trials (empty when there is no file).

        Errors: SearchConfigError when the stored space or seed differs from the spec (both are named); SearchError when the file is damaged.
        """
        path = self.path(directory)
        if not path.is_file():
            return []
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            stored_space, rows = document["space"], document["trials"]
            records = [
                TrialRecord(
                    Trial(int(row["trial_id"]), row["params"]),
                    TrialOutcome(row["status"], row["val_loss"], int(row["steps"]), float(row["seconds"]), Path(row["model_dir"]) if row["model_dir"] else None, row["error"], tuple((int(a), float(b)) for a, b in row["reports"])),
                )
                for row in rows
            ]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SearchError(f"The study file {path} cannot be read ({type(exc).__name__}: {exc}). Delete it to start a new search, or use a new experiment folder.") from exc
        if stored_space != json.loads(_describe(space_document(spec))):
            raise SearchConfigError(
                f"The study in {path} was made with another search space, strategy or seed. Stored: {_describe(stored_space)}. Now: {_describe(space_document(spec))}. "
                "Use a new experiment folder, or restore the old settings."
            )
        return records

    def save(self, directory: Path, spec: SearchSpec, records: list[TrialRecord]) -> Path:
        """Write the whole study (temporary file and rename). Errors: SearchError naming the path when it cannot be written."""
        document = {
            "schema_version": SCHEMA_VERSION,
            "space": space_document(spec),
            "trials": [
                {
                    "trial_id": r.trial.trial_id,
                    "params": dict(r.trial.params),
                    "status": r.outcome.status,
                    "val_loss": r.outcome.val_loss,
                    "steps": r.outcome.steps,
                    "seconds": r.outcome.seconds,
                    "model_dir": str(r.outcome.model_dir) if r.outcome.model_dir else None,
                    "error": r.outcome.error,
                    "reports": [list(x) for x in r.outcome.reports],
                }
                for r in records
            ],
        }
        target = self.path(directory)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
            temporary.replace(target)
        except OSError as exc:
            raise SearchError(f"Cannot write the study file {target} ({exc}). Check that the folder is writable and the disk is not full.") from exc
        return target
