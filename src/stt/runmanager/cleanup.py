"""Cleanup: PROPOSE what could be deleted, and delete only what is explicitly given (standard library only).

How to run: this module is not a script; `ExperimentStore.propose_cleanup` and `.delete` use it. Rules (owner, LOCKED): nothing is deleted
automatically, ever; a proposal changes nothing; `delete` removes exactly the folders it is given, only inside the experiments folder, never
the experiments folder itself, never a file, never an experiment's own control files, and never a folder of an experiment whose lock is fresh.
The model registry lives outside the experiments folder, so it cannot be reached.
"""

from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from stt.runmanager.errors import RunManagerError
from stt.runmanager.types import CleanupPolicy, ExperimentSummary

CONTROL_FILES = frozenset({"config_full.yaml", "config_diff.yaml", "run_info.json", "status.json", "run.lock"})


def _age_days(created_at: str | None, now: str) -> float | None:
    try:
        return (datetime.fromisoformat(now) - datetime.fromisoformat(created_at or "")).total_seconds() / 86400.0
    except ValueError:
        return None


def propose(experiments_dir: Path, summaries: Sequence[ExperimentSummary], policy: CleanupPolicy) -> list[Path]:
    """The folders the policy would remove. Errors: RunManagerError when `older_than_days` is given without a readable `now`."""
    if policy.older_than_days is not None:
        try:
            datetime.fromisoformat(policy.now or "")
        except ValueError as exc:
            raise RunManagerError("CleanupPolicy.older_than_days needs `now` as an ISO date or date and time (for example 2026-10-05T12:00:00).") from exc
    found: list[Path] = []
    for summary in summaries:
        if policy.older_than_days is not None:
            age = _age_days(summary.created_at, policy.now or "")
            if age is None or age < policy.older_than_days:
                continue
        root = experiments_dir / summary.experiment_id
        if summary.status == "failed" and policy.failed_experiments:
            found.append(root)
        if summary.status == "completed":
            candidates = root / "candidates"
            if policy.candidates_of_completed and candidates.is_dir():
                found.extend(sorted(p for p in candidates.iterdir() if p.is_dir()))
            elif policy.checkpoints_of_completed and candidates.is_dir():
                found.extend(sorted(p for p in candidates.rglob("checkpoints") if p.is_dir()))
    return found


def delete(experiments_dir: Path, paths: Sequence[Path], lock_is_fresh: Callable[[Path], bool]) -> list[Path]:
    """Delete exactly `paths` (folders inside `experiments_dir`) and return what was deleted.

    All paths are checked BEFORE anything is deleted. Errors: RunManagerError for a path outside the experiments folder, the experiments folder
    itself, a file, a missing path, or a folder of an experiment with a fresh lock (nothing is deleted then).
    """
    base = experiments_dir.resolve()
    checked: list[Path] = []
    for path in paths:
        target = path.resolve()
        if target == base or base not in target.parents:
            raise RunManagerError(f"Refused to delete {path}: it is not inside the experiments folder {experiments_dir}.")
        if not target.is_dir():
            raise RunManagerError(f"Refused to delete {path}: it is not an existing folder (cleanup removes folders only, never control files like {sorted(CONTROL_FILES)[0]}).")
        root = base / target.relative_to(base).parts[0]
        if lock_is_fresh(root):
            raise RunManagerError(f"Refused to delete {path}: the experiment {root.name} has a fresh lock (a run is active). Nothing was deleted.")
        checked.append(target)
    for target in checked:
        shutil.rmtree(target, onexc=_make_writable_and_retry)
    return checked


def _make_writable_and_retry(function: Callable[[str], object], path: str, exc: BaseException) -> None:
    os.chmod(path, stat.S_IWRITE)  # read-only files (config copies, model files) must not block a deliberate delete
    function(path)
