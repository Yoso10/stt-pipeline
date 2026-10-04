"""ExperimentHandle: the status, stage markers, reports and lock of ONE experiment folder (standard library only).

How to run: this module is not a script; `ExperimentStore.create` and `.open` return handles. Example (PowerShell, project root; the
full working example is in docs/COMPONENTS.md, C12):
    $env:PYTHONPATH = "src"
    python -c "from stt.runmanager import ExperimentStore; print(ExperimentStore)"
A handle writes only inside its own folder: `status.json`, the lock, and `reports/`. It never writes `config_full.yaml` (read-only).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from stt.runmanager.config import RunConfig
from stt.runmanager.errors import RunManagerError
from stt.runmanager.lock import RunLock
from stt.runmanager.types import (
    STAGE_STATUSES,
    STATUSES,
    ExperimentLocations,
    ExperimentPaths,
    RunInfo,
    StageMark,
    StatusRecord,
)


def atomic_write(path: Path, data: bytes) -> None:
    """Write `data` to `path` through a temporary file in the same folder, so a reader never sees a half-written file.

    Errors: RunManagerError naming the path and the cause (folder not writable, disk full).
    """
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    except OSError as exc:
        raise RunManagerError(f"Cannot write {path} ({exc}). Check that the folder is writable and the disk is not full.") from exc


def read_status(path: Path) -> StatusRecord:
    """Read `status.json`. Errors: RunManagerError naming the file when it is missing or damaged (the folder is not touched)."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        stages = {name: StageMark(str(mark["status"]), mark.get("fingerprint")) for name, mark in raw["stages"].items()}
        record = StatusRecord(str(raw["status"]), raw.get("reason"), stages)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise RunManagerError(
            f"The status file {path} is missing or damaged ({type(exc).__name__}: {exc}). The experiment folder was left untouched: "
            "restore the file from a backup, or start a new experiment."
        ) from exc
    if record.status not in STATUSES:
        raise RunManagerError(f"The status file {path} holds the unknown status '{record.status}' (known: {', '.join(STATUSES)}).")
    return record


def write_status(path: Path, record: StatusRecord) -> None:
    """Write `status.json` atomically."""
    body = {"status": record.status, "reason": record.reason, "stages": {n: {"status": m.status, "fingerprint": m.fingerprint} for n, m in record.stages.items()}}
    atomic_write(path, json.dumps(body, indent=1, ensure_ascii=False).encode("utf-8"))


def _single_name(name: str, what: str) -> str:
    if not name or name in (".", "..") or "/" in name or "\\" in name or ":" in name:
        raise RunManagerError(f"Invalid {what} name '{name}': use one plain folder name without slashes (for example exp_0004_trial_003).")
    return name


class ExperimentHandle:
    """One experiment folder: `paths`, the status and stage markers, the report writer and the lock."""

    def __init__(self, experiment_id: str, paths: ExperimentPaths, config: RunConfig) -> None:
        self.experiment_id = experiment_id
        self.paths = paths
        self._config = config

    # ---- status ------------------------------------------------------------------------------------------------
    def status(self) -> StatusRecord:
        """The current content of `status.json`. Errors: RunManagerError when it is damaged."""
        return read_status(self.paths.status)

    def set_status(self, status: str, reason: str | None = None) -> None:
        """Set the status (`created`, `running`, `completed`, `failed`, `stopped`) and the reason; the stage markers are kept."""
        if status not in STATUSES:
            raise RunManagerError(f"Unknown status '{status}'. Known statuses: {', '.join(STATUSES)}.")
        current = self.status()
        write_status(self.paths.status, StatusRecord(status, reason, current.stages))

    def mark_stage(self, stage: str, status: str, fingerprint: str | None = None) -> None:
        """Write the completion marker of `stage` (`ok`, `skipped` or `failed`). `fingerprint` (optional, an addition to the brief) identifies the
        part of the config the stage depends on, so a rerun can decide whether the stage may be skipped."""
        if status not in STAGE_STATUSES:
            raise RunManagerError(f"Unknown stage status '{status}'. Known: {', '.join(STAGE_STATUSES)}.")
        current = self.status()
        write_status(self.paths.status, StatusRecord(current.status, current.reason, {**current.stages, stage: StageMark(status, fingerprint)}))

    def stage_mark(self, stage: str) -> StageMark | None:
        """The marker of `stage`, or None when it has not finished."""
        return self.status().stages.get(stage)

    def run_info(self) -> RunInfo:
        """Read `run_info.json`. Errors: RunManagerError when it is missing or damaged."""
        try:
            raw = json.loads(self.paths.run_info.read_text(encoding="utf-8"))
            return RunInfo(**raw)
        except (OSError, ValueError, TypeError) as exc:
            raise RunManagerError(f"The file {self.paths.run_info} is missing or damaged ({type(exc).__name__}: {exc}).") from exc

    @property
    def config_ref(self) -> str:
        """The value for `TrainRequest.source_config_ref`: the path of `config_full.yaml` relative to the experiments folder."""
        return f"{self.experiment_id}/{self.paths.config_full.name}"

    # ---- folders -----------------------------------------------------------------------------------------------
    def metrics_dir(self, name: str) -> Path:
        """`metrics/<name>/`, created. The composition root gives it to the CSV metrics sink so the live metrics file of a trial is
        `experiments/<id>/metrics/<name>/metrics.csv`."""
        folder = self.paths.metrics / _single_name(name, "metrics folder")
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def candidate_dir(self, name: str) -> Path:
        """`candidates/<name>/` (not created): the output folder of a trial or a finalist. See `model_folder_name` for the names."""
        return self.paths.candidates / _single_name(name, "candidate folder")

    def write_report(self, name: str, content: str | bytes) -> Path:
        """Write `reports/<name>` (text as UTF-8) and return its path. `name` may hold subfolders but may never leave `reports/`.

        Errors: RunManagerError for an absolute path, `..` or any other path that escapes `reports/`.
        """
        reports = self.paths.reports.resolve()
        target = (reports / name).resolve()
        if not name or Path(name).is_absolute() or target == reports or reports not in target.parents:
            raise RunManagerError(f"Refused to write '{name}': a report must stay inside {self.paths.reports}. Use a relative name such as report.html.")
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(target, content.encode("utf-8") if isinstance(content, str) else content)
        return target

    # ---- lock --------------------------------------------------------------------------------------------------
    @contextmanager
    def lock(self) -> Iterator[None]:
        """Hold the experiment lock for the duration of a `with` block (heartbeat file; see RunLock).

        Errors: RunManagerError when the experiment is locked by another run or the lock is stale.
        """
        lock = RunLock(self.paths.lock, self._config.lock_heartbeat_seconds, self._config.lock_stale_minutes)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    # ---- where to read ------------------------------------------------------------------------------------------
    def locations(self) -> ExperimentLocations:
        """The config, status, log, metrics and report files of this experiment, in one object (the files that exist)."""
        p = self.paths
        metrics = tuple(sorted(p.metrics.rglob("*.csv"))) if p.metrics.is_dir() else ()
        metrics += tuple(sorted(p.candidates.rglob("metrics.csv"))) if p.candidates.is_dir() else ()
        reports = tuple(sorted(f for f in p.reports.rglob("*") if f.is_file())) if p.reports.is_dir() else ()
        return ExperimentLocations(p.root, p.config_full, p.config_diff, p.status, p.run_log, metrics, reports)
