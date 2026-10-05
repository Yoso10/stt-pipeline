"""ExperimentStore: creates, opens and lists the numbered experiment folders (standard library, PyYAML).

How to run: from Python (PowerShell, project root; the full working example is in docs/COMPONENTS.md, C12):
    $env:PYTHONPATH = "src"
    python -c "from stt.runmanager import ExperimentStore, RunConfig; print(ExperimentStore(RunConfig()).list_experiments())"
Example output (when no experiment exists yet):
    []
Creating an experiment makes `experiments/exp_NNNN/` with `config_full.yaml` (read-only), `config_diff.yaml`, `run_info.json`, `status.json` and the
folders `logs/`, `metrics/`, `search/`, `candidates/`, `reports/`. Numbers are never reused and never timestamps; the folder is created with an
exclusive `mkdir`, so two processes can never get the same number. Nothing is deleted except by `delete` with explicit paths.
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from stt.config import AppConfig, dump_config
from stt.runmanager import cleanup as cleanup_module
from stt.runmanager.config import RunConfig
from stt.runmanager.diffing import config_tree, defaults_tree, diff_configs, flatten
from stt.runmanager.errors import RunManagerError
from stt.runmanager.handle import ExperimentHandle, atomic_write, read_status, write_status
from stt.runmanager.types import CleanupPolicy, ExperimentPaths, ExperimentSpec, ExperimentSummary, RunInfo, StatusRecord

logger = logging.getLogger("stt.runmanager")
_ID = re.compile(r"^exp_(\d+)$")
WINDOWS_PATH_LIMIT = 260  # characters; longer paths fail on Windows without long-path support
LONGEST_INSIDE = 90  # a generous length of the longest path inside an experiment (a model file in a candidate folder)


def _paths(root: Path) -> ExperimentPaths:
    return ExperimentPaths(
        root=root,
        config_full=root / "config_full.yaml",
        config_diff=root / "config_diff.yaml",
        run_info=root / "run_info.json",
        status=root / "status.json",
        lock=root / "run.lock",
        logs=root / "logs",
        run_log=root / "logs" / "run.log",
        metrics=root / "metrics",
        search=root / "search",
        candidates=root / "candidates",
        reports=root / "reports",
    )


class ExperimentStore:
    """The numbered experiments of one experiments folder (`run.experiments_dir`)."""

    def __init__(self, config: RunConfig) -> None:
        self._config = config
        self._dir = config.experiments_dir

    # ---- create ------------------------------------------------------------------------------------------------
    def create(self, spec: ExperimentSpec) -> ExperimentHandle:
        """Create the next experiment and return its handle (status `created`).

        Errors: RunManagerError when the parent does not exist (the existing experiments are listed), the folder cannot be written, or the
        parent's config cannot be read. A path close to the Windows limit gives a warning that names the path and the fix.
        """
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RunManagerError(f"Cannot create the experiments folder {self._dir} ({exc}). Check the path and that the disk is writable.") from exc
        base = self._base_tree(spec)
        root = self._allocate()
        paths = _paths(root)
        full = dump_config(spec.config)
        try:
            for folder in (paths.logs, paths.metrics, paths.search, paths.candidates, paths.reports):
                folder.mkdir()
        except OSError as exc:
            raise RunManagerError(f"Cannot create the folders of {root} ({exc}). Check that the disk is writable and not full.") from exc
        header = f"# difference from {spec.parent_id}" if spec.parent_id else "# difference from the defaults of the config schemas"
        diff_text = yaml.safe_dump(diff_configs(base, config_tree(spec.config)), sort_keys=False, allow_unicode=True)
        info = RunInfo(root.name, f"{root.name}/{paths.config_full.name}", spec.parent_id, spec.code_version, dict(spec.library_versions), spec.config.runtime.seed, spec.created_at)
        atomic_write(paths.config_full, full.encode("utf-8"))
        if self._config.config_readonly:
            os.chmod(paths.config_full, stat.S_IREAD)
        atomic_write(paths.config_diff, (header + "\n" + diff_text).encode("utf-8"))
        atomic_write(paths.run_info, json.dumps(asdict(info), indent=1, ensure_ascii=False).encode("utf-8"))
        write_status(paths.status, StatusRecord("created", None, {}))
        if len(str(root.resolve())) + LONGEST_INSIDE > WINDOWS_PATH_LIMIT:
            logger.warning(
                "The experiment folder %s is %d characters long; files inside it may exceed the Windows path limit of %d. "
                "Move run.experiments_dir closer to the drive root (for example C:\\stt\\experiments).",
                root,
                len(str(root.resolve())),
                WINDOWS_PATH_LIMIT,
            )
        return ExperimentHandle(root.name, paths, self._config)

    def _allocate(self) -> Path:
        """Take the next free number with an exclusive mkdir (race-safe between threads and processes)."""
        for _ in range(10_000):
            numbers = [int(m.group(1)) for p in self._dir.iterdir() if (m := _ID.match(p.name))]
            root = self._dir / f"exp_{(max(numbers) + 1 if numbers else 1):0{self._config.id_digits}d}"
            try:
                root.mkdir()
                return root
            except FileExistsError:
                time.sleep(0.001)  # another creator took this number: look again
            except OSError as exc:
                raise RunManagerError(f"Cannot create {root} ({exc}). Check that the folder is writable and the disk is not full.") from exc
        raise RunManagerError(f"Could not find a free experiment number in {self._dir} after 10000 tries.")

    def _base_tree(self, spec: ExperimentSpec) -> dict[str, Any]:
        if spec.parent_id is None:
            return defaults_tree(spec.config)
        parent = self._dir / spec.parent_id
        if not (parent / "config_full.yaml").is_file():
            existing = ", ".join(s.experiment_id for s in self.list_experiments()) or "none yet"
            raise RunManagerError(f"The parent experiment '{spec.parent_id}' does not exist in {self._dir}. Existing experiments: {existing}.")
        try:
            loaded = yaml.safe_load((parent / "config_full.yaml").read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise RunManagerError(f"Cannot read the config of the parent {spec.parent_id}: {parent / 'config_full.yaml'} ({exc}).") from exc
        return dict(loaded or {})

    # ---- open --------------------------------------------------------------------------------------------------
    def open(self, experiment_id: str, config: AppConfig | None = None) -> ExperimentHandle:
        """Open an existing experiment (for a resume or for reading).

        With `config` (an addition to the brief) the stored `config_full.yaml` must match it; a mismatch FAILS and shows the first differing keys.
        A key that is new in the current config and equal to its schema default is ignored (O76); every real difference fails.
        Errors: RunManagerError for an unknown experiment (the existing ones are listed), a damaged `status.json`, or a config mismatch.
        """
        root = self._dir / experiment_id
        if not _ID.match(experiment_id) or not root.is_dir():
            existing = ", ".join(s.experiment_id for s in self.list_experiments()) or "none yet"
            raise RunManagerError(f"The experiment '{experiment_id}' does not exist in {self._dir}. Existing experiments: {existing}.")
        paths = _paths(root)
        read_status(paths.status)  # fails with the file name when damaged; nothing is changed
        if config is not None:
            self._check_same_config(paths, config)
        return ExperimentHandle(experiment_id, paths, self._config)

    @staticmethod
    def _check_same_config(paths: ExperimentPaths, config: AppConfig) -> None:
        try:
            stored = yaml.safe_load(paths.config_full.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise RunManagerError(f"Cannot read the stored config {paths.config_full} ({exc}).") from exc
        stored_flat = flatten(stored)
        defaults = flatten(defaults_tree(config))
        # A key that is NEW in the current config (a later version added it) and equals its schema default is not a difference: the experiment ran
        # with exactly that behavior. Any other difference, a changed value, a new key with a non-default value, or a removed key, still fails.
        changes = {key: value for key, value in flatten(diff_configs(stored, config_tree(config))).items() if not (key not in stored_flat and key in defaults and defaults[key] == value)}
        if changes:
            first = [f"{key}: stored {stored_flat.get(key)!r}, now {value!r}" for key, value in list(changes.items())[:5]]
            raise RunManagerError(
                f"Cannot resume {paths.root.name}: the config differs from the stored {paths.config_full.name} ({len(changes)} keys; first: {'; '.join(first)}). "
                "Restore the original settings to continue this experiment, or start a new experiment with the changes (it can name this one as its parent)."
            )

    # ---- list ----------------------------------------------------------------------------------------------------
    def list_experiments(self) -> list[ExperimentSummary]:
        """All experiments, lowest number first. A folder with a damaged status is listed with the status `unreadable` (nothing is changed)."""
        if not self._dir.is_dir():
            return []
        summaries = []
        for folder in sorted(p for p in self._dir.iterdir() if p.is_dir() and _ID.match(p.name)):
            paths = _paths(folder)
            try:
                status = read_status(paths.status).status
            except RunManagerError:
                status = "unreadable"
            try:
                info = json.loads(paths.run_info.read_text(encoding="utf-8"))
                parent, created = info.get("parent_id"), info.get("created_at")
            except (OSError, ValueError, AttributeError):
                parent, created = None, None
            summaries.append(ExperimentSummary(folder.name, status, parent, created))
        return summaries

    # ---- cleanup ---------------------------------------------------------------------------------------------------
    def propose_cleanup(self, policy: CleanupPolicy) -> list[Path]:
        """The folders the policy would remove. Changes nothing."""
        return cleanup_module.propose(self._dir, self.list_experiments(), policy)

    def delete(self, paths: Sequence[Path]) -> list[Path]:
        """Delete exactly these folders (checked first; see `cleanup.delete`). Call it only with an explicit decision, never automatically."""
        def lock_is_fresh(root: Path) -> bool:
            lock = root / "run.lock"
            return lock.exists() and (time.time() - lock.stat().st_mtime) < self._config.lock_stale_minutes * 60.0

        return cleanup_module.delete(self._dir, paths, lock_is_fresh)
