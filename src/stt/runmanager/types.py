"""Typed data objects of the RunManager component.

How to run: this module is not a script; the other modules and later components import it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.runmanager.types import RunInfo; print(list(RunInfo.__dataclass_fields__))"
Example output:
    ['experiment_id', 'config_ref', 'parent_id', 'code_version', 'library_versions', 'seed', 'created_at']

Additions to the brief (DECISIONS 43): `ExperimentPaths.lock` and `.run_log`, `StageMark`, `StatusRecord`, `ExperimentLocations`,
`CleanupPolicy`. Dates are never read from the clock here: `created_at` is passed in.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from stt.config import AppConfig

Status = Literal["created", "running", "completed", "failed", "stopped"]
STATUSES: tuple[str, ...] = ("created", "running", "completed", "failed", "stopped")
StageStatus = Literal["ok", "skipped", "failed"]
STAGE_STATUSES: tuple[str, ...] = ("ok", "skipped", "failed")


@dataclass(frozen=True)
class ExperimentSpec:
    """What a new experiment is made from. `created_at` is passed in (for example an ISO date and time), the component never reads the clock."""

    config: AppConfig
    parent_id: str | None
    code_version: str
    library_versions: Mapping[str, str]
    created_at: str


@dataclass(frozen=True)
class ExperimentPaths:
    """Where everything of one experiment is."""

    root: Path
    config_full: Path
    config_diff: Path
    run_info: Path
    status: Path
    lock: Path
    logs: Path
    run_log: Path
    metrics: Path
    search: Path
    candidates: Path
    reports: Path


@dataclass(frozen=True)
class RunInfo:
    """`run_info.json`. `config_ref` is the path of `config_full.yaml` relative to the experiments folder (for example exp_0004/config_full.yaml);
    it is the value the TrainerCore writes as `source_config_ref` into the model folder."""

    experiment_id: str
    config_ref: str
    parent_id: str | None
    code_version: str
    library_versions: Mapping[str, str]
    seed: int
    created_at: str


@dataclass(frozen=True)
class ExperimentSummary:
    """One line of the experiment list."""

    experiment_id: str
    status: str
    parent_id: str | None
    created_at: str | None


@dataclass(frozen=True)
class StageMark:
    """The completion marker of a stage. `fingerprint` identifies the part of the config the stage depends on (set by the Orchestrator)."""

    status: str
    fingerprint: str | None = None


@dataclass(frozen=True)
class StatusRecord:
    """The content of `status.json`: the status, the reason, and the finished stages."""

    status: str
    reason: str | None
    stages: Mapping[str, StageMark] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentLocations:
    """Everything a human needs to read about one experiment, in one place (`python -m stt.runmanager show exp_0003`)."""

    root: Path
    config_full: Path
    config_diff: Path
    status: Path
    log: Path
    metrics_files: tuple[Path, ...]
    reports: tuple[Path, ...]


@dataclass(frozen=True)
class CleanupPolicy:
    """What a cleanup may PROPOSE. Nothing is selected by default. Dates are compared with `now`, which is passed in.

    `failed_experiments`: whole folders of failed experiments. `candidates_of_completed`: the finalist folders under `candidates/` of completed
    experiments. `checkpoints_of_completed`: only the `checkpoints` folders inside those (the models stay). `older_than_days` with `now`
    restricts every kind to experiments created that many days before `now` or earlier.
    """

    failed_experiments: bool = False
    candidates_of_completed: bool = False
    checkpoints_of_completed: bool = False
    older_than_days: int | None = None
    now: str | None = None
