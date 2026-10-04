"""stt.runmanager: numbered experiment folders with config snapshots, status, lock, logs and reports (C12 RunManager).

How to run: from Python (PowerShell, project root; the working example is in docs/COMPONENTS.md, C12):
    $env:PYTHONPATH = "src"
    python -c "from stt.runmanager import model_folder_name; print(model_folder_name('exp_0004', 'baseline'))"
Example output:
    exp_0004_baseline
Read-only helper for humans: `python -m stt.runmanager list|show`. Standard library and PyYAML only; no model code, no network.
"""

from stt.runmanager.cleanup import CONTROL_FILES
from stt.runmanager.config import LogConfig, RunConfig
from stt.runmanager.diffing import config_tree, defaults_tree, diff_configs, flatten
from stt.runmanager.errors import RunManagerError
from stt.runmanager.handle import ExperimentHandle
from stt.runmanager.lock import RunLock
from stt.runmanager.logsetup import LogSetup
from stt.runmanager.naming import model_folder_name, model_name_prefix
from stt.runmanager.store import ExperimentStore
from stt.runmanager.types import (
    STAGE_STATUSES,
    STATUSES,
    CleanupPolicy,
    ExperimentLocations,
    ExperimentPaths,
    ExperimentSpec,
    ExperimentSummary,
    RunInfo,
    StageMark,
    StatusRecord,
)

__all__ = [
    "CONTROL_FILES",
    "STAGE_STATUSES",
    "STATUSES",
    "CleanupPolicy",
    "ExperimentHandle",
    "ExperimentLocations",
    "ExperimentPaths",
    "ExperimentSpec",
    "ExperimentStore",
    "ExperimentSummary",
    "LogConfig",
    "LogSetup",
    "RunConfig",
    "RunInfo",
    "RunLock",
    "RunManagerError",
    "StageMark",
    "StatusRecord",
    "config_tree",
    "defaults_tree",
    "diff_configs",
    "flatten",
    "model_folder_name",
    "model_name_prefix",
]
