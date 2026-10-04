"""Stored results of finished stages (standard library only).

How to run: this module is not a script; the Orchestrator and the stages use it. Each finished stage writes `reports/stage_results/<stage>.json`
through the RunManager (status, summary, artifacts), so that a rerun can print the earlier result of a skipped stage and a later command (for
example `evaluate`) can find what an earlier command (`train`) made. Example (PowerShell, project root; lists the files of an experiment):
    Get-ChildItem experiments/exp_0001/reports/stage_results
"""

from __future__ import annotations

import json
from pathlib import Path

from stt.orchestrator.types import StageResult
from stt.runmanager import ExperimentHandle

FOLDER = "stage_results"


def save_result(handle: ExperimentHandle, name: str, result: StageResult) -> None:
    """Write the result of a stage next to the other reports."""
    body = {"status": result.status, "summary": dict(result.summary), "artifacts": [str(p) for p in result.artifacts], "error": result.error}
    handle.write_report(f"{FOLDER}/{name}.json", json.dumps(body, indent=1, ensure_ascii=False))


def load_result(handle: ExperimentHandle, name: str) -> StageResult | None:
    """The stored result of a stage, or None when there is none (or it cannot be read)."""
    path = handle.paths.reports / FOLDER / f"{name}.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return StageResult(raw["status"], raw["summary"], [Path(p) for p in raw["artifacts"]], raw.get("error"))
    except (OSError, ValueError, KeyError, TypeError):
        return None
