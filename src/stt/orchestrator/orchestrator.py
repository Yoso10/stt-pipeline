"""Orchestrator: runs the stages of a plan on one experiment, with completion markers, the exit codes and the stop flag.

How to run: from Python (PowerShell, project root; the command line `python -m stt ...` is the normal way, see docs/COMPONENTS.md, C13):
    $env:PYTHONPATH = "src"
    python -c "from stt.orchestrator import default_stage_registry; print(default_stage_registry().names())"
`Orchestrator(config, services, registry, stop_flag, sleep_guard).run(RunPlan(stages, experiment))` -> `RunOutcome`.
Rules: the unknown stage names are checked before anything runs; the experiment lock is held and sleep is prevented while the stages run; each stage
that finishes writes a completion marker (`status.json`, with a fingerprint of its config slice and of the stages it depends on) and its result
(`reports/stage_results/<stage>.json`); a rerun skips a stage whose marker is valid, and a stage whose slice changed runs again together with every later
stage; the first failed stage stops the run (the later stages are listed as skipped), its exit code comes from `services.exit_code_for`, and the status
becomes `failed` with the reason; a requested stop (the stop flag, or a second Ctrl+C) leaves no marker, sets the status `stopped` and gives exit code 130.
There is no automatic retry. The verdict of the outcome is the Selector's (`winner` or `baseline_retained`) when the `select_and_register` stage is in the plan, otherwise `n/a`. The module names no concrete component class.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from typing import Any

from stt.config import AppConfig
from stt.orchestrator.errors import OrchestratorError, StageInterrupted
from stt.orchestrator.planned import announce_plan
from stt.orchestrator.sleepguard import NoopSleepGuard, SleepGuard
from stt.orchestrator.stageio import load_result, save_result
from stt.orchestrator.stages import StageRegistry
from stt.orchestrator.types import RunOutcome, RunPlan, Services, StageContext, StageResult, StopFlag
from stt.runmanager import ExperimentHandle

logger = logging.getLogger("stt.orchestrator")
EXIT_OK = 0
EXIT_INTERRUPTED = 130


def fingerprint(stage: str, config_slice: Mapping[str, Any], dependencies: list[str | None]) -> str:
    """A short hash of a stage's name, its config slice and the fingerprints of the stages it depends on."""
    text = json.dumps({"stage": stage, "slice": config_slice, "after": dependencies}, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class Orchestrator:
    """Runs the stages of a plan."""

    def __init__(self, config: AppConfig, services: Services, registry: StageRegistry, stop_flag: StopFlag, sleep_guard: SleepGuard | None = None, skip_completed: bool = True) -> None:
        self._config = config
        self._services = services
        self._registry = registry
        self._stop = stop_flag
        self._sleep = sleep_guard or NoopSleepGuard()
        self._skip_completed = skip_completed

    def run(self, plan: RunPlan) -> RunOutcome:
        """Run the plan and return the outcome (the caller prints the final JSON). Errors: OrchestratorError for an unknown stage;
        RunManagerError when the experiment is locked or its status file is damaged (both before any stage runs)."""
        stages = [self._registry.create(name) for name in plan.stages]  # unknown names fail here, before anything runs
        handle = plan.experiment
        results: dict[str, StageResult] = {}
        shared: dict[str, Any] = {}  # data prepared once and used by the stages of this run
        with self._sleep, handle.lock():
            handle.set_status("running")
            fingerprints: dict[str, str] = {}
            invalidated = False
            for index, stage in enumerate(stages):
                if self._stop():
                    return self._stopped(handle, results, f"stop requested before stage {stage.name}")
                dependencies = [fingerprints.get(d) or self._stored_fingerprint(handle, d) for d in stage.depends_on]
                current = fingerprint(stage.name, stage.config_slice(self._config), dependencies)
                mark = handle.stage_mark(stage.name)
                stored = load_result(handle, stage.name)
                if self._skip_completed and not invalidated and mark is not None and mark.status == "ok" and mark.fingerprint == current and stored is not None:
                    logger.info("Stage %s is already done with the same settings: skipped.", stage.name)
                    results[stage.name] = StageResult("skipped", stored.summary, stored.artifacts)
                    fingerprints[stage.name] = current
                    if stage.name == "prepare":
                        self._announce(plan, handle, results, shared)
                    continue
                invalidated = True
                logger.info("Stage %s starts.", stage.name)
                ctx = StageContext(self._config, handle, self._services, self._stop, dict(results), dict(plan.options), shared=shared)
                try:
                    result = stage.run(ctx)
                except (StageInterrupted, KeyboardInterrupt) as exc:
                    return self._stopped(handle, results, f"interrupted during stage {stage.name}: {exc}".rstrip(": "))
                except Exception as exc:  # every failure of a stage is reported with its exit code; nothing is retried
                    code = exc.exit_code if isinstance(exc, OrchestratorError) else self._services.exit_code_for(exc)
                    message = f"stage {stage.name} failed: {type(exc).__name__}: {exc}"
                    logger.error(message)
                    handle.mark_stage(stage.name, "failed", current)
                    handle.set_status("failed", message)
                    results[stage.name] = StageResult("failed", {}, [], str(exc))
                    for later in stages[index + 1 :]:
                        results[later.name] = StageResult("skipped", {}, [], f"not run because stage {stage.name} failed")
                    return RunOutcome(code, "n/a", handle.experiment_id, results, "failed", message)
                save_result(handle, stage.name, result)
                handle.mark_stage(stage.name, "ok", current)
                fingerprints[stage.name] = current
                results[stage.name] = result
                logger.info("Stage %s finished.", stage.name)
                if stage.name == "prepare":
                    self._announce(plan, handle, results, shared)
                if self._stop() and index + 1 < len(stages):
                    return self._stopped(handle, results, f"stop requested after stage {stage.name}")
            if handle.status().status != "completed" or any(r.status == "ok" for r in results.values()):
                handle.set_status("completed")
        selected = results.get("select_and_register")
        verdict = str(selected.summary.get("verdict", "n/a")) if selected is not None else "n/a"  # the Selector's verdict once the selection stage ran
        return RunOutcome(EXIT_OK, verdict, handle.experiment_id, results, "completed")

    def _announce(self, plan: RunPlan, handle: ExperimentHandle, results: dict[str, StageResult], shared: dict[str, Any]) -> None:
        """Right after `prepare`: write the planned total time of `run`, `search` or `learning-curve` (`plan.options['announce_plan']`; DECISIONS 78, O114). A problem is logged and never stops the run."""
        kind = plan.options.get("announce_plan")
        if not kind:
            return
        try:
            announce_plan(StageContext(self._config, handle, self._services, self._stop, dict(results), dict(plan.options), shared=shared), kind)
        except Exception as exc:  # noqa: BLE001  the planned time is information only
            logger.warning("The planned time could not be worked out (%s: %s); the run goes on.", type(exc).__name__, exc)

    @staticmethod
    def _stored_fingerprint(handle: ExperimentHandle, stage: str) -> str | None:
        mark = handle.stage_mark(stage)
        return None if mark is None else mark.fingerprint

    @staticmethod
    def _stopped(handle: ExperimentHandle, results: dict[str, StageResult], reason: str) -> RunOutcome:
        logger.warning("The run stopped: %s. Run the same command with --experiment %s to continue.", reason, handle.experiment_id)
        handle.set_status("stopped", reason)
        return RunOutcome(EXIT_INTERRUPTED, "n/a", handle.experiment_id, results, "stopped", reason)
