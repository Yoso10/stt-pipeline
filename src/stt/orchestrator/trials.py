"""SearchTrialRunner: runs ONE trial of the hyperparameter search (a fresh model, the screening data, the trial's hyperparameters).

How to run: this module is not a script; the `search` stage builds the runner and the HyperparamSearcher calls it once per trial. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.orchestrator.trials import SearchTrialRunner; print(SearchTrialRunner.__name__)"
Example output:
    SearchTrialRunner
What one call does: loads a fresh model, builds the features of the screening Train and Val samples, trains with `apply_params(train.hyperparams, trial.params)` for at most
`train.hyperparams.max_steps` steps with `on_eval=report` (the searcher's pruning and time budget answer through it), the thread limit `trial.threads` and the stop flag, and
returns `TrialOutcome(val_loss=best Val loss, steps, seconds, model_dir)`. The trial's folder is `candidates/exp_NNNN_trial_TTT` (the model folder inside it is
`exp_NNNN_trial_TTT_step_SSSSS`), its metrics are in `metrics/trial_TTT/`. Errors: a failing training raises (the searcher records a failed trial and goes on);
`SearchInterrupted` when the stop flag stopped the training (a checkpoint was saved; a rerun resumes this trial). The runner is PICKLABLE (it travels to a worker process when
`search.parallel_trials` is above 1): the services and the stop flag are not sent, the worker rebuilds the services from the stored `config_full.yaml` and installs its own Ctrl+C
handler. The runner names no concrete component class (`services_provider` comes from the composition root).
Interrupted trials: a stored `search/trial_TTT.params.json` says which parameters the folder was started with. A rerun resumes the folder only when the parameters are the same
and a complete checkpoint exists; otherwise the old folder is RENAMED to `..._stale_N` (never deleted) and the trial starts fresh.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from stt.orchestrator.stopflag import InterruptFlag
from stt.orchestrator.types import ScreeningSet, Services, StopFlag
from stt.runmanager import ExperimentHandle, model_name_prefix
from stt.searcher import SearchInterrupted, Trial, TrialOutcome, apply_params
from stt.textnormalizer import NormalizedSample
from stt.trainercore import StepMetrics, TrainRequest

logger = logging.getLogger("stt.orchestrator")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def trial_name(trial_id: int) -> str:
    """`trial_003`: the name of a trial's metrics folder and of its parameter file."""
    return f"trial_{trial_id:03d}"


class SearchTrialRunner:
    """A picklable callable `(trial, report) -> TrialOutcome` (the `TrialRunner` Protocol of the searcher).

    `services` and `stop_flag` are given by the parent process for a sequential search; a worker process gets neither and rebuilds them (`services_provider`).
    """

    def __init__(
        self,
        experiment: ExperimentHandle,
        screening: ScreeningSet,
        services_provider: Callable[[Path], Services],
        services: Services | None = None,
        stop_flag: StopFlag | None = None,
    ) -> None:
        self._experiment = experiment
        self._screening = screening
        self._provider = services_provider
        self._services = services
        self._stop = stop_flag

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_services"] = None  # the services hold models and closures: the worker builds its own
        state["_stop"] = None  # the worker has its own flag
        screening: ScreeningSet = self._screening
        state["_screening"] = replace(screening, train=_sendable(screening.train), val=_sendable(screening.val))
        return state

    # ---- the call ----------------------------------------------------------------------------------------------------
    def __call__(self, trial: Trial, report: Callable[[StepMetrics], bool]) -> TrialOutcome:
        """Run one trial (see the module docstring)."""
        in_worker = self._services is None
        if in_worker:
            self._configure_worker_logging(trial)
            flag = InterruptFlag()
            with flag.installed():
                return self._run(trial, report, self._provider(self._experiment.paths.config_full), flag)
        assert self._services is not None
        return self._run(trial, report, self._services, self._stop or (lambda: False))

    def _run(self, trial: Trial, report: Callable[[StepMetrics], bool], services: Services, stop: StopFlag) -> TrialOutcome:
        if stop():
            raise SearchInterrupted(f"stop requested before trial {trial.trial_id} started")
        experiment, screening = self._experiment, self._screening
        prefix = model_name_prefix(experiment.experiment_id, "trial", trial.trial_id)
        output = experiment.candidate_dir(prefix)
        resume = self._prepare_folder(services, output, trial)
        model = services.models.load()
        features = services.models.features(model)
        train = features.build(screening.train, screening.train_split)
        val = features.build(screening.val, screening.val_split)
        hyperparams = apply_params(services.training.hyperparams(), dict(trial.params))
        request = TrainRequest(
            model=model,
            train=train.dataset,
            val=val.dataset,
            collator=features.build_collator(),
            hyperparams=hyperparams,
            output_dir=output,
            on_eval=report,
            source_config_ref=experiment.config_ref,
            model_name=prefix,
            stop_flag=stop,
        )
        logger.info(
            "Trial %d%s with %s: %d train and %d val examples, %d thread(s).",
            trial.trial_id, " (resumed)" if resume else "", dict(trial.params), len(train.dataset), len(val.dataset), trial.threads,
        )
        result = services.training.train(request, experiment.metrics_dir(trial_name(trial.trial_id)), resume, trial.threads)
        steps = result.history[-1].step if result.history else 0
        if result.stopped_reason == "stopped_by_callback":
            if stop():
                raise SearchInterrupted(f"trial {trial.trial_id} stopped on request at step {steps}; a checkpoint was saved and a rerun resumes it")
            return TrialOutcome("pruned", result.best_val_loss, steps, result.timings.total_seconds, result.model_dir)  # the searcher stopped it (pruned, or the time budget)
        return TrialOutcome("complete", result.best_val_loss, steps, result.timings.total_seconds, result.model_dir)

    # ---- the trial folder ----------------------------------------------------------------------------------------------
    def _prepare_folder(self, services: Services, output: Path, trial: Trial) -> bool:
        return prepare_trial_folder(self._experiment, services, output, f"{trial_name(trial.trial_id)}.params.json", trial.params)

    def _configure_worker_logging(self, trial: Trial) -> None:
        """A worker process has no log setup of its own: its lines go to the console and to `logs/trial_TTT.log` of the experiment."""
        root = logging.getLogger("stt")
        if root.handlers:
            return
        root.setLevel(logging.INFO)
        formatter = logging.Formatter(LOG_FORMAT)
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(formatter)
        root.addHandler(console)
        target = self._experiment.paths.logs / f"{trial_name(trial.trial_id)}.log"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(target, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError:
            root.warning("Cannot write the trial log %s; the lines go to the console only.", target)


def prepare_trial_folder(experiment: ExperimentHandle, services: Services, output: Path, marker_name: str, parameters: Mapping[str, Any]) -> bool:
    """True when the trial or finalist folder `output` resumes an interrupted run. Stores the parameters of this attempt in `search/<marker_name>`; a folder that cannot be
    resumed (other parameters, or no complete checkpoint) is RENAMED to `..._stale_N`, never deleted."""
    marker = experiment.paths.search / marker_name
    params = json.loads(json.dumps(dict(parameters), sort_keys=True))
    resume = False
    if output.exists() and any(output.iterdir()):
        stored = _read_json(marker)
        if stored == params and services.training.can_resume(output):
            resume = True
        else:
            aside = _free_name(output)
            output.rename(aside)
            logger.warning("The folder %s of an earlier, interrupted run cannot be resumed (%s); it was renamed to %s and the run starts fresh.", output, "other parameters" if stored != params else "no complete checkpoint", aside.name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(params, indent=1, sort_keys=True), encoding="utf-8")
    return resume


def _sendable(samples: tuple[NormalizedSample, ...]) -> tuple[NormalizedSample, ...]:
    """The samples with a plain `extra` dict: the loader keeps it as a read-only mapping (`mappingproxy`), which cannot be sent to a worker process."""
    return tuple(replace(n, sample=replace(n.sample, extra=dict(n.sample.extra))) for n in samples)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _free_name(folder: Path) -> Path:
    """`<folder>_stale_1`, `_stale_2`, ... the first one that does not exist."""
    number = 1
    while (candidate := folder.with_name(f"{folder.name}_stale_{number}")).exists():
        number += 1
    return candidate
