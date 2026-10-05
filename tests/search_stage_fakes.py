"""Fake services for the tests of the `search` stage (module-level classes and functions, so that a worker process can import them by name).

How to run: this module is not a script; tests/test_orchestrator_search.py imports it. The fake training is synthetic: its Val loss is a smooth bowl of the learning rate and the
accumulation (tests/search_fakes.py), it writes a checkpoint at every evaluation, answers the pruning callback and the stop flag like the real TrainerCore, and writes what it
was asked into `metrics/trial_TTT/call.json` (the hyperparameters it received, the thread limit, whether it resumed), so that a test can read it even when the trial ran in a
worker process.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

from search_fakes import bowl

from stt.config import load_config
from stt.datasource import AudioSample
from stt.orchestrator import PreparedData, Services
from stt.orchestrator.wiring import SearchWiring, exit_code_for, section_registry
from stt.textnormalizer import NormalizedSample
from stt.trainercore import FreezeReport, StepMetrics, TrainHyperparams, TrainRequest, TrainResult, TrainTimings

SPLIT_SIZES = {"train": 40, "val": 10, "test": 10}


def make_samples(split: str, n: int) -> list[NormalizedSample]:
    """`extra` is a read-only mapping, as the real data source makes it (a mappingproxy cannot be pickled: the runner must send plain copies to a worker)."""
    return [NormalizedSample(AudioSample(f"{split}{i:03d}", Path(f"audio/{split}{i:03d}.wav"), "raw", f"{split}_g{i:03d}", 6.0, None, MappingProxyType({"meta_note": "x"})), "טקסט", "טקסט", None, ()) for i in range(n)]


class FakeData:
    """prepare() returns Train, Val and Test samples without reading anything."""

    def prepare(self, out_dir: Path) -> PreparedData:
        kept = {name: make_samples(name, n) for name, n in SPLIT_SIZES.items()}
        fingerprints = {"data": "d" * 64, "split": "s" * 64, "subsample": None}
        return PreparedData(kept, 1, str, {"load_report.txt": "fake load report"}, {"samples": sum(SPLIT_SIZES.values())}, fingerprints)


class FakeFeatures:
    def __init__(self, requested: list[str]) -> None:
        self.requested = requested

    def build(self, samples: Sequence[NormalizedSample], split: str) -> Any:
        self.requested.append(split)
        return SimpleNamespace(dataset=list(samples), report=None)

    def build_collator(self) -> Callable[[Sequence[Any]], dict[str, object]]:
        return lambda batch: {}


class FakeModels:
    """load() returns a placeholder; features() remembers which splits were asked for (the Test split must never be)."""

    def __init__(self) -> None:
        self.requested: list[str] = []

    def load(self, path: Path | None = None) -> object:
        return SimpleNamespace(path=path)

    def features(self, model: object) -> FakeFeatures:
        return FakeFeatures(self.requested)


def metrics(step: int, loss: float) -> StepMetrics:
    return StepMetrics(step, 0.0, loss, loss, 1e-5, None, 0.1, 0.0, None)


class FakeTraining:
    """A training run in a few lines. `hook(call_number, step)` runs before the evaluation of each step (a test sets the stop flag there); `fail` lists the trial names that raise."""

    EVAL_STEPS = (50, 100, 150)

    def __init__(self, hook: Callable[[int, int], None] | None = None, fail: frozenset[str] = frozenset()) -> None:
        self.hook = hook
        self.fail = fail
        self.calls = 0

    def hyperparams(self) -> TrainHyperparams:
        return TrainHyperparams(1e-5, 150, 4, 4, 10, 0.0, 1.0, 42)

    def can_resume(self, output_dir: Path) -> bool:
        return any((output_dir / "checkpoints").glob("checkpoint-*"))

    def train(self, request: TrainRequest, metrics_dir: Path, resume: bool, threads: int | None = None) -> TrainResult:
        self.calls += 1
        name = request.model_name or ""
        hp = request.hyperparams
        done = sorted(int(p.name.split("-")[1]) for p in (request.output_dir / "checkpoints").glob("checkpoint-*")) if resume else []
        (metrics_dir / "call.json").write_text(
            json.dumps({"learning_rate": hp.learning_rate, "grad_accum_steps": hp.grad_accum_steps, "threads": threads, "resume": resume, "name": name, "n_train": len(request.train), "n_val": len(request.val)}),  # type: ignore[arg-type]
            encoding="utf-8",
        )
        if name in self.fail:
            raise RuntimeError(f"fake failure of {name}")
        final = bowl({"learning_rate": hp.learning_rate, "grad_accum_steps": hp.grad_accum_steps})
        history: list[StepMetrics] = []
        reason = "max_steps"
        best: tuple[int, float] | None = None
        for i, step in enumerate(self.EVAL_STEPS, 1):
            if step <= (done[-1] if done else 0):
                continue
            if self.hook is not None:
                self.hook(self.calls, step)
            loss = final * (1.0 + 1.0 / i)
            history.append(metrics(step, loss))
            (request.output_dir / "checkpoints" / f"checkpoint-{step}").mkdir(parents=True, exist_ok=True)
            best = (step, loss) if best is None or loss < best[1] else best
            stop_for_pruning = request.on_eval is not None and request.on_eval(history[-1])
            if stop_for_pruning or (request.stop_flag is not None and request.stop_flag()):
                reason = "stopped_by_callback"
                break
        steps = history[-1].step if history else done[-1] if done else 0
        for old in request.output_dir.glob(f"{name}_step_*"):  # like the real TrainerCore: the model of an earlier part of a resumed run is replaced
            shutil.rmtree(old)
        model_dir = request.output_dir / f"{name}_step_{steps:05d}"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.safetensors").write_bytes(b"fake")
        timings = TrainTimings(0.5, steps, 0.01, len(history), 0.1, 0.0)
        return TrainResult(model_dir, tuple(history), best[0] if best else steps, best[1] if best else None, reason, timings, FreezeReport("none", 1, 1, 0, ()), 42)  # type: ignore[arg-type]


def fake_provider(path: Path) -> Services:
    """Module-level: a worker process receives it by name. It reads the stored config (as the real provider does) and builds the fakes."""
    config = load_config(path, registry=section_registry(), environ={}, env_file=None, apply_environment=False)
    return fake_services(config)


class FakeSearchWiring(SearchWiring):
    """The real search service (grid, screening) but a worker process rebuilds FAKE services."""

    services_provider = staticmethod(fake_provider)


def fake_services(config: Any, hook: Callable[[int, int], None] | None = None, fail: frozenset[str] = frozenset()) -> Services:
    return Services(FakeData(), FakeModels(), FakeTraining(hook, fail), None, exit_code_for, FakeSearchWiring(config))  # type: ignore[arg-type]
