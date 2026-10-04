"""Unit tests of the TrainerCore component (stt.trainercore): the acceptance tests of the C6 brief, offline.

How to run (project root, virtual environment active):
    python -m unittest tests.test_trainercore -v
The tests train the stored tiny test model (tests/assets/tiny_whisper: the real tokenizer and feature extractor, random weights of
4 dimensions) on tiny synthetic datasets in a temporary folder, a few steps each. Nothing is downloaded.
"""

from __future__ import annotations

import ast
import csv
import json
import logging
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stt.config import ConfigError, SectionRegistry, load_config  # noqa: E402
from stt.featurepipeline import FeatureExample  # noqa: E402
from stt.featurepipeline.collator import FeatureCollator  # noqa: E402
from stt.modelloader import LoadedModel, ModelConfig, create_loader  # noqa: E402
from stt.trainercore import (  # noqa: E402
    ConsoleMetricsSink,
    CsvMetricsSink,
    EvalMonitor,
    FreezeEncoder,
    NoFreeze,
    SinkContext,
    SinkRegistry,
    StepMetrics,
    TrainConfig,
    TrainerConfigError,
    TrainerCore,
    TrainerCoreError,
    TrainHyperparams,
    TrainRequest,
    TrainResult,
    check_hyperparams,
    default_freeze_registry,
    default_sink_registry,
    hyperparams_from_config,
    resolve_seed,
    resolve_threads,
)
from stt.trainercore.types import RunState  # noqa: E402

TINY = ROOT / "tests" / "assets" / "tiny_whisper"
TEXTS = ["שלום עולם", "מה שלומך היום", "זה מבחן קטן", "עוד משפט קצר", "ועוד אחד", "והאחרון בסדרה"]


def load_model() -> LoadedModel:
    return create_loader(ModelConfig(path=TINY)).load()


class SyntheticDataset:
    """A tiny in-memory FeatureDataset: random features and real label ids."""

    def __init__(self, loaded: LoadedModel, size: int, seed: int = 0, prefix: str = "s") -> None:
        rng = np.random.default_rng(seed)
        self._items = [
            FeatureExample(f"{prefix}{i}", rng.standard_normal((80, 3000)).astype(np.float32) * 0.1, loaded.tokenizer.encode_label(TEXTS[i % len(TEXTS)]))
            for i in range(size)
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> FeatureExample:
        return self._items[index]


def conf(**overrides: Any) -> TrainConfig:
    """A fast config: log every step, evaluate every 2, save every 2, one thread, the console sink off."""
    base: dict[str, Any] = {
        "hyperparams": {"max_steps": 4, "batch_size": 2, "grad_accum_steps": 1, "warmup_steps": 1, "learning_rate": 1e-3},
        "log_every_steps": 1,
        "eval_every_steps": 2,
        "save_every_steps": 2,
        "cpu_threads": 1,
        "sinks": ["csv"],
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return TrainConfig.model_validate(base)


class TrainerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.out = self.tmp / "run"

    def request(self, config: TrainConfig, loaded: LoadedModel | None = None, **kwargs: Any) -> TrainRequest:
        loaded = loaded or load_model()
        values: dict[str, Any] = {
            "model": loaded,
            "train": SyntheticDataset(loaded, 8, seed=1),
            "val": SyntheticDataset(loaded, 4, seed=2, prefix="v"),
            "collator": FeatureCollator(decoder_start_token_id=int(loaded.model.config.decoder_start_token_id)),  # type: ignore[attr-defined]
            "hyperparams": hyperparams_from_config(config, 42),
            "output_dir": self.out,
        }
        values.update(kwargs)
        return TrainRequest(**values)

    def run_training(self, config: TrainConfig | None = None, **kwargs: Any) -> TrainResult:
        config = config or conf()
        return TrainerCore(config).train(self.request(config, **kwargs))

    def csv_rows(self) -> list[dict[str, str]]:
        with (self.out / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


class ShortRunTests(TrainerTestCase):
    def test_1_short_run_returns_a_result_and_the_folder_loads_through_the_model_loader(self) -> None:
        result = self.run_training(source_config_ref="exp_0001/config.yaml")  # type: ignore[arg-type]
        self.assertEqual((result.stopped_reason, result.history[-1].step, result.seed), ("max_steps", 4, 42))
        self.assertEqual(result.model_dir, self.out / "model")
        for name in ("model.safetensors", "config.json", "tokenizer.json", "preprocessor_config.json", "pipeline_model_info.json"):
            self.assertTrue((result.model_dir / name).is_file(), name)
        loaded = create_loader(ModelConfig(path=result.model_dir)).load()
        self.assertEqual(loaded.info.source_config_ref, "exp_0001/config.yaml")
        base = load_model()
        self.assertEqual(loaded.info.num_parameters, base.info.num_parameters)
        changed = any(not torch.equal(a, b) for a, b in zip(base.model.parameters(), loaded.model.parameters()))  # type: ignore[attr-defined]
        self.assertTrue(changed, "training did not change any weight")
        info = json.loads((result.model_dir / "pipeline_model_info.json").read_text(encoding="utf-8"))
        self.assertEqual((info["base_model"], info["seed"], info["stopped_reason"]), ("tiny_whisper", 42, "max_steps"))

    def test_2_no_network_during_a_run(self) -> None:
        with mock.patch("socket.socket.connect", side_effect=AssertionError("network used")):
            self.run_training()

    def test_3_live_metrics_file_grows_during_the_run_and_every_record_has_all_fields(self) -> None:
        seen: list[int] = []
        out = self.out

        class Probe:
            def __init__(self, context: SinkContext) -> None:
                pass

            def write(self, record: StepMetrics) -> None:
                with (out / "metrics.csv").open(newline="", encoding="utf-8") as handle:
                    seen.append(len(list(csv.reader(handle))) - 1)  # rows already on disk, the header excluded

        registry = SinkRegistry({"csv": CsvMetricsSink, "probe": Probe})
        config = conf(sinks=["csv", "probe"])
        result = TrainerCore(config, sink_registry=registry).train(self.request(config))
        self.assertGreater(len(seen), 2)
        self.assertEqual(seen, list(range(1, len(seen) + 1)))  # grew by one row at every record, during the run
        self.assertEqual(len(self.csv_rows()), len(result.history))
        columns = set(StepMetrics.__dataclass_fields__)
        self.assertEqual(set(self.csv_rows()[0]), columns)
        first = result.history[0]
        self.assertEqual(first.step, 1)  # a record at step 1: the first live line
        self.assertGreater(first.seconds_per_step, 0)
        self.assertTrue(all(r.elapsed_sec >= 0 for r in result.history))

    def test_4_val_loss_every_eval_interval_and_no_generation(self) -> None:
        loaded = load_model()
        loaded.model.generate = mock.Mock(side_effect=AssertionError("generate was called"))  # type: ignore[attr-defined]
        result = self.run_training(conf(hyperparams={"max_steps": 6}, eval_every_steps=3), loaded=loaded)
        evals = [r.step for r in result.history if r.val_loss is not None]
        self.assertEqual(evals, [3, 6])
        loaded.model.generate.assert_not_called()  # type: ignore[attr-defined]
        self.assertEqual(result.timings.eval_runs, 2)

    def test_5_fewer_steps_than_one_evaluation_still_evaluates_once_at_the_end(self) -> None:
        result = self.run_training(conf(hyperparams={"max_steps": 3}, eval_every_steps=50, save_every_steps=100))
        self.assertEqual([r.step for r in result.history if r.val_loss is not None], [3])
        self.assertEqual(result.best_step, 3)
        self.assertIsNotNone(result.best_val_loss)

    def test_6_a_failing_sink_does_not_stop_the_training(self) -> None:
        class Broken:
            def __init__(self, context: SinkContext) -> None:
                pass

            def write(self, record: StepMetrics) -> None:
                raise RuntimeError("sink broke")

        registry = SinkRegistry({"csv": CsvMetricsSink, "broken": Broken})
        config = conf(sinks=["broken", "csv"])
        with self.assertLogs("stt.trainercore", level="ERROR"):
            result = TrainerCore(config, sink_registry=registry).train(self.request(config))
        self.assertEqual(result.stopped_reason, "max_steps")
        self.assertEqual(len(self.csv_rows()), len(result.history))


class StopTests(TrainerTestCase):
    def test_early_stopping_stops_at_the_expected_step_and_returns_the_best_checkpoint(self) -> None:
        # min_delta so large that nothing after the first evaluation counts as an improvement: with patience 2 the run
        # stops at the third evaluation (steps 2, 4, 6) and the first evaluation (step 2) is the best.
        config = conf(hyperparams={"max_steps": 20}, early_stopping={"patience": 2, "min_delta": 1e9})
        with self.assertLogs("stt.trainercore", level="WARNING") as logs:
            result = self.run_training(config)
        self.assertEqual((result.stopped_reason, result.best_step, result.history[-1].step), ("early_stopping", 2, 6))
        self.assertTrue(any("did not improve after the first evaluation" in line for line in logs.output))
        info = json.loads((result.model_dir / "pipeline_model_info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["best_step"], 2)
        steps = sorted(int(p.name.split("-")[1]) for p in (self.out / "checkpoints").iterdir())
        self.assertEqual(steps, [2, 6])  # only the best (2) and the latest (6) are kept

    def test_on_eval_callback_stops_the_run(self) -> None:
        calls: list[int] = []

        def stop_at_four(record: StepMetrics) -> bool:
            calls.append(record.step)
            return record.step >= 4

        result = self.run_training(conf(hyperparams={"max_steps": 10}), on_eval=stop_at_four)
        self.assertEqual((result.stopped_reason, calls), ("stopped_by_callback", [2, 4]))
        self.assertEqual(result.history[-1].step, 4)

    def test_time_cap_stops_gracefully_with_a_saved_model(self) -> None:
        ticks = iter(range(0, 10_000, 10))  # every clock call is 10 s later
        config = conf(hyperparams={"max_steps": 50}, max_runtime_minutes=1.0, eval_every_steps=1000, save_every_steps=1000)
        result = TrainerCore(config, clock=lambda: float(next(ticks))).train(self.request(config))
        self.assertEqual(result.stopped_reason, "time_limit")
        self.assertLess(result.history[-1].step, 50)
        self.assertTrue((result.model_dir / "model.safetensors").is_file())  # saved although no evaluation ran
        self.assertIsNone(result.best_val_loss)
        self.assertEqual(result.best_step, result.history[-1].step)

    def test_nan_loss_stops_keeps_the_last_good_checkpoint_and_fails(self) -> None:
        loaded = load_model()
        real = self.request(conf(hyperparams={"max_steps": 8}), loaded=loaded)
        calls = {"n": 0}

        def poisoned(examples: Sequence[FeatureExample]) -> Any:
            batch = real.collator(examples)
            if examples[0].sample_id.startswith("s"):  # a Train batch (the Val ids start with "v")
                calls["n"] += 1
            if calls["n"] >= 4:  # the Trainer reads one batch ahead, so from the third step on the features are NaN
                batch["input_features"] = torch.full_like(batch["input_features"], float("nan"))  # type: ignore[call-overload]
            return batch

        with self.assertRaises(TrainerCoreError) as ctx:
            TrainerCore(conf(hyperparams={"max_steps": 8})).train(replace(real, collator=poisoned))
        message = str(ctx.exception)
        self.assertRegex(message, r"at step [345]\.")  # the exact step depends on how far ahead the Trainer reads batches
        self.assertIn("learning rate", message)
        self.assertIn("checkpoint-2", message)
        self.assertFalse((self.out / "model").exists())
        self.assertEqual(sorted(p.name for p in (self.out / "checkpoints").iterdir()), ["checkpoint-2"])  # nothing bad was saved

    def test_out_of_memory_fails_with_a_hint(self) -> None:
        with mock.patch("transformers.Seq2SeqTrainer.train", side_effect=RuntimeError("DefaultCPUAllocator: not enough memory")):
            with self.assertRaises(TrainerCoreError) as ctx:
                self.run_training()
        self.assertIn("batch_size", str(ctx.exception))
        self.assertIn("freeze_encoder", str(ctx.exception))


class ResumeTests(TrainerTestCase):
    def test_resume_continues_from_the_saved_step_without_duplicate_rows(self) -> None:
        first = self.run_training(conf(hyperparams={"max_steps": 6}), on_eval=lambda r: r.step >= 4)
        self.assertEqual((first.stopped_reason, first.history[-1].step), ("stopped_by_callback", 4))
        second = self.run_training(conf(hyperparams={"max_steps": 6}, resume=True))
        self.assertEqual((second.stopped_reason, second.history[-1].step), ("max_steps", 6))
        steps = [int(r["step"]) for r in self.csv_rows()]
        self.assertEqual(steps, sorted(steps))
        self.assertEqual(len(steps), len(set(steps)) + sum(1 for s in set(steps) if steps.count(s) == 2))  # a step may appear twice only as a train and an eval row
        self.assertEqual([r.step for r in second.history][:len(first.history)], [r.step for r in first.history])
        self.assertTrue((second.model_dir / "model.safetensors").is_file())

    def test_a_resumed_run_has_exactly_the_losses_of_an_uninterrupted_run(self) -> None:
        self.out = self.tmp / "full"
        full = self.run_training(conf(hyperparams={"max_steps": 6}))
        self.out = self.tmp / "split"
        self.run_training(conf(hyperparams={"max_steps": 6}), on_eval=lambda r: r.step >= 4)
        resumed = self.run_training(conf(hyperparams={"max_steps": 6}, resume=True))
        losses = lambda result: [(r.step, r.train_loss, r.val_loss) for r in result.history]  # noqa: E731
        self.assertEqual(losses(resumed), losses(full))

    def test_resume_with_a_different_configuration_fails_naming_both(self) -> None:
        self.run_training(conf(hyperparams={"max_steps": 4}), on_eval=lambda r: True)
        with self.assertRaises(TrainerConfigError) as ctx:
            self.run_training(conf(hyperparams={"max_steps": 4, "learning_rate": 5e-4}, resume=True))
        message = str(ctx.exception)
        self.assertIn("learning_rate", message)
        self.assertIn("0.001", message)
        self.assertIn("0.0005", message)

    def test_resume_without_a_checkpoint_fails(self) -> None:
        with self.assertRaises(TrainerConfigError) as ctx:
            self.run_training(conf(resume=True))
        self.assertIn("no complete checkpoint", str(ctx.exception))

    def test_a_non_empty_output_folder_is_not_overwritten(self) -> None:
        self.out.mkdir()
        (self.out / "keep.txt").write_text("mine", encoding="utf-8")
        with self.assertRaises(TrainerConfigError) as ctx:
            self.run_training()
        self.assertIn("never overwritten", str(ctx.exception))
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["keep.txt"])
        self.assertEqual((self.out / "keep.txt").read_text(encoding="utf-8"), "mine")

    def test_an_empty_existing_output_folder_is_accepted(self) -> None:
        self.out.mkdir()
        self.assertEqual(self.run_training().stopped_reason, "max_steps")

    def test_csv_sink_removes_rows_after_the_resumed_step(self) -> None:
        sink = CsvMetricsSink(SinkContext(self.tmp))
        for step in (1, 2, 3, 4):
            sink.write(StepMetrics(step, 0.1, 1.0, None, 1e-5, 0.5, 0.2, 1.0, None))
        CsvMetricsSink(SinkContext(self.tmp, resume_step=2))
        rows = list(csv.DictReader((self.tmp / "metrics.csv").open(newline="", encoding="utf-8")))
        self.assertEqual([r["step"] for r in rows], ["1", "2"])
        self.assertEqual(rows[0]["val_loss"], "")  # None is an empty cell


class FreezeAndThreadTests(TrainerTestCase):
    def test_freeze_encoder_freezes_exactly_the_encoder_parameters_and_none_freezes_nothing(self) -> None:
        loaded = load_model()
        parameters = dict(loaded.model.named_parameters())  # type: ignore[attr-defined]
        encoder = {n: p for n, p in parameters.items() if n.startswith("model.encoder.")}
        encoder_count = sum(p.numel() for p in encoder.values())
        total = sum(p.numel() for p in parameters.values())
        report = FreezeEncoder().apply(loaded.model)
        self.assertEqual((report.frozen_params, report.trainable_params, report.total_params), (encoder_count, total - encoder_count, total))
        self.assertTrue(all(not p.requires_grad for p in encoder.values()))
        self.assertTrue(all(p.requires_grad for n, p in parameters.items() if n not in encoder))
        none_report = NoFreeze().apply(loaded.model)  # makes everything trainable again
        self.assertEqual((none_report.frozen_params, none_report.frozen_groups, none_report.trainable_params), (0, (), total))

    def test_run_with_freeze_encoder_keeps_the_encoder_unchanged(self) -> None:
        loaded = load_model()
        before = {n: p.detach().clone() for n, p in loaded.model.named_parameters() if n.startswith("model.encoder.")}  # type: ignore[attr-defined]
        result = self.run_training(conf(freeze_policy="freeze_encoder"), loaded=loaded)
        self.assertGreater(result.freeze_report.frozen_params, 0)
        self.assertTrue(all(torch.equal(before[n], p) for n, p in loaded.model.named_parameters() if n in before))  # type: ignore[attr-defined]

    def test_the_thread_limit_is_applied_during_the_run_and_restored_after(self) -> None:
        seen: list[int] = []

        class Probe:
            def __init__(self, context: SinkContext) -> None:
                pass

            def write(self, record: StepMetrics) -> None:
                seen.append(torch.get_num_threads())

        before = torch.get_num_threads()
        registry = SinkRegistry({"csv": CsvMetricsSink, "probe": Probe})
        config = conf(sinks=["csv", "probe"], cpu_threads=2)
        TrainerCore(config, sink_registry=registry).train(self.request(config))
        self.assertEqual(set(seen), {2})
        self.assertEqual(torch.get_num_threads(), before)

    def test_default_threads_are_half_of_the_cores(self) -> None:
        with mock.patch("os.cpu_count", return_value=8):
            self.assertEqual(resolve_threads(TrainConfig()), 4)
        with mock.patch("os.cpu_count", return_value=1):
            self.assertEqual(resolve_threads(TrainConfig()), 1)
        self.assertEqual(resolve_threads(TrainConfig(cpu_threads=3)), 3)


class SeedAndFailureTests(TrainerTestCase):
    def test_same_seed_gives_the_same_loss_history(self) -> None:
        def losses(folder: str) -> list[tuple[int, float | None, float | None]]:
            self.out = self.tmp / folder
            result = self.run_training(conf(hyperparams={"seed": 7}))
            return [(r.step, r.train_loss, r.val_loss) for r in result.history]

        first, second = losses("a"), losses("b")
        self.assertEqual(first, second)
        self.assertGreater(len(first), 3)

    def test_unknown_policy_and_sink_fail_and_list_the_registered_names(self) -> None:
        with self.assertRaises(TrainerConfigError) as ctx:
            self.run_training(conf(freeze_policy="lora"))
        self.assertIn("freeze_encoder, none", str(ctx.exception))
        self.assertFalse(self.out.exists())  # nothing was created
        with self.assertRaises(TrainerConfigError) as ctx2:
            self.run_training(conf(sinks=["csv", "tensorboard"]))
        self.assertIn("console, csv", str(ctx2.exception))
        self.assertFalse(self.out.exists())

    def test_empty_datasets_fail(self) -> None:
        loaded = load_model()
        for field in ("train", "val"):
            with self.assertRaises(TrainerCoreError) as ctx:
                self.run_training(loaded=loaded, **{field: SyntheticDataset(loaded, 0)})
            self.assertIn("empty", str(ctx.exception))

    def test_impossible_hyperparameters_fail_with_all_problems(self) -> None:
        good = hyperparams_from_config(conf(), 1)
        with self.assertRaises(TrainerConfigError) as ctx:
            check_hyperparams(replace(good, learning_rate=0.0, batch_size=0, warmup_steps=99))
        message = str(ctx.exception)
        for part in ("learning_rate", "batch_size", "warmup_steps"):
            self.assertIn(part, message)
        check_hyperparams(good)


class MonitorAndConfigTests(unittest.TestCase):
    def test_monitor_counts_evaluations_without_improvement(self) -> None:
        monitor = EvalMonitor(patience=2, min_delta=0.0, state=RunState())
        verdicts = [monitor.record(s, v) for s, v in [(10, 1.0), (20, 0.9), (30, 0.95), (40, 0.9), (50, 0.89)]]
        self.assertEqual([v.improved for v in verdicts], [True, True, False, False, True])  # 0.9 equals the best: not an improvement
        self.assertEqual([v.stop for v in verdicts], [False, False, False, True, False])
        self.assertEqual((monitor.state.best_step, monitor.state.best_val_loss), (50, 0.89))

    def test_monitor_min_delta_and_first_evaluation_flag(self) -> None:
        monitor = EvalMonitor(patience=1, min_delta=0.1, state=RunState())
        self.assertFalse(monitor.record(10, 1.0).stop)
        verdict = monitor.record(20, 0.95)  # better, but not by more than 0.1
        self.assertEqual((verdict.improved, verdict.stop, verdict.never_improved_after_first), (False, True, True))
        with self.assertRaises(ValueError):
            monitor.record(30, float("nan"))

    def test_config_defaults_and_validation(self) -> None:
        train = TrainConfig()
        h = train.hyperparams
        self.assertEqual((train.freeze_policy, train.log_every_steps, train.eval_every_steps, train.save_every_steps), ("none", 10, 50, 100))
        self.assertEqual((h.learning_rate, h.batch_size, h.max_grad_norm), (1e-5, 4, 1.0))
        self.assertEqual((train.early_stopping.patience, train.early_stopping.min_delta, train.max_runtime_minutes), (5, 0.0, None))
        self.assertEqual((train.cpu_threads, train.resume, train.sinks), (None, False, ["csv", "console"]))
        for bad in ({"sinks": []}, {"log_every_steps": 0}, {"hyperparams": {"learning_rate": 0}}, {"unknown_key": 1}, {"max_runtime_minutes": 0}):
            with self.assertRaises(ValueError):
                TrainConfig.model_validate(bad)

    def test_seed_falls_back_to_the_runtime_seed(self) -> None:
        self.assertEqual(resolve_seed(TrainConfig().hyperparams, 42), 42)
        self.assertEqual(hyperparams_from_config(TrainConfig.model_validate({"hyperparams": {"seed": 5}}), 42).seed, 5)

    def test_yaml_example_loads_and_the_section_registers(self) -> None:
        config = load_config(ROOT / "configs" / "train_default.yaml", env_file=None, target_environ={}, registry=SectionRegistry({"train": TrainConfig}))
        self.assertEqual(config.section("train", TrainConfig), TrainConfig())  # the YAML shows exactly the defaults
        with self.assertRaises(ConfigError):
            load_config(ROOT / "configs" / "train_default.yaml", overrides=["train.sinks=[]"], env_file=None, target_environ={}, registry=SectionRegistry({"train": TrainConfig}))

    def test_registries_list_their_names(self) -> None:
        self.assertEqual(default_freeze_registry().names(), ["freeze_encoder", "none"])
        self.assertEqual(default_sink_registry().names(), ["console", "csv"])

    def test_console_sink_logs_one_line_per_record(self) -> None:
        sink = ConsoleMetricsSink(SinkContext(Path(".")))
        with self.assertLogs("stt.trainercore", level="INFO") as logs:
            sink.write(StepMetrics(10, 0.5, 2.5, None, 1e-5, None, 0.3, 4.0, 27.0))
        self.assertEqual(len(logs.output), 1)
        self.assertIn("step 10", logs.output[0])
        self.assertIn("train_loss 2.5000", logs.output[0])


class ImportRuleTests(unittest.TestCase):
    def test_only_the_adapter_module_imports_torch_and_transformers(self) -> None:
        heavy = {"torch", "transformers", "accelerate", "safetensors"}
        offenders = []
        for path in sorted((ROOT / "src" / "stt" / "trainercore").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                if any(m.split(".")[0] in heavy for m in modules) and path.name != "hf_trainer.py":
                    offenders.append(path.name)
        self.assertEqual(offenders, [])

    def test_importing_the_package_does_not_import_torch(self) -> None:
        code = "import sys; sys.path.insert(0, 'src'); import stt.trainercore; print('torch' in sys.modules, 'transformers' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False False")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
