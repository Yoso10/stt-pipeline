"""Tests of `runtime.device` (DECISIONS 87): the key, the resolution (`cpu` / `cuda` / `auto`), the loud failure without a GPU, and how the value reaches the TrainerCore (7a)
and the ModelLoader and the transcriber (7b).

How to run (project root, virtual environment active):
    python -m unittest tests.test_device -v
Offline and CPU only: the availability of a GPU is FAKED (a `TorchFacts` object), so no CUDA is needed. The default (`cpu`) is checked to behave as before.
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from stt.config import ConfigError, dump_config, load_config, resolve_device  # noqa: E402
from stt.config.device import TorchFacts, describe  # noqa: E402
from stt.orchestrator import cli  # noqa: E402
from stt.orchestrator.wiring import build_services, section_registry  # noqa: E402

CPU_BUILD = TorchFacts("2.14.1+cpu", None, False, 0)
NO_DRIVER = TorchFacts("2.14.1+cu126", "12.6", False, 0)
ONE_GPU = TorchFacts("2.14.1+cu126", "12.6", True, 1, "Fake GPU 24GB")
NO_TORCH = TorchFacts(None, None, False, 0)


def must_not_probe() -> TorchFacts:
    raise AssertionError("the probe must not be called for runtime.device cpu")


REQUIRED = [  # the sections of the whole program need these three values; nothing is read from them in these tests
    "data.source.type=csv",
    f"data.source.path={ROOT.as_posix()}",
    "data.columns.audio=audio",
    "data.columns.text=text",
    "data.columns.group=group",
    f"model.path={(ROOT / 'tests' / 'assets' / 'tiny_whisper').as_posix()}",
]


def config_with(*overrides: str, environ: dict[str, str] | None = None) -> Any:
    return load_config(None, overrides=[*REQUIRED, *overrides], registry=section_registry(), env_file=None, environ=environ or {}, apply_environment=False)


class KeyTests(unittest.TestCase):
    def test_the_default_is_cpu_and_the_old_keys_are_unchanged(self) -> None:
        runtime = config_with().runtime
        self.assertEqual((runtime.device, runtime.online, runtime.seed), ("cpu", False, 42))

    def test_set_and_environment_work_and_set_wins(self) -> None:
        self.assertEqual(config_with("runtime.device=cuda").runtime.device, "cuda")
        self.assertEqual(config_with(environ={"STT__RUNTIME__DEVICE": "auto"}).runtime.device, "auto")
        self.assertEqual(config_with("runtime.device=cpu", environ={"STT__RUNTIME__DEVICE": "cuda"}).runtime.device, "cpu")

    def test_an_unknown_value_fails_at_config_time_naming_the_key(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            config_with("runtime.device=gpu")
        self.assertIn("runtime.device", str(ctx.exception))

    def test_the_dump_shows_the_key(self) -> None:
        self.assertIn("device: cpu", dump_config(config_with()))


class ResolveTests(unittest.TestCase):
    def test_cpu_never_asks_torch(self) -> None:
        self.assertEqual(resolve_device("cpu", must_not_probe), "cpu")

    def test_auto_takes_cuda_when_a_gpu_is_there_and_cpu_otherwise(self) -> None:
        self.assertEqual(resolve_device("auto", lambda: ONE_GPU), "cuda")
        for facts in (CPU_BUILD, NO_DRIVER, NO_TORCH):
            self.assertEqual(resolve_device("auto", lambda facts=facts: facts), "cpu")

    def test_auto_logs_the_choice_at_info(self) -> None:
        with self.assertLogs("stt.config", level="INFO") as logs:
            resolve_device("auto", lambda: CPU_BUILD)
        self.assertIn("runtime.device is auto: using cpu", logs.output[0])
        self.assertIn("torch 2.14.1+cpu", logs.output[0])

    def test_cuda_with_a_gpu_is_cuda(self) -> None:
        self.assertEqual(resolve_device("cuda", lambda: ONE_GPU), "cuda")

    def test_cuda_without_a_gpu_fails_loudly_and_never_falls_back(self) -> None:
        for facts, expected in ((CPU_BUILD, "CPU-only build"), (NO_DRIVER, "built for CUDA 12.6"), (NO_TORCH, "torch is not installed")):
            with self.assertRaises(ConfigError) as ctx:
                resolve_device("cuda", lambda facts=facts: facts)
            message = str(ctx.exception)
            self.assertIn("runtime.device", message)  # the key
            self.assertIn(expected, message)  # what was found
            self.assertIn("device count: 0" if facts.version else "torch is not installed", message)
            self.assertIn("cuda available: False" if facts.version else "torch is not installed", message)
            self.assertIn("Fix:", message)  # how to fix it
            self.assertIn("does NOT fall back", message)

    def test_an_unknown_value_is_refused(self) -> None:
        with self.assertRaises(ConfigError):
            resolve_device("tpu", lambda: ONE_GPU)

    def test_describe_names_the_version_the_build_and_the_count(self) -> None:
        text = describe(ONE_GPU)
        self.assertIn("torch 2.14.1+cu126", text)
        self.assertIn("CUDA 12.6", text)
        self.assertIn("device count: 1", text)

    def test_the_default_probe_runs_here_without_failing(self) -> None:
        facts = __import__("stt.config.device", fromlist=["torch_facts"]).torch_facts()
        self.assertIsInstance(facts.cuda_available, bool)


class WiringAndCliTests(unittest.TestCase):
    def test_the_default_wiring_is_on_the_cpu_and_does_not_probe(self) -> None:
        with mock.patch("stt.config.device.torch_facts", must_not_probe):
            services = build_services(config_with())
        self.assertEqual(services.training._device, "cpu")  # type: ignore[attr-defined]

    def test_the_resolved_device_reaches_the_training_wiring(self) -> None:
        with mock.patch("stt.config.device.torch_facts", lambda: ONE_GPU):
            self.assertEqual(build_services(config_with("runtime.device=cuda")).training._device, "cuda")  # type: ignore[attr-defined]
            self.assertEqual(build_services(config_with("runtime.device=auto")).training._device, "cuda")  # type: ignore[attr-defined]

    def test_cuda_without_a_gpu_stops_the_command_with_exit_code_2_before_any_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = config_with("runtime.device=cuda", f"run.experiments_dir={Path(tmp) / 'experiments'}")
            err = io.StringIO()
            with mock.patch("stt.config.device.torch_facts", lambda: CPU_BUILD), mock.patch.object(cli, "load_config", return_value=config), contextlib.redirect_stderr(err):
                code = cli.main(["evaluate", "--config", "unused.yaml"])
            self.assertEqual(code, 2)
            self.assertIn("runtime.device", err.getvalue())
            self.assertIn("CPU-only build", err.getvalue())
            self.assertFalse((Path(tmp) / "experiments").exists())  # nothing was created


class TrainerCoreDeviceTests(unittest.TestCase):
    def test_the_trainer_refuses_an_unresolved_value(self) -> None:
        from stt.trainercore import TrainConfig, TrainerConfigError, TrainerCore

        with self.assertRaises(TrainerConfigError):
            TrainerCore(TrainConfig(), device="auto")

    def run_until_the_arguments(self, device: str | None) -> dict[str, Any]:
        """Run a real TrainerCore call on the tiny test model, stopped at the moment the Hugging Face arguments are made; returns those arguments."""
        from test_trainercore import TrainerTestCase, conf

        class Stopped(Exception):
            pass

        captured: dict[str, Any] = {}

        def fake_arguments(**kwargs: Any) -> Any:
            captured.update(kwargs)
            raise Stopped

        case = TrainerTestCase()
        case.setUp()
        try:
            from stt.trainercore import TrainerCore

            core = TrainerCore(conf()) if device is None else TrainerCore(conf(), device=device)
            with mock.patch("stt.trainercore.hf_trainer.Seq2SeqTrainingArguments", fake_arguments), self.assertRaises(Stopped):
                core.train(case.request(conf()))
        finally:
            case._tmp.cleanup()
        return captured

    def test_the_default_and_cpu_use_the_cpu_exactly_as_before(self) -> None:
        self.assertIs(self.run_until_the_arguments(None)["use_cpu"], True)
        self.assertIs(self.run_until_the_arguments("cpu")["use_cpu"], True)

    def test_cuda_switches_use_cpu_off(self) -> None:
        self.assertIs(self.run_until_the_arguments("cuda")["use_cpu"], False)


class ModelLoaderDeviceTests(unittest.TestCase):
    """7b: the device reaches the loader and the loader places the model (the GPU is faked: `.to` is recorded, nothing is moved)."""

    def config(self) -> Any:
        from stt.modelloader import ModelConfig

        return ModelConfig(path=ROOT / "tests" / "assets" / "tiny_whisper")

    def test_cpu_is_the_default_and_the_model_is_not_moved(self) -> None:
        from stt.modelloader import create_loader

        with mock.patch("transformers.WhisperForConditionalGeneration.to") as moved:
            loaded = create_loader(self.config()).load()
        moved.assert_not_called()
        self.assertEqual(str(loaded.model.device), "cpu")  # type: ignore[attr-defined]

    def test_cuda_moves_the_model_after_loading(self) -> None:
        from stt.modelloader import create_loader

        with mock.patch("transformers.WhisperForConditionalGeneration.to", autospec=True, side_effect=lambda self, *a, **k: self) as moved:
            create_loader(self.config(), device="cuda").load()
        self.assertEqual(moved.call_args.args[1], "cuda")

    def test_a_failed_move_is_a_loader_error_with_the_device_named(self) -> None:
        from stt.modelloader import ModelLoaderError, create_loader

        with mock.patch("transformers.WhisperForConditionalGeneration.to", side_effect=RuntimeError("CUDA out of memory")), self.assertRaises(ModelLoaderError) as ctx:
            create_loader(self.config(), device="cuda").load()
        self.assertIn("cuda", str(ctx.exception))
        self.assertIn("out of memory", str(ctx.exception))

    def test_a_loader_without_device_support_is_refused_for_cuda_but_works_on_cpu(self) -> None:
        from stt.modelloader import LoaderRegistry, ModelLoaderError, create_loader

        class Legacy:
            def __init__(self, config: Any) -> None:
                self.config = config

        registry = LoaderRegistry({"hf_whisper": Legacy})  # type: ignore[dict-item]
        self.assertIsInstance(create_loader(self.config(), registry), Legacy)
        with self.assertRaises(ModelLoaderError) as ctx:
            create_loader(self.config(), registry, device="cuda")
        self.assertIn("does not support", str(ctx.exception))

    def test_an_unresolved_device_is_refused(self) -> None:
        from stt.modelloader import ModelLoaderError, create_loader

        with self.assertRaises(ModelLoaderError):
            create_loader(self.config(), device="auto")

    def test_the_model_wiring_passes_the_resolved_device(self) -> None:
        with mock.patch("stt.config.device.torch_facts", lambda: ONE_GPU):
            self.assertEqual(build_services(config_with("runtime.device=cuda")).models._device, "cuda")  # type: ignore[attr-defined]
        self.assertEqual(build_services(config_with()).models._device, "cpu")  # type: ignore[attr-defined]


class TranscriberDeviceTests(unittest.TestCase):
    """7b: the inputs go to the device of the model. A `meta` device stands in for the GPU (it exists without CUDA)."""

    def run_with_model_on(self, device: str) -> str:
        import types

        import numpy as np
        import torch

        from stt.evaluator.hf_transcriber import HfTranscriber
        from stt.evaluator.types import DecodeSettings

        seen: list[str] = []

        class Native:
            training = False
            config = types.SimpleNamespace(max_target_positions=448, eos_token_id=7)

            def __init__(self) -> None:
                self.device = torch.device(device)

            def eval(self) -> None:
                pass

            def train(self, mode: bool = True) -> None:
                pass

            def generate(self, input_features: Any, **kwargs: Any) -> Any:
                seen.append(input_features.device.type)
                return torch.tensor([[1, 2, 7]] * input_features.shape[0])

        class Data:
            def __len__(self) -> int:
                return 2

            def __getitem__(self, i: int) -> Any:
                return types.SimpleNamespace(sample_id=f"s{i}", input_features=np.zeros((4, 6), dtype=np.float32))

        model = types.SimpleNamespace(model=Native(), tokenizer=types.SimpleNamespace(decode=lambda ids, skip_special_tokens=True: "x"))
        HfTranscriber().transcribe(model, Data(), DecodeSettings(1, 50, 2, None, None))  # type: ignore[arg-type]
        return seen[0]

    def test_the_inputs_follow_the_device_of_the_model(self) -> None:
        self.assertEqual(self.run_with_model_on("cpu"), "cpu")  # unchanged behaviour
        self.assertEqual(self.run_with_model_on("meta"), "meta")  # the inputs are moved to wherever the model is


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
