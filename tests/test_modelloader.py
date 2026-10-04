"""Unit tests of the ModelLoader component (stt.modelloader): the acceptance tests of the C5 brief, offline.

How to run (project root, virtual environment active):
    python -m unittest tests.test_modelloader -v
The tests load the tiny test model stored in tests/assets/tiny_whisper (built by tools/make_test_model.py, about
4 MB, random weights, the real Whisper tokenizer). No model is downloaded and the network is blocked.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stt.config import ConfigError, SectionRegistry, load_config  # noqa: E402
from stt.modelloader import (  # noqa: E402
    LoaderRegistry,
    ModelConfig,
    ModelLoaderError,
    check_model_folder,
    create_loader,
    default_registry,
    make_token_counter,
)

TINY = ROOT / "tests" / "assets" / "tiny_whisper"


def hashes(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}


class LoaderTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def copy(self, name: str = "m", drop: tuple[str, ...] = ()) -> Path:
        target = self.tmp / name
        shutil.copytree(TINY, target)
        for item in drop:
            (target / item).unlink()
        return target

    def load(self, path: Path, **kwargs: object):  # type: ignore[no-untyped-def]
        return create_loader(ModelConfig(path=path, **kwargs)).load()  # type: ignore[arg-type]


class LoadTests(LoaderTestCase):
    def test_1_valid_folder_loads_and_model_info_is_filled(self) -> None:
        loaded = self.load(TINY)
        info = loaded.info
        self.assertEqual((info.name, info.path, info.dtype, info.language, info.task), ("tiny_whisper", TINY, "float32", "he", "transcribe"))
        self.assertEqual(info.num_mel_bins, 80)
        self.assertEqual(info.num_mel_bins, loaded.feature_extractor.num_mel_bins)
        self.assertEqual(info.num_parameters, sum(p.numel() for p in loaded.model.parameters()))  # type: ignore[attr-defined]
        self.assertGreater(info.num_parameters, 100_000)
        self.assertIsNone(info.source_config_ref)

    def test_feature_extractor_reads_everything_from_the_model(self) -> None:
        fe = self.load(TINY).feature_extractor
        self.assertEqual((fe.sampling_rate, fe.window_seconds, fe.num_mel_bins), (16000, 30.0, 80))
        features = fe.extract(np.zeros(16000, dtype=np.float32), 16000)
        self.assertEqual((features.shape, features.dtype), ((80, 3000), np.float32))  # shorter audio is padded to the window
        full = fe.extract(np.zeros(16000 * 30, dtype=np.float32), 16000)
        self.assertEqual(full.shape, (80, 3000))

    def test_feature_extractor_refuses_other_rates_and_too_long_audio(self) -> None:
        fe = self.load(TINY).feature_extractor
        with self.assertRaises(ValueError) as ctx:
            fe.extract(np.zeros(8000, dtype=np.float32), 8000)
        self.assertIn("16000", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx2:
            fe.extract(np.zeros(16000 * 31, dtype=np.float32), 16000)
        self.assertIn("longer than the model window", str(ctx2.exception))

    def test_tokenizer_label_prefix_and_counts(self) -> None:
        tok = self.load(TINY).tokenizer
        label = tok.encode_label("שלום עולם")
        self.assertEqual(label[:4], [50258, 50279, 50359, 50363])  # start, he, transcribe, no timestamps
        self.assertEqual(label[-1], 50257)  # end of text
        self.assertEqual(tok.count_tokens("שלום עולם"), len(label) - 5)
        self.assertEqual(tok.decode(label), "שלום עולם")
        self.assertIn("<|he|>", tok.decode(label, skip_special_tokens=False))
        self.assertEqual(tok.max_label_tokens, 448)

    def test_language_and_task_are_written_from_config(self) -> None:
        loaded = self.load(TINY, language="en", task="translate")
        generation = loaded.model.generation_config  # type: ignore[attr-defined]
        self.assertEqual((generation.language, generation.task), ("en", "translate"))
        self.assertIsNone(generation.forced_decoder_ids)
        self.assertEqual(loaded.tokenizer.encode_label("a")[1], 50259)  # <|en|>
        self.assertEqual((loaded.info.language, loaded.info.task), ("en", "translate"))

    def test_token_counter_is_a_plain_callable(self) -> None:
        tok = self.load(TINY).tokenizer
        counter = make_token_counter(tok)
        self.assertEqual(counter("שלום עולם"), tok.count_tokens("שלום עולם"))
        self.assertGreater(counter("שלום עולם " * 20), counter("שלום עולם"))
        self.assertEqual(counter(""), 0)

    def test_dtype_is_applied(self) -> None:
        import torch

        loaded = self.load(TINY, dtype="bfloat16")
        self.assertEqual(next(loaded.model.parameters()).dtype, torch.bfloat16)  # type: ignore[attr-defined]
        self.assertEqual(loaded.info.dtype, "bfloat16")

    def test_the_loaded_model_runs_a_forward_pass_in_eval_mode(self) -> None:
        import torch

        loaded = self.load(TINY)
        self.assertFalse(loaded.model.training)  # type: ignore[attr-defined]
        features = torch.from_numpy(loaded.feature_extractor.extract(np.zeros(16000, dtype=np.float32), 16000))[None]
        labels = torch.tensor([loaded.tokenizer.encode_label("שלום")])
        with torch.no_grad():
            out = loaded.model(input_features=features, labels=labels)  # type: ignore[operator]
        self.assertTrue(torch.isfinite(out.loss))

    def test_5_saved_folder_with_pipeline_model_info_exposes_source_config_ref(self) -> None:
        folder = self.copy()
        (folder / "pipeline_model_info.json").write_text(json.dumps({"source_config_ref": "experiments/exp_0003/config_full.yaml"}), encoding="utf-8")
        self.assertEqual(self.load(folder).info.source_config_ref, "experiments/exp_0003/config_full.yaml")
        (folder / "pipeline_model_info.json").write_text("{broken", encoding="utf-8")
        with self.assertRaises(ModelLoaderError):
            self.load(folder)

    def test_6_two_loads_are_independent_objects(self) -> None:
        a, b = self.load(TINY), self.load(TINY)
        self.assertIsNot(a.model, b.model)
        self.assertIsNot(a.tokenizer, b.tokenizer)
        self.assertIsNot(a.feature_extractor, b.feature_extractor)
        first_a = next(a.model.parameters())  # type: ignore[attr-defined]
        first_b = next(b.model.parameters())  # type: ignore[attr-defined]
        before = first_b.clone()
        first_a.data.add_(1.0)
        self.assertTrue((first_b == before).all().item())

    def test_7_input_folder_is_byte_identical_after_loading(self) -> None:
        folder = self.copy()
        before = hashes(folder)
        self.load(folder)
        self.assertEqual(hashes(folder), before)


class FailureTests(LoaderTestCase):
    def test_2_missing_folder_and_not_a_folder_fail_with_the_fix(self) -> None:
        with self.assertRaises(ModelLoaderError) as ctx:
            self.load(self.tmp / "nothing")
        message = str(ctx.exception)
        self.assertIn(str(self.tmp / "nothing"), message)
        self.assertIn("fetch_model", message)
        a_file = self.tmp / "file.txt"
        a_file.write_text("x", encoding="utf-8")
        with self.assertRaises(ModelLoaderError):
            self.load(a_file)

    def test_2_missing_files_are_all_listed(self) -> None:
        folder = self.copy(drop=("config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json"))
        with self.assertRaises(ModelLoaderError) as ctx:
            self.load(folder)
        message = str(ctx.exception)
        for name in ("config.json", "preprocessor_config.json", "tokenizer files", "tokenizer_config.json"):
            self.assertIn(name, message)
        folder2 = self.copy("m2", drop=("model.safetensors",))
        with self.assertRaises(ModelLoaderError) as ctx2:
            self.load(folder2)
        self.assertIn("model.safetensors", str(ctx2.exception))

    def test_3_unsupported_language_fails_naming_model_and_language(self) -> None:
        with self.assertRaises(ModelLoaderError) as ctx:
            self.load(TINY, language="xx")
        self.assertIn("tiny_whisper", str(ctx.exception))
        self.assertIn("'xx'", str(ctx.exception))

    def test_3_english_only_model_fails_when_a_language_is_requested(self) -> None:
        folder = self.copy()
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        config["vocab_size"] = 51864
        (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaises(ModelLoaderError) as ctx:
            self.load(folder, language="he")
        self.assertIn("English-only", str(ctx.exception))
        self.assertIn("'he'", str(ctx.exception))

    def test_not_a_whisper_model_fails_naming_what_is_wrong(self) -> None:
        folder = self.copy()
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        config["model_type"] = "bert"
        (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaises(ModelLoaderError) as ctx:
            self.load(folder)
        self.assertIn("not a Whisper model", str(ctx.exception))
        self.assertIn("bert", str(ctx.exception))

    def test_4_old_format_weights_fail_with_the_explanation_and_the_fix(self) -> None:
        folder = self.copy(drop=("model.safetensors",))
        (folder / "pytorch_model.bin").write_bytes(b"old")
        with self.assertRaises(ModelLoaderError) as ctx:
            self.load(folder)
        message = str(ctx.exception)
        self.assertIn("pytorch_model.bin", message)
        self.assertIn("safetensors", message)
        self.assertIn("Convert", message)

    def test_a_damaged_weights_file_fails_with_an_actionable_message(self) -> None:
        folder = self.copy()
        (folder / "model.safetensors").write_bytes(b"not a real weights file")
        with self.assertRaises(ModelLoaderError) as ctx:
            self.load(folder)
        self.assertIn("Cannot load the model", str(ctx.exception))

    def test_8_unknown_loader_name_lists_registered_names(self) -> None:
        with self.assertRaises(ModelLoaderError) as ctx:
            create_loader(ModelConfig(path=TINY, loader="magic"))
        self.assertIn("hf_whisper", str(ctx.exception))
        self.assertIn("magic", str(ctx.exception))
        self.assertEqual(default_registry().names(), ["hf_whisper"])

    def test_a_second_loader_is_added_through_the_registry_only(self) -> None:
        class Dummy:
            def __init__(self, config: ModelConfig) -> None:
                self.config = config

            def load(self):  # type: ignore[no-untyped-def]
                raise NotImplementedError

        registry = LoaderRegistry({"dummy": Dummy})  # type: ignore[dict-item]
        self.assertIsInstance(create_loader(ModelConfig(path=TINY, loader="dummy"), registry), Dummy)

    def test_check_model_folder_accepts_the_tiny_model(self) -> None:
        check_model_folder(TINY, "he")


class ConfigTests(unittest.TestCase):
    def load(self, *overrides: str) -> ModelConfig:
        config = load_config(overrides=list(overrides), registry=SectionRegistry({"model": ModelConfig}), environ={}, env_file=None, target_environ={})
        return config.section("model", ModelConfig)

    def test_defaults_and_required_path(self) -> None:
        c = self.load("model.path=models/x")
        self.assertEqual((c.loader, c.language, c.task, c.dtype, c.path), ("hf_whisper", "he", "transcribe", "float32", Path("models/x")))
        with self.assertRaises(ConfigError) as ctx:
            self.load()
        message = str(ctx.exception)
        self.assertIn("model.path", message)
        self.assertIn("STT__MODEL__PATH", message)

    def test_unknown_dtype_fails_at_validation_listing_the_allowed_values(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self.load("model.path=x", "model.dtype=int4")
        self.assertIn("float32", str(ctx.exception))
        self.assertIn("bfloat16", str(ctx.exception))

    def test_shipped_example_config_loads(self) -> None:
        config = load_config(ROOT / "configs" / "model_default.yaml", registry=SectionRegistry({"model": ModelConfig}), environ={}, env_file=None, target_environ={})
        self.assertEqual(config.section("model", ModelConfig).path, Path("models/whisper-tiny"))


class IsolationTests(unittest.TestCase):
    def test_9_no_network_call_even_when_online_mode_is_set(self) -> None:
        def blocked(*args: object, **kwargs: object) -> None:
            raise AssertionError("network access attempted")

        with mock.patch.object(socket.socket, "connect", blocked), mock.patch.object(socket, "create_connection", blocked), mock.patch.dict(
            "os.environ", {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}
        ):
            loaded = create_loader(ModelConfig(path=TINY)).load()
        self.assertEqual(loaded.info.num_mel_bins, 80)

    def test_9_torch_and_transformers_are_imported_only_by_the_adapter_module(self) -> None:
        for path in (ROOT / "src" / "stt").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    top = name.split(".")[0]
                    if top in {"torch", "transformers", "safetensors", "accelerate"}:
                        self.assertEqual(path.name, "hf_whisper.py", f"{path} imports {name}")

    def test_importing_the_package_does_not_import_torch_or_transformers(self) -> None:
        code = (
            "import sys; sys.path.insert(0, 'src'); import stt.modelloader; "
            "print([m for m in ('torch', 'transformers') if m in sys.modules])"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "[]")

    def test_the_package_never_enables_remote_code_or_downloads(self) -> None:
        text = (ROOT / "src" / "stt" / "modelloader" / "hf_whisper.py").read_text(encoding="utf-8")
        self.assertIn("local_files_only=True", text)
        self.assertNotIn("trust_remote_code=True", text)
        self.assertEqual(text.count("local_files_only=True"), 3)


if __name__ == "__main__":
    unittest.main()
