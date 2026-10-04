"""Unit tests of the Config component (stt.config): the acceptance tests of the C0 brief, offline.

How to run (project root, virtual environment active):
    python -m unittest tests.test_config -v
The tests use temporary YAML and .env files and small made-up section schemas; nothing is
read from the real environment (a dict is passed instead) and nothing touches the network.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pydantic import ValidationError, model_validator  # noqa: E402

from stt.config import (  # noqa: E402
    AppConfig,
    ConfigError,
    SectionModel,
    SectionRegistry,
    add_config_arguments,
    dump_config,
    load_config,
    load_config_from_args,
)
from stt.config.sources import parse_dotenv  # noqa: E402
from stt.config.validators import ExistingPath, check_ratios_sum_to_one  # noqa: E402


class TrainSection(SectionModel):
    """A made-up section standing in for a component's schema."""

    learning_rate: float = 1e-4
    steps: int = 10


class DataSection(SectionModel):
    """A made-up section with a required value, a must-exist path and ratios."""

    path: ExistingPath
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    @model_validator(mode="after")
    def _ratios(self) -> DataSection:
        check_ratios_sum_to_one({"train": self.train, "val": self.val, "test": self.test})
        return self


REGISTRY = SectionRegistry({"train": TrainSection})


class ConfigTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def write(self, name: str, text: str) -> Path:
        path = self.tmp / name
        path.write_text(text, encoding="utf-8")
        return path

    def load(self, path: Path | None = None, **kwargs: object) -> AppConfig:
        kwargs.setdefault("environ", {})
        kwargs.setdefault("env_file", None)
        kwargs.setdefault("target_environ", {})
        return load_config(path, **kwargs)  # type: ignore[arg-type]


class DefaultsAndPrecedenceTests(ConfigTestCase):
    def test_1_defaults_only_give_valid_frozen_config(self) -> None:
        config = self.load()
        self.assertFalse(config.runtime.online)
        self.assertEqual(config.runtime.seed, 42)
        with self.assertRaises(ValidationError):
            config.runtime.seed = 1  # type: ignore[misc]
        with self.assertRaises(ValidationError):
            config.runtime = config.runtime  # type: ignore[misc]
        with self.assertRaises(TypeError):
            config.sections["x"] = None  # type: ignore[index]

    def test_2_precedence_yaml_then_env_then_cli(self) -> None:
        yaml_path = self.write("c.yaml", "runtime:\n  seed: 1\ntrain:\n  steps: 5\n  learning_rate: 0.5\n")
        self.assertEqual(self.load(yaml_path, registry=REGISTRY).runtime.seed, 1)
        env = {"STT__RUNTIME__SEED": "2", "STT__TRAIN__STEPS": "6"}
        config = self.load(yaml_path, registry=REGISTRY, environ=env)
        self.assertEqual((config.runtime.seed, config.section("train", TrainSection).steps), (2, 6))
        config = self.load(yaml_path, registry=REGISTRY, environ=env, overrides=["runtime.seed=3"])
        self.assertEqual(config.runtime.seed, 3)
        self.assertEqual(config.section("train", TrainSection).steps, 6)
        self.assertEqual(config.section("train", TrainSection).learning_rate, 0.5)

    def test_env_file_is_below_real_environment_and_above_yaml(self) -> None:
        yaml_path = self.write("c.yaml", "runtime:\n  seed: 1\n")
        env_file = self.write(".env", "STT__RUNTIME__SEED=2\nOTHER=ignored\n")
        self.assertEqual(self.load(yaml_path, env_file=env_file).runtime.seed, 2)
        config = self.load(yaml_path, env_file=env_file, environ={"STT__RUNTIME__SEED": "9"})
        self.assertEqual(config.runtime.seed, 9)

    def test_missing_env_file_is_allowed(self) -> None:
        self.assertEqual(self.load(env_file=self.tmp / "no.env").runtime.seed, 42)

    def test_same_input_gives_same_config(self) -> None:
        yaml_path = self.write("c.yaml", "runtime:\n  seed: 5\ntrain:\n  steps: 3\n")
        self.assertEqual(self.load(yaml_path, registry=REGISTRY), self.load(yaml_path, registry=REGISTRY))


class ConversionAndErrorTests(ConfigTestCase):
    def test_3_set_converts_to_schema_type(self) -> None:
        config = self.load(registry=REGISTRY, overrides=["train.learning_rate=3e-5", "train.steps=12", "runtime.online=true"])
        train = config.section("train", TrainSection)
        self.assertEqual(train.learning_rate, 3e-5)
        self.assertEqual(train.steps, 12)
        self.assertTrue(config.runtime.online)

    def test_3_bad_value_fails_naming_key_and_fix(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self.load(registry=REGISTRY, overrides=["train.learning_rate=abc"])
        message = str(ctx.exception)
        self.assertIn("train.learning_rate", message)
        self.assertIn("'abc'", message)
        self.assertIn("--set train.learning_rate", message)
        self.assertIn("STT__TRAIN__LEARNING_RATE", message)

    def test_6_unknown_key_names_closest_valid_key(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self.load(registry=REGISTRY, overrides=["train.learning_rat=1"])
        self.assertIn("Unknown key 'train.learning_rat'", str(ctx.exception))
        self.assertIn("Closest valid key: 'train.learning_rate'", str(ctx.exception))

    def test_6_unknown_section_and_unknown_yaml_key_fail(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self.load(self.write("c.yaml", "runtim:\n  seed: 1\n"))
        self.assertIn("Closest valid key: 'runtime'", str(ctx.exception))
        with self.assertRaises(ConfigError) as ctx2:
            self.load(self.write("d.yaml", "runtime:\n  sed: 1\n"))
        self.assertIn("Closest valid key: 'runtime.seed'", str(ctx2.exception))

    def test_bad_set_arguments_fail(self) -> None:
        for bad in ("noequals", "=3", "a..b=1"):
            with self.assertRaises(ConfigError, msg=bad):
                self.load(overrides=[bad])

    def test_set_through_a_plain_value_fails(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self.load(self.write("c.yaml", "runtime:\n  seed: 1\n"), overrides=["runtime.seed.deeper=1"])
        self.assertIn("single value", str(ctx.exception))

    def test_malformed_environment_name_fails(self) -> None:
        with self.assertRaises(ConfigError):
            self.load(environ={"STT__RUNTIME__": "1"})

    def test_missing_required_value_names_key_and_three_places(self) -> None:
        registry = REGISTRY.with_section("data", DataSection)
        with self.assertRaises(ConfigError) as ctx:
            self.load(registry=registry)
        message = str(ctx.exception)
        self.assertIn("Missing required value 'data.path'", message)
        for place in ("YAML", "STT__DATA__PATH", "--set data.path"):
            self.assertIn(place, message)

    def test_path_that_must_exist_and_ratios(self) -> None:
        registry = REGISTRY.with_section("data", DataSection)
        with self.assertRaises(ConfigError) as ctx:
            self.load(registry=registry, overrides=[f"data.path={self.tmp / 'missing'}"])
        self.assertIn("Path does not exist", str(ctx.exception))
        ok = self.load(registry=registry, overrides=[f"data.path={self.tmp}"])
        self.assertEqual(ok.section("data", DataSection).path, self.tmp)
        with self.assertRaises(ConfigError) as ctx2:
            self.load(registry=registry, overrides=[f"data.path={self.tmp}", "data.train=0.9"])
        self.assertIn("sum to 1", str(ctx2.exception))

    def test_all_problems_are_reported_together(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self.load(registry=REGISTRY, overrides=["train.steps=x", "train.nope=1", "runtime.seed=y"])
        message = str(ctx.exception)
        for part in ("train.steps", "train.nope", "runtime.seed"):
            self.assertIn(part, message)

    def test_yaml_errors_are_actionable(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self.load(self.write("bad.yaml", "runtime:\n\tseed: 1\n"))
        self.assertIn("line", str(ctx.exception))
        with self.assertRaises(ConfigError):
            self.load(self.write("list.yaml", "- a\n- b\n"))
        with self.assertRaises(ConfigError):
            self.load(self.tmp / "missing.yaml")

    def test_section_must_be_a_mapping(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self.load(self.write("c.yaml", "runtime: 5\n"))
        self.assertIn("runtime", str(ctx.exception))

    def test_registry_rejects_bad_sections(self) -> None:
        with self.assertRaises(ConfigError):
            SectionRegistry({"runtime": TrainSection})
        with self.assertRaises(ConfigError):
            SectionRegistry({"Bad": TrainSection})
        with self.assertRaises(ConfigError):
            SectionRegistry({"train": dict})  # type: ignore[dict-item]

    def test_section_accessor_checks_type(self) -> None:
        config = self.load(registry=REGISTRY)
        with self.assertRaises(ConfigError):
            config.section("data", TrainSection)
        with self.assertRaises(ConfigError):
            config.section("train", DataSection)


class ExtendsTests(ConfigTestCase):
    def test_4_extends_merges_base_and_child(self) -> None:
        self.write("base.yaml", "runtime:\n  seed: 1\n  online: true\ntrain:\n  steps: 5\n  learning_rate: 0.1\n")
        child = self.write("child.yaml", "extends: base.yaml\nruntime:\n  seed: 2\ntrain:\n  steps: 9\n")
        config = self.load(child, registry=REGISTRY)
        train = config.section("train", TrainSection)
        self.assertEqual(config.runtime.seed, 2)
        self.assertTrue(config.runtime.online)
        self.assertEqual((train.steps, train.learning_rate), (9, 0.1))

    def test_4_extends_chain_of_three(self) -> None:
        self.write("a.yaml", "runtime:\n  seed: 1\n")
        self.write("b.yaml", "extends: a.yaml\nruntime:\n  online: true\n")
        c = self.write("c.yaml", "extends: b.yaml\n")
        config = self.load(c)
        self.assertEqual((config.runtime.seed, config.runtime.online), (1, True))

    def test_4_cycle_is_detected_with_chain(self) -> None:
        self.write("a.yaml", "extends: b.yaml\n")
        self.write("b.yaml", "extends: a.yaml\n")
        with self.assertRaises(ConfigError) as ctx:
            self.load(self.tmp / "a.yaml")
        self.assertIn("a.yaml -> b.yaml -> a.yaml", str(ctx.exception))

    def test_missing_base_names_chain(self) -> None:
        child = self.write("child.yaml", "extends: nothere.yaml\n")
        with self.assertRaises(ConfigError) as ctx:
            self.load(child)
        self.assertIn("nothere.yaml", str(ctx.exception))
        self.assertIn("child.yaml", str(ctx.exception))

    def test_extends_must_be_text(self) -> None:
        with self.assertRaises(ConfigError):
            self.load(self.write("c.yaml", "extends: [a, b]\n"))

    def test_shipped_base_config_loads(self) -> None:
        config = load_config(ROOT / "configs" / "base.yaml", environ={}, env_file=None, target_environ={})
        self.assertEqual((config.runtime.online, config.runtime.seed), (False, 42))


class DotenvTests(unittest.TestCase):
    def test_5_comments_blank_lines_quotes_export(self) -> None:
        text = (
            "# comment\n\nA=1\nB = two words  # trailing\nC=\"quoted # not a comment\"\n"
            "D='single'\nexport E=5\nEMPTY=\n"
        )
        self.assertEqual(
            parse_dotenv(text),
            {"A": "1", "B": "two words", "C": "quoted # not a comment", "D": "single", "E": "5", "EMPTY": ""},
        )

    def test_5_malformed_line_names_file_and_line(self) -> None:
        for bad, line in (("A=1\nthis is bad\n", 2), ("A=1\n1BAD=2\n", 2), ('A="open\n', 1)):
            with self.assertRaises(ConfigError) as ctx:
                parse_dotenv(bad, source="my.env")
            self.assertIn(f"my.env line {line}", str(ctx.exception))


class DumpAndEnvironmentTests(ConfigTestCase):
    def test_7_dump_reloads_to_equal_config(self) -> None:
        registry = REGISTRY.with_section("data", DataSection)
        yaml_path = self.write("c.yaml", f"runtime:\n  seed: 8\n  online: true\ntrain:\n  steps: 4\ndata:\n  path: {self.tmp.as_posix()}\n")
        original = self.load(yaml_path, registry=registry)
        dumped = dump_config(original)
        reloaded = self.load(self.write("dump.yaml", dumped), registry=registry)
        self.assertEqual(reloaded, original)
        self.assertEqual(dump_config(reloaded), dumped)

    def test_8_offline_mode_sets_three_variables(self) -> None:
        target: dict[str, str] = {}
        self.load(target_environ=target)
        self.assertEqual(target, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})

    def test_8_online_mode_sets_nothing(self) -> None:
        target: dict[str, str] = {}
        self.load(overrides=["runtime.online=true"], target_environ=target)
        self.assertEqual(target, {})

    def test_apply_environment_false_writes_nothing(self) -> None:
        target: dict[str, str] = {}
        self.load(apply_environment=False, target_environ=target)
        self.assertEqual(target, {})


class CliTests(ConfigTestCase):
    def test_argparse_helpers(self) -> None:
        parser = argparse.ArgumentParser()
        add_config_arguments(parser)
        yaml_path = self.write("c.yaml", "runtime:\n  seed: 1\n")
        args = parser.parse_args(["--config", str(yaml_path), "--set", "runtime.seed=4", "--set", "runtime.online=true"])
        config = load_config_from_args(args, environ={}, env_file=None, target_environ={})
        self.assertEqual((config.runtime.seed, config.runtime.online), (4, True))
        args = parser.parse_args([])
        self.assertEqual(load_config_from_args(args, environ={}, env_file=None, target_environ={}).runtime.seed, 42)


class ImportRulesTests(unittest.TestCase):
    def test_9_import_pulls_no_heavy_or_network_modules(self) -> None:
        code = (
            "import sys; sys.path.insert(0, 'src'); import stt.config; "
            "bad = [m for m in ('torch','transformers','huggingface_hub','datasets','optuna','scipy','requests','httpx','urllib3') if m in sys.modules]; "
            "print(bad)"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "[]")

    def test_import_has_no_side_effects_on_environment(self) -> None:
        code = "import sys, os; sys.path.insert(0, 'src'); import stt.config; print(os.environ.get('HF_HUB_OFFLINE'))"
        env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env=env)
        self.assertEqual(proc.stdout.strip(), "None")


if __name__ == "__main__":
    unittest.main()
