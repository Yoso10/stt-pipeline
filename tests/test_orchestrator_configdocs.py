"""Tests of the `config-docs` command and of docs/CONFIG_REFERENCE.md (Wave 3 task 6): the file is generated from the schemas, is up to date, and generating it changes no key and no default.

How to run (project root, virtual environment active):
    python -m unittest tests.test_orchestrator_configdocs -v
If `test_the_committed_reference_is_up_to_date` fails, a config key, default or description changed: run `python -m stt config-docs` and commit the new docs/CONFIG_REFERENCE.md.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pydantic import BaseModel, Field  # noqa: E402

from stt.orchestrator import cli  # noqa: E402
from stt.orchestrator.configdocs import MARKER, count_keys, default_text, render_config_reference, status_tag, type_name, write_reference  # noqa: E402
from stt.orchestrator.errors import OrchestratorError  # noqa: E402
from stt.orchestrator.wiring import documented_sections  # noqa: E402

REFERENCE = ROOT / "docs" / "CONFIG_REFERENCE.md"


class Inner(BaseModel):
    """The inner block."""

    depth: int = Field(default=3, description="NEW 3. How deep | with a pipe.")


class Demo(BaseModel):
    """A demo section. It has two lines
    of text."""

    name: str = Field(default="a", description="DEFAULT a. A name.")
    mode: Literal["fast", "slow"] = Field(default="fast", description="No tag here.")
    needed: Path = Field(description="REQUIRED. Must be set.")
    untagged_required: int = Field(description="Also needed.")
    folder: Path | None = Field(default=None, description="DEFAULT none.")
    inner: Inner = Field(default_factory=Inner)
    names: list[str] = Field(default_factory=lambda: ["x", "y"], description="DEFAULT x and y.")


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class RenderTests(unittest.TestCase):
    def test_every_key_type_default_status_and_description_is_in_the_table_with_dotted_keys(self) -> None:
        text = render_config_reference({"demo": Demo})
        self.assertIn("| `demo.name` | str | \"a\" | DEFAULT | DEFAULT a. A name. |", text)
        self.assertIn("| `demo.mode` | 'fast' \\| 'slow' | \"fast\" | DEFAULT | No tag here. |", text)  # the tag is derived from the default
        self.assertIn("| `demo.needed` | Path | (required) | REQUIRED | REQUIRED. Must be set. |", text)
        self.assertIn("| `demo.untagged_required` | int | (required) | REQUIRED | Also needed. |", text)
        self.assertIn("| `demo.folder` | Path \\| None | null | DEFAULT |", text)
        self.assertIn("| `demo.inner.depth` | int | 3 | NEW | NEW 3. How deep \\| with a pipe. |", text)  # a nested block is flattened; the pipe is escaped
        self.assertIn('| `demo.names` | list[str] | ["x", "y"] | DEFAULT |', text)
        self.assertIn("A demo section. It has two lines of text.", text)
        self.assertEqual(count_keys(text), (1, 7))

    def test_the_output_is_deterministic_starts_with_the_marker_and_the_how_to_run_text(self) -> None:
        a, b = render_config_reference({"demo": Demo}), render_config_reference({"demo": Demo})
        self.assertEqual(a, b)
        self.assertEqual(a.splitlines()[0], MARKER)
        self.assertIn("HOW TO RUN: `python -m stt config-docs`", a)
        self.assertNotIn("\r", a)

    def test_type_names_defaults_and_status_tags(self) -> None:
        fields = Demo.model_fields
        self.assertEqual((type_name(fields["names"].annotation), type_name(fields["folder"].annotation)), ("list[str]", "Path | None"))
        self.assertEqual((default_text(fields["needed"]), default_text(fields["names"])), ("(required)", '["x", "y"]'))
        self.assertEqual([status_tag(fields[k]) for k in ("name", "mode", "needed", "inner")], ["DEFAULT", "DEFAULT", "REQUIRED", "DEFAULT"])

    def test_a_very_long_default_is_shortened(self) -> None:
        class Long(BaseModel):
            items: list[str] = Field(default_factory=lambda: ["word"] * 100)

        text = default_text(Long.model_fields["items"])
        self.assertTrue(text.endswith("…") and len(text) == 160)


class RealSchemaTests(unittest.TestCase):
    def test_the_committed_reference_is_up_to_date(self) -> None:
        expected = render_config_reference(documented_sections())
        self.assertTrue(REFERENCE.is_file(), "docs/CONFIG_REFERENCE.md does not exist: run `python -m stt config-docs`")
        self.assertEqual(REFERENCE.read_text(encoding="utf-8"), expected, "docs/CONFIG_REFERENCE.md is out of date: run `python -m stt config-docs` and commit it")

    def test_every_registered_section_and_known_keys_with_their_defaults_are_documented(self) -> None:
        text = render_config_reference(documented_sections())
        for name in ("runtime", "data", "train", "eval", "run", "search", "report", "export", "select", "registry", "pipeline", "smoke", "learning_curve", "benchmark"):
            self.assertIn(f"## `{name}`", text)
        for row in ("| `runtime.seed` | int | 42 |", "| `export.formats` | list[str] | [\"hf_safetensors\"] |", "| `registry.cleanup` | bool | true |", "| `data.source.path` | Path | (required) | REQUIRED |", "| `train.hyperparams.learning_rate` | float | 1e-05 |"):
            self.assertIn(row, text)

    def test_generating_the_reference_changes_no_key_and_no_default(self) -> None:
        def defaults() -> dict[str, str]:
            found = {}
            for name, model in documented_sections().items():
                for key, field in model.model_fields.items():
                    found[f"{name}.{key}"] = f"{field.annotation}|{field.is_required()}|{default_text(field)}|{field.description}"
            return found

        before = defaults()
        render_config_reference(documented_sections())
        with tempfile.TemporaryDirectory() as tmp:
            run_cli("config-docs", "--out", str(Path(tmp) / "r.md"))
        self.assertEqual(defaults(), before)


class WriteTests(unittest.TestCase):
    def test_the_command_writes_the_file_and_reports_the_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "ref.md"
            code, out, err = run_cli("config-docs", "--out", str(target))
            self.assertEqual(code, 0, err)
            body = json.loads(out)
            self.assertEqual((body["sections"], body["exit_code"]), (len(documented_sections()), 0))  # one table per section (the number follows the sections, so a new section does not break this test)
            self.assertGreater(body["keys"], 100)
            self.assertEqual(target.read_text(encoding="utf-8"), render_config_reference(documented_sections()))
            self.assertEqual(run_cli("config-docs", "--out", str(target))[0], 0)  # a file made by the command is replaced

    def test_a_file_that_the_command_did_not_make_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "mine.md"
            target.write_text("# my own notes\n", encoding="utf-8")
            code, out, err = run_cli("config-docs", "--out", str(target))
            self.assertEqual((code, out), (2, ""))
            self.assertIn("not overwritten", err)
            self.assertEqual(target.read_text(encoding="utf-8"), "# my own notes\n")
            with self.assertRaises(OrchestratorError):
                write_reference("x", Path(tmp) / "no_folder" / "r.md")

    def test_the_command_has_help_and_an_example(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            cli.main(["config-docs", "--help"])
        for part in ("python -m stt config-docs", "Example:", "--out"):
            self.assertIn(part, out.getvalue())


if __name__ == "__main__":
    unittest.main()
