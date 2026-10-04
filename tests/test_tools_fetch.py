"""Unit tests of tools/fetch_model.py, tools/fetch_data.py and the shared _common.py (no network).

How to run (project root, virtual environment active):
    python -m unittest discover -s tests -v
or only this file:
    python -m unittest tests.test_tools_fetch -v
The Hub is replaced by fake downloaders, so these tests run offline in about a second.
"""

from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import _common  # noqa: E402
import fetch_data  # noqa: E402
import fetch_model  # noqa: E402
from _common import FetchError, RepoNotFoundError, RepoNotPublicError  # noqa: E402


def good_model_downloader(repo: str, repo_type: str, target: Path, patterns: list[str]) -> None:
    for name in ("config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json"):
        (target / name).write_text("{}", encoding="utf-8")
    (target / "model.safetensors").write_bytes(b"x" * 100)
    (target / ".cache").mkdir()
    (target / ".cache" / "junk").write_text("hub bookkeeping", encoding="utf-8")


class FetchModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name) / "models" / "tiny"

    def test_complete_download_writes_folder_and_info(self) -> None:
        result = fetch_model.fetch_model("openai/whisper-tiny", self.out, good_model_downloader)
        self.assertFalse(result.reused)
        self.assertTrue((self.out / "fetch_info.json").is_file())
        self.assertFalse((self.out / ".cache").exists())
        self.assertFalse(self.out.with_name("tiny.partial").exists())
        self.assertEqual(result.info["repo"], "openai/whisper-tiny")
        paths = {f["path"] for f in result.info["files"]}  # type: ignore[union-attr]
        self.assertIn("model.safetensors", paths)

    def test_second_run_reuses_and_does_not_download(self) -> None:
        fetch_model.fetch_model("openai/whisper-tiny", self.out, good_model_downloader)

        def boom(*_: object) -> None:
            raise AssertionError("must not download again")

        result = fetch_model.fetch_model("openai/whisper-tiny", self.out, boom)
        self.assertTrue(result.reused)

    def test_other_repo_in_same_folder_fails_and_keeps_folder(self) -> None:
        fetch_model.fetch_model("openai/whisper-tiny", self.out, good_model_downloader)
        with self.assertRaises(FetchError) as ctx:
            fetch_model.fetch_model("openai/whisper-base", self.out, good_model_downloader)
        self.assertIn("openai/whisper-tiny", str(ctx.exception))
        self.assertTrue((self.out / "model.safetensors").is_file())

    def test_interrupted_download_is_rebuilt(self) -> None:
        self.out.mkdir(parents=True)
        (self.out / "model.safetensors").write_bytes(b"half")  # no fetch_info.json
        stale = self.out.with_name("tiny.partial")
        stale.mkdir()
        (stale / "old").write_text("old", encoding="utf-8")
        result = fetch_model.fetch_model("openai/whisper-tiny", self.out, good_model_downloader)
        self.assertFalse(result.reused)
        self.assertEqual((self.out / "model.safetensors").stat().st_size, 100)
        self.assertFalse(stale.exists())

    def test_missing_required_files_fails_and_stores_nothing(self) -> None:
        def only_config(repo: str, repo_type: str, target: Path, patterns: list[str]) -> None:
            (target / "config.json").write_text("{}", encoding="utf-8")

        with self.assertRaises(FetchError) as ctx:
            fetch_model.fetch_model("x/y", self.out, only_config)
        message = str(ctx.exception)
        self.assertIn("preprocessor_config.json", message)
        self.assertIn("model.safetensors", message)
        self.assertFalse(self.out.exists())
        self.assertFalse(self.out.with_name("tiny.partial").exists())

    def test_not_found_and_gated_errors_pass_through_and_clean_up(self) -> None:
        for exc in (RepoNotFoundError("nf"), RepoNotPublicError("gated")):

            def failing(repo: str, repo_type: str, target: Path, patterns: list[str], e: Exception = exc) -> None:
                (target / "half").write_text("x", encoding="utf-8")
                raise e

            with self.assertRaises(type(exc)):
                fetch_model.fetch_model("x/y", self.out, failing)
            self.assertFalse(self.out.exists())
            self.assertFalse(self.out.with_name("tiny.partial").exists())

    def test_main_exit_codes(self) -> None:
        self.assertEqual(fetch_model.main(["--repo", "a/b", "--out", str(self.out)], good_model_downloader), 0)

        def failing(repo: str, repo_type: str, target: Path, patterns: list[str]) -> None:
            raise FetchError("no network")

        self.assertEqual(fetch_model.main(["--repo", "a/b", "--out", str(self.out) + "2"], failing), 1)


class FetchDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name) / "data" / "raw"

    @staticmethod
    def parquet_downloader(repo: str, repo_type: str, target: Path, patterns: list[str]) -> None:
        assert repo_type == "dataset"
        (target / "data").mkdir()
        (target / "data" / "train-00000.parquet").write_bytes(b"PAR1")
        (target / "README.md").write_text("card", encoding="utf-8")

    def test_download_then_reuse(self) -> None:
        first = fetch_data.fetch_data("o/d", self.out, self.parquet_downloader)
        self.assertFalse(first.reused)
        self.assertTrue((self.out / "data" / "train-00000.parquet").is_file())
        second = fetch_data.fetch_data("o/d", self.out, lambda *_: (_ for _ in ()).throw(AssertionError()))
        self.assertTrue(second.reused)

    def test_no_parquet_fails(self) -> None:
        def readme_only(repo: str, repo_type: str, target: Path, patterns: list[str]) -> None:
            (target / "README.md").write_text("card", encoding="utf-8")

        with self.assertRaises(FetchError) as ctx:
            fetch_data.fetch_data("o/d", self.out, readme_only)
        self.assertIn(".parquet", str(ctx.exception))
        self.assertFalse(self.out.exists())

    def test_dry_run_reports_size_without_downloading(self) -> None:
        files = [("data/a.parquet", 600_000_000), ("data/b.parquet", 200_000_000), ("README.md", 5_000), ("x.bin", 9)]
        text = fetch_data.dry_run_summary("o/d", lambda repo: files)
        self.assertIn("3 files", text)
        self.assertIn("800.0 MB", text)
        self.assertIn("parquet: 2 files", text)

    def test_main_dry_run_needs_no_out_and_exit_zero(self) -> None:
        code = fetch_data.main(["--repo", "o/d", "--dry-run"], lister=lambda repo: [("a.parquet", 10)])
        self.assertEqual(code, 0)


class NoNetworkAndRulesTests(unittest.TestCase):
    def test_default_downloader_translates_hub_errors(self) -> None:
        """The real downloader maps Hub exceptions to our errors (huggingface_hub is faked)."""
        import huggingface_hub
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

        import httpx

        response = httpx.Response(401, request=httpx.Request("GET", "https://hub.invalid/x"))
        original = huggingface_hub.snapshot_download
        cases = {
            RepositoryNotFoundError("nf", response=response): RepoNotFoundError,
            GatedRepoError("gated", response=response): RepoNotPublicError,
            ConnectionError("offline"): FetchError,
        }
        try:
            for raised, expected in cases.items():

                def fake(**_: object) -> None:
                    raise raised

                huggingface_hub.snapshot_download = fake  # type: ignore[assignment]
                with self.assertRaises(expected) as ctx:
                    _common.hub_downloader("a/b", "model", Path("."), ["*.json"])
                self.assertIn("a/b", str(ctx.exception))
        finally:
            huggingface_hub.snapshot_download = original  # type: ignore[assignment]

    def test_no_pipeline_module_imports_tools(self) -> None:
        for path in (ROOT / "src").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    self.assertNotEqual(name.split(".")[0], "tools", f"{path} imports {name}")

    def test_tools_have_no_interactive_prompts(self) -> None:
        for path in (ROOT / "tools").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotEqual(node.func.id, "input", f"{path} calls input()")

    def test_every_tool_starts_with_how_to_run(self) -> None:
        for path in (ROOT / "tools").glob("[a-z]*.py"):
            if path.name.startswith("_"):
                continue
            doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
            self.assertIn("How to run", doc, f"{path} lacks a How to run header")


if __name__ == "__main__":
    unittest.main()
