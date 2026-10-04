"""Unit tests of the DataSource component (stt.datasource): the acceptance tests of the C1 brief, offline.

How to run (project root, virtual environment active):
    python -m unittest tests.test_datasource -v
The tests generate tiny wav files and tiny parquet files in a temporary folder. Nothing is
downloaded and the real data folder is not touched.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stt.config import ConfigError, SectionRegistry, load_config  # noqa: E402
from stt.datasource import (  # noqa: E402
    CsvAudioSource,
    DataConfig,
    DataSourceError,
    HfParquetSource,
    create_data_source,
    default_registry,
)
from stt.datasource.audio_header import read_duration_seconds  # noqa: E402


def wav_bytes(seconds: float = 1.0, sr: int = 16000) -> bytes:
    buffer = io.BytesIO()
    t = np.arange(int(seconds * sr)) / sr
    soundfile.write(buffer, (0.2 * np.sin(2 * np.pi * 330 * t)).astype("float32"), sr, format="WAV")
    return buffer.getvalue()


def make_config(path: Path, **overrides: Any) -> DataConfig:
    base: dict[str, Any] = {
        "source": {"type": "csv", "path": str(path)},
        "columns": {"audio": "audio", "text": "text", "group": "group", "split": "split"},
        "extra_types": {"quality": "float", "count": "int", "golden": "bool"},
        "max_rejected_ratio": 1.0,
    }
    base.update(overrides)
    return DataConfig.model_validate(base)


def write_manifest(folder: Path, rows: list[dict[str, str]], bom: bool = False) -> Path:
    path = folder / "manifest.csv"
    with path.open("w", encoding="utf-8-sig" if bom else "utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def hashes(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}


class CsvTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data = self.tmp / "data"
        self.data.mkdir()
        self.out = self.tmp / "out"

    def add_wav(self, name: str, seconds: float = 1.0) -> str:
        (self.data / name).write_bytes(wav_bytes(seconds))
        return name

    def row(self, ident: str, audio: str | None = None, text: str = "שלום", group: str = "g1", split: str = "train", **extra: str) -> dict[str, str]:
        audio = self.add_wav(f"{ident}.wav") if audio is None else audio
        return {"id": ident, "audio": audio, "text": text, "group": group, "split": split,
                "quality": extra.get("quality", "0.5"), "count": extra.get("count", "3"), "golden": extra.get("golden", "True"), "note": "n"}  # fmt: skip

    def source(self, rows: list[dict[str, str]], **overrides: Any) -> CsvAudioSource:
        write_manifest(self.data, rows)
        overrides.setdefault("columns", {"id": "id", "audio": "audio", "text": "text", "group": "group", "split": "split"})
        return CsvAudioSource(make_config(self.data, **overrides), self.out)


class CsvSourceTests(CsvTestCase):
    def test_1_three_valid_rows_sorted_typed_and_extra_preserved(self) -> None:
        rows = [self.row("c", group="g2", split="eval", quality="0.25"), self.row("a"), self.row("b", golden="no")]
        result = self.source(rows).load()
        self.assertEqual([s.id for s in result.samples], ["a", "b", "c"])
        a, b, c = result.samples
        self.assertEqual((a.group_id, a.source_split, c.group_id, c.source_split), ("g1", "train", "g2", "eval"))
        self.assertEqual(dict(a.extra), {"quality": 0.5, "count": 3, "golden": True, "note": "n"})
        self.assertIs(b.extra["golden"], False)
        self.assertEqual(c.extra["quality"], 0.25)
        self.assertAlmostEqual(a.duration_sec, 1.0, places=3)
        self.assertEqual(a.audio_path, self.data / "a.wav")
        self.assertEqual(result.report.loaded, 3)
        self.assertEqual(result.report.loaded_per_split, {"eval": 1, "train": 2})
        self.assertIn("quality", result.report.extra_distributions)
        self.assertEqual(result.report.duration.count, 3)  # type: ignore[union-attr]
        with self.assertRaises(TypeError):
            a.extra["x"] = 1  # type: ignore[index]

    def test_report_counts_groups_in_several_splits(self) -> None:
        rows = [self.row("a", group="g1", split="train"), self.row("b", group="g1", split="eval"), self.row("c", group="g2")]
        report = self.source(rows).load().report
        self.assertEqual((report.distinct_groups, report.groups_in_several_splits), (2, 1))
        self.assertIn("seen 3", report.summary())

    def test_2_rejections_have_correct_reasons_and_are_listed(self) -> None:
        (self.data / "broken.wav").write_bytes(b"this is not audio")
        (self.data / "empty.wav").write_bytes(b"")
        rows = [
            self.row("ok"),
            self.row("noaudio", audio="missing.wav"),
            self.row("notext", text="   "),
            self.row("broken", audio="broken.wav"),
            self.row("emptyfile", audio="empty.wav"),
            self.row("nogroup", group=" "),
            self.row("badnum", count="three"),
            self.row("noaudiocell", audio=""),
        ]
        result = self.source(rows).load()
        self.assertEqual([s.id for s in result.samples], ["ok"])
        reasons = {r.id: r.reason for r in result.rejections}
        self.assertEqual(
            reasons,
            {"noaudio": "missing_audio", "notext": "empty_text", "broken": "unreadable_audio", "emptyfile": "unreadable_audio",
             "nogroup": "missing_group", "badnum": "bad_extra_value", "noaudiocell": "missing_audio"},
        )  # fmt: skip
        self.assertEqual(result.report.rejection_reasons["unreadable_audio"], 2)
        with (self.out / "rejected.csv").open(encoding="utf-8", newline="") as handle:
            listed = {row["id"]: row["reason"] for row in csv.DictReader(handle)}
        self.assertEqual(listed, reasons)

    def test_malformed_row_is_rejected(self) -> None:
        write_manifest(self.data, [self.row("a")])
        with (self.data / "manifest.csv").open("a", encoding="utf-8", newline="") as handle:
            handle.write("short,row\n")
        result = CsvAudioSource(make_config(self.data, columns={"id": "id", "audio": "audio", "text": "text", "group": "group", "split": "split"}), self.out).load()
        self.assertEqual([r.reason for r in result.rejections], ["malformed_row"])

    def test_empty_typed_value_is_left_out_of_extra_and_counted(self) -> None:
        result = self.source([self.row("a", quality=""), self.row("b")]).load()
        self.assertNotIn("quality", result.samples[0].extra)
        self.assertEqual(result.report.missing_extra_values, {"quality": 1})

    def test_duration_comes_from_header_not_from_a_metadata_column(self) -> None:
        rows = [self.row("a", audio=self.add_wav("long.wav", 2.5))]
        rows[0]["duration"] = "30"
        sample = self.source(rows, extra_types={"duration": "float"}).load().samples[0]
        self.assertAlmostEqual(sample.duration_sec, 2.5, places=3)
        self.assertEqual(sample.extra["duration"], 30.0)

    def test_3_duplicate_id_fails_naming_it(self) -> None:
        with self.assertRaises(DataSourceError) as ctx:
            self.source([self.row("a"), self.row("b"), self.row("a")]).load()
        self.assertIn("Duplicate", str(ctx.exception))
        self.assertIn("a", str(ctx.exception))

    def test_empty_id_fails(self) -> None:
        with self.assertRaises(DataSourceError):
            self.source([self.row(" ", audio=self.add_wav("x.wav"))]).load()

    def test_4_unknown_columns_fail_and_list_existing_ones(self) -> None:
        rows = [self.row("a")]
        for override, bad in (
            ({"columns": {"id": "id", "audio": "wav", "text": "text", "group": "group"}}, "wav"),
            ({"columns": {"id": "id", "audio": "audio", "text": "text", "group": "lesson"}}, "lesson"),
            ({"extra_types": {"nothere": "int"}}, "nothere"),
        ):
            with self.assertRaises(DataSourceError) as ctx:
                self.source(rows, **override).load()
            message = str(ctx.exception)
            self.assertIn(bad, message)
            self.assertIn("Existing columns: id, audio, text, group", message)

    def test_no_group_column_fails_with_three_fixes_unless_fallback_is_set(self) -> None:
        cols = {"id": "id", "audio": "audio", "text": "text"}
        with self.assertRaises(DataSourceError) as ctx:
            self.source([self.row("a")], columns=cols).load()
        self.assertIn("leakage", str(ctx.exception))
        for fix in ("(1)", "(2)", "(3)"):
            self.assertIn(fix, str(ctx.exception))
        sub = self.data / "lesson1"
        sub.mkdir()
        (sub / "clip.wav").write_bytes(wav_bytes())
        rows = [self.row("a", audio="lesson1/clip.wav")]
        self.assertEqual(self.source(rows, columns=cols, group_fallback="parent_dir").load().samples[0].group_id, "lesson1")
        self.assertEqual(self.source(rows, columns=cols, group_fallback="file_stem").load().samples[0].group_id, "clip")

    def test_without_id_column_the_audio_path_is_the_id(self) -> None:
        rows = [self.row("a")]
        sample = self.source(rows, columns={"audio": "audio", "text": "text", "group": "group"}).load().samples[0]
        self.assertEqual(sample.id, "a.wav")

    def test_5_rejected_ratio_gate(self) -> None:
        rows = [self.row("a"), self.row("b"), self.row("c", audio="missing.wav"), self.row("d")]
        self.assertEqual(self.source(rows, max_rejected_ratio=0.25).load().report.rejected, 1)  # equal to the limit passes
        with self.assertRaises(DataSourceError) as ctx:
            self.source(rows, max_rejected_ratio=0.2).load()
        self.assertIn("rejected.csv", str(ctx.exception))
        self.assertIn("missing_audio", str(ctx.exception))
        self.assertTrue((self.out / "rejected.csv").is_file())  # written before failing

    def test_zero_samples_fails(self) -> None:
        with self.assertRaises(DataSourceError):
            self.source([self.row("a", audio="missing.wav")]).load()

    def test_on_invalid_fail_stops_at_first_bad_sample(self) -> None:
        with self.assertRaises(DataSourceError) as ctx:
            self.source([self.row("a"), self.row("b", text="")], on_invalid="fail").load()
        self.assertIn("on_invalid", str(ctx.exception))

    def test_missing_manifest_fails(self) -> None:
        with self.assertRaises(DataSourceError) as ctx:
            CsvAudioSource(make_config(self.data), self.out).load()
        self.assertIn("manifest.csv", str(ctx.exception))

    def test_8_hebrew_text_bom_and_paths_with_spaces_and_hebrew(self) -> None:
        folder = self.data / "תיקייה עם רווחים"
        folder.mkdir()
        (folder / "קובץ אחד.wav").write_bytes(wav_bytes())
        rows = [self.row("a", audio="תיקייה עם רווחים/קובץ אחד.wav", text="אמר רבא: שלום, עולם?")]
        write_manifest(self.data, rows, bom=True)
        cols = {"id": "id", "audio": "audio", "text": "text", "group": "group", "split": "split"}
        sample = CsvAudioSource(make_config(self.data, columns=cols), self.out).load().samples[0]
        self.assertEqual(sample.text, "אמר רבא: שלום, עולם?")
        self.assertEqual(sample.audio_path, self.data / "תיקייה עם רווחים" / "קובץ אחד.wav")

    def test_relative_paths_resolve_relative_to_manifest_and_absolute_work(self) -> None:
        absolute = self.add_wav("abs.wav")
        rows = [self.row("a", audio=str(self.data / absolute)), self.row("b")]
        samples = self.source(rows).load().samples
        self.assertEqual([s.audio_path.name for s in samples], ["abs.wav", "b.wav"])

    def test_text_is_passed_raw(self) -> None:
        raw = "  <|0.00|>טקסט  עם   רווחים <|1.00|> \n"
        self.assertEqual(self.source([self.row("a", text=raw)]).load().samples[0].text, raw)

    def test_9_input_files_are_unchanged(self) -> None:
        source = self.source([self.row("a"), self.row("b", audio="missing.wav")])
        before = hashes(self.data)
        source.load()
        self.assertEqual(hashes(self.data), before)

    def test_read_duration_rejects_missing_and_garbage(self) -> None:
        with self.assertRaises(ValueError):
            read_duration_seconds(self.data / "nope.wav")


class ParquetTestCase(unittest.TestCase):
    META = pa.struct([("entry_id", pa.string()), ("quality_score", pa.float32()), ("duration", pa.float32()), ("golden_segments", pa.bool_())])

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.src = self.tmp / "raw"
        self.src.mkdir()
        self.out = self.tmp / "out"
        self.extract = self.tmp / "extracted"

    def write_parquet(self, name: str, rows: list[tuple[bytes | None, str, str]], text_col: str = "transcript") -> None:
        table = pa.table(
            {
                "audio": pa.array([{"bytes": b, "path": "same-path"} for b, _, _ in rows], pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                text_col: [t for _, t, _ in rows],
                "metadata": pa.array([{"entry_id": e, "quality_score": 0.5, "duration": 30.0, "golden_segments": False} for _, _, e in rows], self.META),
                "has_prev": [True] * len(rows),
            }
        )
        pq.write_table(table, self.src / name, row_group_size=2)

    def populate(self) -> None:
        self.write_parquet("train-00000-of-00002.parquet", [(wav_bytes(1.0), "אחד", "e1"), (wav_bytes(2.0), "שניים", "e1"), (wav_bytes(1.5), "שלוש", "e2")])
        self.write_parquet("train-00001-of-00002.parquet", [(wav_bytes(1.0), "ארבע", "e3")])
        self.write_parquet("eval-00000-of-00001.parquet", [(wav_bytes(1.0), "חמש", "e1")])

    def config(self, **overrides: Any) -> DataConfig:
        base: dict[str, Any] = {
            "source": {"type": "hf_parquet", "path": str(self.src)},
            "columns": {"audio": "audio", "text": "transcript", "group": "meta_entry_id", "split": "source_split"},
            "extra_types": {"meta_quality_score": "float", "meta_golden_segments": "bool", "has_prev": "bool"},
            "extract": {"dir": str(self.extract)},
        }
        base.update(overrides)
        return DataConfig.model_validate(base)

    def load(self, **overrides: Any) -> Any:
        return HfParquetSource(self.config(**overrides), self.out).load()


class ParquetSourceTests(ParquetTestCase):
    def test_6_extraction_creates_files_manifest_flattened_metadata_and_deterministic_ids(self) -> None:
        self.populate()
        result = self.load()
        self.assertEqual([s.id for s in result.samples], ["eval_000000", "train_000000", "train_000001", "train_000002", "train_000003"])
        self.assertTrue((self.extract / "manifest.csv").is_file())
        self.assertTrue((self.extract / "extract_info.json").is_file())
        self.assertTrue((self.extract / "audio" / "train" / "train_000003.wav").is_file())  # running index over sorted files
        train3 = result.samples[-1]
        self.assertEqual((train3.group_id, train3.source_split, train3.text), ("e3", "train", "ארבע"))
        self.assertEqual(dict(train3.extra), {"meta_quality_score": 0.5, "meta_duration": "30.0", "meta_golden_segments": False, "has_prev": True})
        self.assertEqual(result.report.loaded_per_split, {"eval": 1, "train": 4})
        self.assertEqual(result.report.groups_in_several_splits, 1)  # e1 is in train and eval
        self.assertAlmostEqual(result.samples[2].duration_sec, 2.0, places=3)  # duration from the header
        self.assertEqual(self.load().samples, result.samples)

    def test_6_second_run_reuses_extraction(self) -> None:
        self.populate()
        self.load()
        marker = self.extract / "audio" / "train" / "marker.txt"
        marker.write_text("kept", encoding="utf-8")
        before = (self.extract / "manifest.csv").stat().st_mtime_ns
        self.load()
        self.assertTrue(marker.exists())
        self.assertEqual((self.extract / "manifest.csv").stat().st_mtime_ns, before)

    def test_6_overwrite_true_extracts_again(self) -> None:
        self.populate()
        self.load()
        marker = self.extract / "audio" / "train" / "marker.txt"
        marker.write_text("gone", encoding="utf-8")
        self.load(extract={"dir": str(self.extract), "overwrite": True})
        self.assertFalse(marker.exists())

    def test_6_manifest_less_folder_is_extracted_again(self) -> None:
        self.populate()
        self.load()
        (self.extract / "manifest.csv").unlink()  # simulates a crash before the manifest was written
        stale = self.extract / "audio" / "train" / "stale.txt"
        stale.write_text("old", encoding="utf-8")
        result = self.load()
        self.assertEqual(len(result.samples), 5)
        self.assertFalse(stale.exists())
        self.assertTrue((self.extract / "manifest.csv").is_file())

    def test_changed_source_files_are_detected_when_reusing(self) -> None:
        self.populate()
        self.load()
        self.write_parquet("eval-00000-of-00001.parquet", [(wav_bytes(1.0), "חמש", "e1"), (wav_bytes(1.0), "שש", "e9")])
        with self.assertRaises(DataSourceError) as ctx:
            self.load()
        self.assertIn("overwrite", str(ctx.exception))

    def test_7_both_sources_return_equal_samples_for_equivalent_data(self) -> None:
        self.populate()
        from_parquet = self.load().samples
        csv_config = DataConfig.model_validate(
            {
                "source": {"type": "csv", "path": str(self.extract)},
                "columns": {"id": "id", "audio": "audio", "text": "transcript", "group": "meta_entry_id", "split": "source_split"},
                "extra_types": {"meta_quality_score": "float", "meta_golden_segments": "bool", "has_prev": "bool"},
            }
        )
        from_csv = CsvAudioSource(csv_config, self.out).load().samples
        self.assertEqual(from_parquet, from_csv)

    def test_null_audio_row_is_rejected_as_missing_audio(self) -> None:
        self.write_parquet("train-00000-of-00001.parquet", [(wav_bytes(), "a", "e1"), (None, "b", "e2"), (wav_bytes(), "c", "e3")])
        result = self.load(max_rejected_ratio=0.5)
        self.assertEqual([(r.id, r.reason) for r in result.rejections], [("train_000001", "missing_audio")])

    def test_no_parquet_files_fails(self) -> None:
        with self.assertRaises(DataSourceError) as ctx:
            self.load()
        self.assertIn("No .parquet files", str(ctx.exception))

    def test_unknown_parquet_column_fails_and_lists_columns(self) -> None:
        self.write_parquet("train-0.parquet", [(wav_bytes(), "a", "e1")], text_col="other_text")
        with self.assertRaises(DataSourceError) as ctx:
            self.load()
        self.assertIn("transcript", str(ctx.exception))
        self.assertIn("other_text", str(ctx.exception))

    def test_9_source_parquet_files_are_unchanged(self) -> None:
        self.populate()
        before = hashes(self.src)
        self.load()
        self.assertEqual(hashes(self.src), before)


class FactoryAndConfigTests(CsvTestCase):
    def test_registry_selects_by_name_and_lists_names(self) -> None:
        self.assertEqual(default_registry().names(), ["csv", "hf_parquet"])
        write_manifest(self.data, [self.row("a")])
        cols = {"id": "id", "audio": "audio", "text": "text", "group": "group", "split": "split"}
        source = create_data_source(make_config(self.data, columns=cols), self.out)
        self.assertIsInstance(source, CsvAudioSource)
        with self.assertRaises(DataSourceError) as ctx:
            create_data_source(make_config(self.data, source={"type": "xml", "path": str(self.data)}), self.out)
        self.assertIn("csv, hf_parquet", str(ctx.exception))

    def test_section_loads_through_the_config_loader_and_reports_problems(self) -> None:
        registry = SectionRegistry({"data": DataConfig})
        base = [f"data.source.type=csv", f"data.source.path={self.data}", "data.columns.audio=audio", "data.columns.text=text"]
        config = load_config(overrides=base, registry=registry, environ={}, env_file=None, target_environ={})
        data = config.section("data", DataConfig)
        self.assertEqual((data.group_fallback, data.on_invalid, data.max_rejected_ratio, data.extract.overwrite), ("fail", "skip", 0.05, False))
        with self.assertRaises(ConfigError) as ctx:
            load_config(overrides=base[:3], registry=registry, environ={}, env_file=None, target_environ={})
        self.assertIn("data.columns.text", str(ctx.exception))
        with self.assertRaises(ConfigError) as ctx2:
            load_config(overrides=[*base, "data.source.type=hf_parquet"], registry=registry, environ={}, env_file=None, target_environ={})
        self.assertIn("data.extract.dir", str(ctx2.exception))
        with self.assertRaises(ConfigError) as ctx3:
            load_config(overrides=[*base, "data.max_rejected_ratio=1.5"], registry=registry, environ={}, env_file=None, target_environ={})
        self.assertIn("data.max_rejected_ratio", str(ctx3.exception))

    def test_shipped_example_config_loads(self) -> None:
        raw = (ROOT / "configs" / "data_gmara.yaml").read_text(encoding="utf-8")
        self.assertIn("meta_entry_id", raw)


class ImportRuleTests(unittest.TestCase):
    def test_10_pyarrow_is_imported_only_in_the_parquet_adapter_and_no_network_modules(self) -> None:
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
                    if top == "pyarrow":
                        self.assertEqual(path.name, "parquet_source.py", f"{path} imports pyarrow")
                    self.assertNotIn(top, {"requests", "urllib3", "httpx", "huggingface_hub", "socket", "tools"}, f"{path} imports {name}")

    def test_the_datasource_makes_no_decisions_about_splits_or_text(self) -> None:
        # Out of scope by the brief: no module of the component may import resampling or text-cleaning code.
        for path in (ROOT / "src" / "stt" / "datasource").glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for forbidden in ("import scipy", "import torch", "resample"):
                self.assertNotIn(forbidden, text, f"{path} contains {forbidden}")


if __name__ == "__main__":
    unittest.main()
