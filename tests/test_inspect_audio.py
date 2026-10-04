"""Unit test of tools/inspect_audio.py on a tiny synthetic parquet dataset (offline).

How to run (project root, virtual environment active):
    python -m unittest tests.test_inspect_audio -v
The test builds two small parquet files with generated audio (wav, mp3, one broken row) in a
temporary folder. Nothing is downloaded.
"""

from __future__ import annotations

import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import inspect_audio  # noqa: E402


def _audio_bytes(seconds: float, sr: int, fmt: str) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    buffer = io.BytesIO()
    soundfile.write(buffer, (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32"), sr, format=fmt)
    return buffer.getvalue()


def _write(path: Path, rows: list[tuple[bytes, str, str]]) -> None:
    meta_type = pa.struct([("entry_id", pa.string()), ("duration", pa.float32()), ("source", pa.string()),
                           ("creator", pa.string()), ("quality_score", pa.float32()), ("seek", pa.float32()),
                           ("number_of_segments", pa.int32()), ("bad_segments", pa.int32()), ("golden_segments", pa.bool_())])  # fmt: skip
    table = pa.table({
        "audio": pa.array([{"bytes": b, "path": "x"} for b, _, _ in rows], pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        "transcript": [t for _, t, _ in rows],
        "metadata": pa.array([{"entry_id": e, "duration": 1.0, "source": "s", "creator": "c", "quality_score": 0.5,
                               "seek": 0.0, "number_of_segments": 1, "bad_segments": 0, "golden_segments": False}
                              for _, _, e in rows], meta_type),
        "has_prev": [False] * len(rows),
        "has_timestamps": [True] * len(rows),
    })  # fmt: skip
    pq.write_table(table, path, row_group_size=2)


class InspectAudioTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data = Path(self._tmp.name)
        mp3 = _audio_bytes(1.0, 16000, "MP3")
        _write(self.data / "train-00000-of-00001.parquet", [
            (mp3, "<|0.00|>שלום עולם. <|1.00|>", "e1"),
            (_audio_bytes(2.0, 16000, "WAV"), "<|0.00|>hello 123 (x) <|2.00|>", "e2"),
            (b"not audio", "<|0.00|>כו׳ <|0.50|>", "e3"),
            (_audio_bytes(31.0, 16000, "WAV"), "<|0.00|><|1.00|>", "e4"),
        ])  # fmt: skip
        _write(self.data / "eval-00000-of-00001.parquet", [(mp3, "<|0.00|>טקסט, כן? <|1.00|>", "e1")])

    def test_schema_report_counts_rows_from_metadata(self) -> None:
        lines = inspect_audio.schema_report(inspect_audio.find_parquet_files(self.data))
        self.assertIn("TOTAL rows: 5", lines)
        self.assertTrue(any("column audio" in line for line in lines))

    def test_overlap_report_finds_shared_entry_id(self) -> None:
        lines = inspect_audio.overlap_report(inspect_audio.find_parquet_files(self.data))
        self.assertTrue(any("shared by eval and train: 1" in line for line in lines), lines)

    def test_audio_report_counts_failures_and_long_clips(self) -> None:
        files = inspect_audio.find_parquet_files(self.data)
        text = "\n".join(inspect_audio.audio_report(files, 50, 0))
        self.assertIn("decoded 4, failed 1", text)
        self.assertIn("DECODE FAILURE", text)
        self.assertIn("longer than 30 s: 1 of 4", text)
        self.assertIn("{16000: 4}", text)
        self.assertIn("resample_poly", text)

    def test_choose_rows_is_deterministic(self) -> None:
        self.assertEqual(inspect_audio.choose_rows([10, 10], 5, 7), inspect_audio.choose_rows([10, 10], 5, 7))
        self.assertEqual(len(inspect_audio.choose_rows([3], 50, 0)), 3)

    def test_text_flags_mask_timestamps_first(self) -> None:
        plain = inspect_audio.text_flags("<|0.00|>שלום עולם <|1.00|>")
        self.assertTrue(plain["timestamp_like"])
        self.assertFalse(plain["digits"] or plain["brackets"] or plain["punctuation"])
        rich = inspect_audio.text_flags("<|0.00|>a, 12 (b) כו׳ <|1.00|>")
        for key in ("punctuation", "digits", "brackets", "latin_letters", "geresh_or_gershayim"):
            self.assertTrue(rich[key], key)
        self.assertTrue(inspect_audio.text_flags("<|0.00|><|1.00|>")["empty_after_masking_timestamps"])

    def test_full_report_runs_and_changes_nothing(self) -> None:
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.data.iterdir()}
        report = inspect_audio.build_report(self.data, seed=0, audio_samples=50, text_samples=3)
        for title in ("A. Parquet", "B. Metadata", "B2.", "C. Audio", "D. Label texts"):
            self.assertIn(title, report)
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.data.iterdir()}
        self.assertEqual(before, after)
        self.assertEqual(sorted(after), sorted(before))

    def test_main_exit_codes(self) -> None:
        self.assertEqual(inspect_audio.main(["--data", str(self.data), "--audio-samples", "2", "--text-samples", "1"]), 0)
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(inspect_audio.main(["--data", empty]), 1)


if __name__ == "__main__":
    unittest.main()
