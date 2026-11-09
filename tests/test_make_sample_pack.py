"""Unit test of tools/make_sample_pack.py on tiny synthetic data (offline).

How to run (project root, virtual environment active):
    python -m unittest tests.test_make_sample_pack -v
The test writes a few one-second wav files and a manifest into a temporary folder, builds packs from them and checks the selection rules:
eligibility, many small groups, independence of the row order, byte-identical output, the audio-only checksum list, the refusal to overwrite.
"""

from __future__ import annotations

import csv
import hashlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import soundfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import make_sample_pack as pack  # noqa: E402

HEADER = ["id", "audio", "source_split", "transcript", "lesson"]


def _wav(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), (0.1 * np.sin(np.arange(int(seconds * 8000)) / 5.0)).astype("float32"), 8000)


def _source(folder: Path, groups: int, per_group: int, extra_rows: list[list[str]] | None = None) -> list[list[str]]:
    rows = []
    for g in range(groups):
        for k in range(per_group):
            sample = f"clip_{g:02d}_{k:02d}"
            _wav(folder / "audio" / f"{sample}.wav", 1.0)
            rows.append([sample, f"audio/{sample}.wav", "train" if g % 2 else "eval", f"שלום {sample}", f"L{g:02d}"])
    rows += extra_rows or []
    with (folder / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return rows


def _run(*args: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = pack.main(list(args))
    return code, out.getvalue(), err.getvalue()


def _tree(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}


class SelectionTest(unittest.TestCase):
    def rows(self) -> list[dict[str, str]]:
        return [{"id": f"c{g}_{k}", "lesson": f"L{g}"} for g in range(12) for k in range(6 if g < 10 else 2)]

    def test_many_groups_few_clips_each(self) -> None:
        chosen = pack.select_clips(self.rows(), "id", "lesson", groups=5, per_group=3, seed=1)
        self.assertEqual(len(chosen), 15)
        self.assertEqual(len({r["lesson"] for r in chosen}), 5)

    def test_only_groups_with_enough_clips(self) -> None:
        chosen = pack.select_clips(self.rows(), "id", "lesson", groups=10, per_group=3, seed=1)
        self.assertEqual({r["lesson"] for r in chosen}, {f"L{g}" for g in range(10)})  # L10 and L11 have 2 clips only
        with self.assertRaisesRegex(ValueError, "only 10 groups"):
            pack.select_clips(self.rows(), "id", "lesson", groups=11, per_group=3, seed=1)

    def test_order_of_the_rows_does_not_matter(self) -> None:
        rows = self.rows()
        self.assertEqual(pack.select_clips(rows, "id", "lesson", 6, 4, 7), pack.select_clips(list(reversed(rows)), "id", "lesson", 6, 4, 7))

    def test_the_seed_changes_the_choice(self) -> None:
        a = pack.select_clips(self.rows(), "id", "lesson", 4, 3, 1)
        b = pack.select_clips(self.rows(), "id", "lesson", 4, 3, 2)
        self.assertNotEqual([r["id"] for r in a], [r["id"] for r in b])

    def test_ascii_name(self) -> None:
        self.assertEqual(pack.ascii_name("train_0001"), "train_0001")
        self.assertEqual(pack.ascii_name("שלום 1/a"), "_____1_a")


class PackTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.src = self.tmp / "src"
        _wav(self.src / "audio" / "long.wav", 3.0)
        extra = [
            ["long_clip", "audio/long.wav", "train", "ארוך", "L99"],  # longer than --max-seconds 2
            ["no_audio", "audio/missing.wav", "train", "אין", "L99"],
            ["empty_text", "audio/clip_00_00.wav", "train", "   ", "L99"],
            ["rejected_one", "audio/clip_00_01.wav", "train", "נדחה", "L99"],
        ]
        _source(self.src, groups=6, per_group=4, extra_rows=extra)
        with (self.tmp / "rejected.csv").open("w", encoding="utf-8", newline="") as handle:
            handle.write("id,reason,detail\nrejected_one,x,y\n")

    def build(self, out: Path, *extra: str) -> tuple[int, str, str]:
        return _run("--source", str(self.src), "--group-column", "lesson", "--groups", "4", "--per-group", "3", "--seed", "5", "--out", str(out), "--rejected", str(self.tmp / "rejected.csv"),
                    "--max-seconds", "2", "--date", "2026-01-02", *extra)

    def test_pack_layout_and_eligibility(self) -> None:
        out = self.tmp / "p1"
        code, stdout, _ = self.build(out)
        self.assertEqual(code, 0, stdout)
        with (out / "manifest.csv").open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, HEADER)  # every original column, same order
            rows = list(reader)
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({r["lesson"] for r in rows}), 4)
        self.assertNotIn("L99", {r["lesson"] for r in rows})  # the group of the four ineligible rows never qualifies
        self.assertEqual(len(list((out / "audio").glob("*.wav"))), 12)
        for r in rows:
            self.assertEqual(r["audio"], f"audio/{r['id']}.wav")
            self.assertTrue((out / r["audio"]).is_file())
        self.assertIn("שלום", rows[0]["transcript"])

    def test_checksums_cover_the_audio_only(self) -> None:
        out = self.tmp / "p2"
        self.build(out)
        lines = (out / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 12)
        for line in lines:
            digest, name = line.split("  ", 1)
            self.assertTrue(name.startswith("audio/"))
            self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), digest)
        self.assertNotIn(b"\r", (out / "SHA256SUMS.txt").read_bytes())

    def test_same_input_gives_identical_bytes(self) -> None:
        self.build(self.tmp / "a")
        self.build(self.tmp / "b")
        self.assertEqual(_tree(self.tmp / "a"), _tree(self.tmp / "b"))

    def test_nothing_is_overwritten(self) -> None:
        out = self.tmp / "p3"
        self.build(out)
        before = _tree(out)
        code, _, err = self.build(out)
        self.assertEqual(code, 3)
        self.assertIn("already exists", err)
        self.assertEqual(_tree(out), before)

    def test_too_many_groups_is_an_error_and_writes_nothing(self) -> None:
        out = self.tmp / "p4"
        code, _, err = _run("--source", str(self.src), "--group-column", "lesson", "--groups", "9", "--per-group", "3", "--out", str(out), "--max-seconds", "2")
        self.assertEqual(code, 2)
        self.assertIn("only 6 groups", err)
        self.assertFalse(out.exists())

    def test_report_only_writes_nothing(self) -> None:
        out = self.tmp / "p5"
        code, stdout, _ = _run("--source", str(self.src), "--group-column", "lesson", "--groups", "4", "--per-group", "3", "--out", str(out), "--max-seconds", "2", "--report-only")
        self.assertEqual(code, 0)
        self.assertIn("groups with at least 3 eligible clips: 6", stdout)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
