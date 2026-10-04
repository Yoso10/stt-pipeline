"""Minimal report texts of an evaluation (pure functions: they return text, the RunManager writes the files).

How to run: this module is not a script; the evaluate stage uses it. It is a stop-gap until the ReportBuilder (Wave 3): plain JSON, CSV and text
files that can be opened in any editor or Excel. Files written per scored model `<name>` (`baseline` or `trained`) into `reports/`:
    evaluation_<name>.json   the corpus WER, CER, counts, flags, decode settings
    predictions_<name>.csv   one row per file: id, group, reference, hypothesis, wer, cer, flags (UTF-8 with a byte order mark so Excel reads Hebrew)
    review_<name>.txt        the human review list: worst files first, then random ones, with the word-level difference and the audio path
and `comparison.json` with the verdict.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict
from typing import Any

from stt.evaluator import ComparisonResult, EvalResult


def evaluation_json(result: EvalResult) -> str:
    """The numbers of one evaluation as JSON text."""
    body: dict[str, Any] = {
        "split": result.split_name,
        "eval_version": result.eval_version,
        "model": result.model_info.name,
        "wer": result.wer,
        "wer_mean_per_file": result.wer_mean_per_file,
        "cer": result.cer,
        "n_files": result.n_files,
        "n_words": result.n_words,
        "share_files_wer_below": {str(k): v for k, v in result.share_files_wer_below.items()},
        "flags": dict(result.flags),
        "excluded_ids": list(result.excluded_ids),
        "decode_settings": {
            "num_beams": result.decode_settings.num_beams,
            "max_new_tokens": result.decode_settings.max_new_tokens,
            "batch_size": result.decode_settings.batch_size,
            "max_tokens_per_audio_sec": result.decode_settings.max_tokens_per_audio_sec,
        },
        "seconds": result.seconds,
    }
    return json.dumps(body, indent=1, ensure_ascii=False)


def predictions_csv(result: EvalResult) -> str:
    """One row per scored file (the table that shows where the model fails)."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["id", "group", "reference", "hypothesis", "wer", "cer", "empty_output", "hit_length_cap"])
    for f in result.files:
        writer.writerow([f.sample_id, f.group_id, f.reference, f.hypothesis, f"{f.wer:.4f}", "" if f.cer is None else f"{f.cer:.4f}", f.empty_output, f.hit_length_cap])
    return "﻿" + out.getvalue()


def review_text(result: EvalResult) -> str:
    """The review list as plain text, to read and to listen along."""
    lines = [f"Review list: {result.model_info.name} on {result.split_name} (worst files first, then random files). Listen to the audio file to judge the text.", ""]
    for e in result.review.entries:
        lines += [f"[{e.reason}] {e.sample_id}  WER {e.wer:.2f}  group {e.group_id}", f"  audio: {e.audio_path}", f"  REF : {e.reference}", f"  HYP : {e.hypothesis}", f"  DIFF: {e.diff}", ""]
    return "\n".join(lines)


def comparison_json(comparison: ComparisonResult) -> str:
    """The comparison as JSON text."""
    return json.dumps(asdict(comparison), indent=1, ensure_ascii=False)
