"""Minimal report texts of an evaluation (pure functions: they return text, the RunManager writes the files).

How to run: this module is not a script; the evaluate stage uses it. It is a stop-gap until the ReportBuilder (Wave 3): plain JSON, CSV and text
files that can be opened in any editor or Excel. Files written per scored model `<name>` (`baseline` or `trained`) into `reports/`:
    evaluation_<name>.json   the corpus WER, CER, counts, flags, decode settings
    predictions_<name>.csv   one row per file: id, group, reference, hypothesis, wer, cer, flags (UTF-8 with a byte order mark so Excel reads Hebrew)
    review_<name>.txt        the human review list: worst files first, then random ones, with the word-level difference and the audio path
and `comparison.json` with the verdict. The `search` stage adds (pure functions below, DECISIONS 55):
    search_report.txt        what the search did: the trials in a table, the finalists, the notes (plain text)
    search_trials.json       the same as data: the space, the screening set, every trial with its parameters, status, Val loss and reports, the finalist trial ids
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict
from typing import Any

from stt.evaluator import ComparisonResult, EvalResult
from stt.orchestrator.types import ScreeningSet
from stt.searcher import SearchResult, SearchSpec, TrialRecord


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


def _loss(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _params(params: object) -> str:
    return ", ".join(f"{k}={v:g}" if isinstance(v, int | float) else f"{k}={v}" for k, v in dict(params).items())  # type: ignore[call-overload]


def search_report_text(result: SearchResult, screening: ScreeningSet, unrecorded_folders: list[str]) -> str:
    """The search as plain text: one line per trial, the finalists and the notes. `unrecorded_folders` are trial folders in `candidates/` that the study does not list."""
    lines = [result.report.summary(), f"  stopped because: {result.stopped_reason}", f"  screening data: {len(screening.train)} Train and {len(screening.val)} Val samples (see search_screening.txt)", ""]
    lines.append("trial  status    Val loss   steps  seconds  parameters")
    for r in sorted(result.trials, key=lambda r: r.trial.trial_id):
        o = r.outcome
        note = "" if o.error is None else f"   [{o.error}]"
        lines.append(f"{r.trial.trial_id:>5}  {o.status:<8}  {_loss(o.val_loss):>8}  {o.steps:>6}  {o.seconds:>7.1f}  {_params(r.trial.params)}{note}")
    lines += ["", f"finalists (best {len(result.ranked)} by Val loss; they are retrained and scored by WER in a later stage):"]
    lines += [f"  {i}. trial {r.trial.trial_id}  Val loss {_loss(r.outcome.val_loss)}  {_params(r.trial.params)}" for i, r in enumerate(result.ranked, 1)] or ["  none"]
    if unrecorded_folders:
        lines += ["", f"trial folders in candidates/ that the study does not list (an interrupted trial of a parallel search, or an old attempt renamed to _stale_N): {', '.join(unrecorded_folders)}"]
        lines.append("  They hold checkpoints and use disk space; a later `cleanup` proposes them, nothing deletes them by itself.")
    return "\n".join(lines) + "\n"


def _record(r: TrialRecord) -> dict[str, Any]:
    o = r.outcome
    return {
        "trial_id": r.trial.trial_id,
        "params": dict(r.trial.params),
        "status": o.status,
        "val_loss": o.val_loss,
        "steps": o.steps,
        "seconds": o.seconds,
        "model_dir": None if o.model_dir is None else str(o.model_dir),
        "error": o.error,
        "val_loss_reports": [list(x) for x in o.reports],
    }


def search_trials_json(result: SearchResult, spec: SearchSpec, screening: ScreeningSet) -> str:
    """The search as JSON data (what a later stage reads: the finalists and the model folders of the trials)."""
    body: dict[str, Any] = {
        "strategy": spec.strategy,
        "seed": spec.seed,
        "stopped_reason": result.stopped_reason,
        "space": {n: {"kind": p.kind, "low": p.low, "high": p.high, "choices": list(p.choices), "points": p.points} for n, p in spec.parameters.items()},
        "screening": {"train_split": screening.train_split, "val_split": screening.val_split, "n_train": len(screening.train), "n_val": len(screening.val), "fingerprint": screening.fingerprint},
        "trials": [_record(r) for r in sorted(result.trials, key=lambda r: r.trial.trial_id)],
        "finalists": [r.trial.trial_id for r in result.ranked],
        "best_val_loss": result.report.best_val_loss,
        "best_params": None if result.report.best_params is None else dict(result.report.best_params),
        "notes": list(result.report.notes),
    }
    return json.dumps(body, indent=1, ensure_ascii=False)
