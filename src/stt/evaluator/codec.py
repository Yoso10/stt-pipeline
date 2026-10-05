"""A lossless JSON codec for `EvalResult` and `ComparisonResult` (standard library only; DECISIONS 52, O82).

How to run: this module is not a script; the stages use it to keep a full result between commands. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.evaluator import ComparisonResult, comparison_to_json, comparison_from_json; c = ComparisonResult(0.05, 0.01, 0.09, 0.9, True, 12, True, 'improved', 'ok'); print(comparison_from_json(comparison_to_json(c)) == c)"
Example output:
    True
`eval_result_to_json(result)` gives UTF-8 safe JSON text and `eval_result_from_json(text)` gives back an `EvalResult` that is EQUAL to the original: every per-file score with its
group, attributes and extra values, the decode settings with the per-file token caps, the review list, the flags and the model description are kept (the numbers are written
with their full precision, the float keys of `share_files_wer_below` as pairs). Both files carry a `codec` name and a `schema_version`; an unknown version, a damaged file or a
missing key FAILS with `EvaluatorError` (exit code 4) and the file is never "repaired". The stored text is what the later stages (selection, registration) read; the human
reports (`evaluation_*.json`, `predictions_*.csv`, `review_*.txt`) are separate and unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stt.evaluator.errors import EvaluatorError
from stt.evaluator.types import ComparisonResult, DecodeSettings, EvalResult, FileScore, ReviewEntry, ReviewList
from stt.modelloader import ModelInfo

SCHEMA_VERSION = 1
EVAL_CODEC = "stt.eval_result"
COMPARISON_CODEC = "stt.comparison_result"


def eval_result_to_json(result: EvalResult) -> str:
    """The full `EvalResult` as JSON text (see the module docstring). Errors: none for a well-formed result."""
    m, d = result.model_info, result.decode_settings
    body: dict[str, Any] = {
        "codec": EVAL_CODEC,
        "schema_version": SCHEMA_VERSION,
        "split_name": result.split_name,
        "eval_version": result.eval_version,
        "model_info": {
            "name": m.name,
            "path": str(m.path),
            "num_parameters": m.num_parameters,
            "dtype": m.dtype,
            "num_mel_bins": m.num_mel_bins,
            "language": m.language,
            "task": m.task,
            "source_config_ref": m.source_config_ref,
        },
        "wer": result.wer,
        "wer_mean_per_file": result.wer_mean_per_file,
        "cer": result.cer,
        "n_files": result.n_files,
        "n_words": result.n_words,
        "share_files_wer_below": [[threshold, share] for threshold, share in result.share_files_wer_below.items()],
        "files": [
            {
                "sample_id": f.sample_id,
                "group_id": f.group_id,
                "reference": f.reference,
                "hypothesis": f.hypothesis,
                "hypothesis_raw": f.hypothesis_raw,
                "ref_words": f.ref_words,
                "word_errors": f.word_errors,
                "wer": f.wer,
                "cer": f.cer,
                "empty_output": f.empty_output,
                "hit_length_cap": f.hit_length_cap,
                "attributes": dict(f.attributes),
                "extra": dict(f.extra),
            }
            for f in result.files
        ],
        "decode_settings": {
            "num_beams": d.num_beams,
            "max_new_tokens": d.max_new_tokens,
            "batch_size": d.batch_size,
            "max_tokens_per_audio_sec": d.max_tokens_per_audio_sec,
            "token_caps": None if d.token_caps is None else dict(d.token_caps),
        },
        "seconds": result.seconds,
        "flags": dict(result.flags),
        "excluded_ids": list(result.excluded_ids),
        "review": [
            {
                "sample_id": e.sample_id,
                "group_id": e.group_id,
                "audio_path": str(e.audio_path),
                "reference": e.reference,
                "hypothesis": e.hypothesis,
                "wer": e.wer,
                "diff": e.diff,
                "reason": e.reason,
            }
            for e in result.review.entries
        ],
        "extra_metrics": dict(result.extra_metrics),
    }
    return json.dumps(body, ensure_ascii=False, indent=1)


def eval_result_from_json(text: str) -> EvalResult:
    """The `EvalResult` stored by `eval_result_to_json`. Errors: EvaluatorError naming what is wrong (not JSON, another codec or version, a missing key)."""
    raw = _load(text, EVAL_CODEC)
    try:
        m, d = raw["model_info"], raw["decode_settings"]
        files = tuple(
            FileScore(
                f["sample_id"], f["group_id"], f["reference"], f["hypothesis"], f["hypothesis_raw"], f["ref_words"], f["word_errors"], f["wer"], f["cer"],
                f["empty_output"], f["hit_length_cap"], dict(f["attributes"]), dict(f["extra"]),
            )
            for f in raw["files"]
        )
        review = ReviewList(tuple(ReviewEntry(e["sample_id"], e["group_id"], Path(e["audio_path"]), e["reference"], e["hypothesis"], e["wer"], e["diff"], e["reason"]) for e in raw["review"]))
        return EvalResult(
            split_name=raw["split_name"],
            eval_version=raw["eval_version"],
            model_info=ModelInfo(m["name"], Path(m["path"]), m["num_parameters"], m["dtype"], m["num_mel_bins"], m["language"], m["task"], m["source_config_ref"]),
            wer=raw["wer"],
            wer_mean_per_file=raw["wer_mean_per_file"],
            cer=raw["cer"],
            n_files=raw["n_files"],
            n_words=raw["n_words"],
            share_files_wer_below={float(k): float(v) for k, v in raw["share_files_wer_below"]},
            files=files,
            decode_settings=DecodeSettings(d["num_beams"], d["max_new_tokens"], d["batch_size"], d["max_tokens_per_audio_sec"], None if d["token_caps"] is None else dict(d["token_caps"])),
            seconds=raw["seconds"],
            flags=dict(raw["flags"]),
            excluded_ids=tuple(raw["excluded_ids"]),
            review=review,
            extra_metrics=dict(raw["extra_metrics"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluatorError(f"The stored evaluation result is damaged ({type(exc).__name__}: {exc}). Run the evaluate stage again to write it anew.") from exc


def comparison_to_json(comparison: ComparisonResult) -> str:
    """The `ComparisonResult` as JSON text."""
    body: dict[str, Any] = {
        "codec": COMPARISON_CODEC,
        "schema_version": SCHEMA_VERSION,
        "delta_wer": comparison.delta_wer,
        "ci_low": comparison.ci_low,
        "ci_high": comparison.ci_high,
        "confidence": comparison.confidence,
        "one_sided": comparison.one_sided,
        "n_groups": comparison.n_groups,
        "min_effect_met": comparison.min_effect_met,
        "verdict": comparison.verdict,
        "explanation": comparison.explanation,
    }
    return json.dumps(body, ensure_ascii=False, indent=1)


def comparison_from_json(text: str) -> ComparisonResult:
    """The `ComparisonResult` stored by `comparison_to_json`. Errors: EvaluatorError (see `eval_result_from_json`)."""
    raw = _load(text, COMPARISON_CODEC)
    try:
        return ComparisonResult(
            raw["delta_wer"], raw["ci_low"], raw["ci_high"], raw["confidence"], raw["one_sided"], raw["n_groups"], raw["min_effect_met"], raw["verdict"], raw["explanation"]
        )
    except (KeyError, TypeError) as exc:
        raise EvaluatorError(f"The stored comparison result is damaged ({type(exc).__name__}: {exc}). Run the evaluate stage again to write it anew.") from exc


def _load(text: str, codec: str) -> dict[str, Any]:
    try:
        raw = json.loads(text)
    except ValueError as exc:
        raise EvaluatorError(f"The stored result is not valid JSON ({exc}). Run the evaluate stage again to write it anew.") from exc
    if not isinstance(raw, dict) or raw.get("codec") != codec:
        raise EvaluatorError(f"The stored text is not a '{codec}' document (found: {raw.get('codec') if isinstance(raw, dict) else type(raw).__name__}).")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise EvaluatorError(f"The stored '{codec}' document has schema version {raw.get('schema_version')}, this code reads version {SCHEMA_VERSION}. Run the evaluate stage again.")
    return raw
