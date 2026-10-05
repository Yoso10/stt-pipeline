"""Model cards: the JSON codec, the human-readable MODEL_CARD.md, and the helpers that build a card from a SelectionResult (pure functions, no files).

How to run: this module is not a script; the registry and the Orchestrator use it. Example (PowerShell, project root; `selection` is a SelectionResult):
    $env:PYTHONPATH = "src"
    python -c "from stt.registry import card_from_selection, render_markdown; print(card_from_selection, render_markdown)"
FORMAT OF THE VAL AND TEST SUMMARIES (DECISIONS 53): a flat `EvalSummary` of numbers (WER, CER, mean per-file WER, files, words, the share of files under each WER
threshold, the baseline WER, the difference, the interval, the confidence, the verdict and the number of groups). No per-file data, so a card stays small and
self-contained; the ReportBuilder reads these keys from `model_card.json`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from stt.evaluator import ComparisonResult, EvalResult, ReviewList
from stt.registry.types import EvalSummary, FinalistRow, Kind, ModelCard
from stt.selector import SelectionResult

SCHEMA_VERSION = 1


# ---- fingerprint and summaries ---------------------------------------------------------------------------------------
def test_set_fingerprint(result: EvalResult) -> str:
    """The fingerprint of an evaluated set: SHA-256 over the sorted lines `sample_id|group_id|first 16 hex characters of SHA-256 of the reference`.

    The same files with the same reference texts give the same fingerprint, whatever the model, the order or the decoding; a changed file list, group or
    reference text (a new `eval_version` profile changes the references) gives another one. Excluded files (empty references) are part of the set.
    """
    lines = [f"{f.sample_id}|{f.group_id}|{hashlib.sha256(f.reference.encode('utf-8')).hexdigest()[:16]}" for f in result.files]
    lines += [f"{i}|excluded|-" for i in result.excluded_ids]
    return hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()


test_set_fingerprint.__test__ = False  # type: ignore[attr-defined]  # not a unittest test, in spite of the name


def summary_from_eval(result: EvalResult, comparison: ComparisonResult | None = None, baseline_wer: float | None = None) -> EvalSummary:
    """The flat summary of an evaluation (and, when given, of its comparison with the baseline)."""
    return EvalSummary(
        split=result.split_name,
        model=result.model_info.name,
        n_files=result.n_files,
        n_words=result.n_words,
        wer=result.wer,
        cer=result.cer,
        wer_mean_per_file=result.wer_mean_per_file,
        share_files_wer_below={f"{k:g}": v for k, v in result.share_files_wer_below.items()},
        baseline_wer=baseline_wer,
        delta_wer_vs_baseline=None if comparison is None else comparison.delta_wer,
        ci_low=None if comparison is None else comparison.ci_low,
        ci_high=None if comparison is None else comparison.ci_high,
        confidence=None if comparison is None else comparison.confidence,
        verdict=None if comparison is None else comparison.verdict,
        n_groups=None if comparison is None else comparison.n_groups,
    )


def card_from_selection(
    selection: SelectionResult,
    baseline_val: EvalResult,
    *,
    base_model_name: str,
    base_model_path: str,
    data_fingerprint: str | None,
    split_fingerprint: str | None,
    subsample_fingerprint: str | None,
    config_ref: str,
    library_versions: dict[str, str],
    seed: int,
    code_version: str,
    generated_at: str,
) -> ModelCard:
    """Build the card of a selection: the kind, the Val and Test summaries, the decode settings, the finalists, the reason and the flags.

    The caller passes what only it knows (the base model, the fingerprints, the config reference, the versions, the seed, the code version, the time).
    """
    winner = selection.winner
    kind: Kind = "model" if winner is not None else "baseline_retained"
    flags = list(selection.warnings)
    if selection.test_status == "not_confirmed":
        flags.append("the Test does not confirm the Val result (flagged, not dropped)")
    if selection.test_status == "not_evaluated":
        flags.append("test_not_evaluated (the Test was not evaluated)" + (f": {selection.test_error}" if selection.test_error else ""))
    if winner is not None:
        val = summary_from_eval(winner.val, selection.ranking[0].comparison, baseline_val.wer)
        settings = winner.val.decode_settings
    else:
        val = summary_from_eval(baseline_val)
        settings = baseline_val.decode_settings
    test = None
    test_fp = None
    if winner is not None and selection.test_winner is not None and selection.test_baseline is not None:
        test = summary_from_eval(selection.test_winner, selection.test_comparison, selection.test_baseline.wer)
        test_fp = test_set_fingerprint(selection.test_winner)
    finalists = [
        FinalistRow(r.trial_id, dict(r.params), r.val_wer, r.delta_wer, r.verdict, r.val_loss, r.steps, r.eligible, r.note) for r in selection.comparison_table
    ]
    return ModelCard(
        kind=kind,
        verdict=selection.verdict,
        base_model_name=base_model_name,
        base_model_path=base_model_path,
        data_fingerprint=data_fingerprint,
        split_fingerprint=split_fingerprint,
        subsample_fingerprint=subsample_fingerprint,
        test_fingerprint=test_fp,
        config_ref=config_ref,
        eval_version=baseline_val.eval_version,
        val=val,
        test=test,
        test_status=selection.test_status,
        decode_settings={"num_beams": settings.num_beams, "max_new_tokens": settings.max_new_tokens, "batch_size": settings.batch_size, "max_tokens_per_audio_sec": settings.max_tokens_per_audio_sec},
        searched_params=dict(winner.params) if winner is not None else {},
        finalists=finalists,
        library_versions=dict(library_versions),
        seed=seed,
        code_version=code_version,
        generated_at=generated_at,
        selector_reason=selection.reason,
        warnings=tuple(flags),
        trial_id=winner.trial_id if winner is not None else None,
        steps=winner.steps if winner is not None else None,
        val_loss=winner.val_loss if winner is not None else None,
    )


# ---- JSON codec ----------------------------------------------------------------------------------------------------------
def card_to_json(card: ModelCard, entry_id: str, experiment_id: str) -> str:
    """The text of `model_card.json` (the entry id and the experiment id are part of it)."""
    body: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "entry_id": entry_id, "experiment_id": experiment_id, **asdict(card)}
    return json.dumps(body, indent=1, ensure_ascii=False, sort_keys=False)


def _summary(raw: dict[str, Any] | None) -> EvalSummary | None:
    return None if raw is None else EvalSummary(**raw)


def card_from_json(text: str) -> tuple[str, str, ModelCard]:
    """Read `model_card.json`: (entry id, experiment id, card). Errors: ValueError or KeyError or TypeError for a damaged file."""
    raw = json.loads(text)
    entry_id, experiment_id = raw.pop("entry_id"), raw.pop("experiment_id")
    raw.pop("schema_version")
    raw["val"], raw["test"] = _summary(raw["val"]), _summary(raw["test"])
    raw["finalists"] = tuple(FinalistRow(**row) for row in raw["finalists"])
    raw["warnings"] = tuple(raw["warnings"])
    return str(entry_id), str(experiment_id), ModelCard(**raw)


# ---- the human-readable card -------------------------------------------------------------------------------------------------
def _number(value: float | None, spec: str = ".4f") -> str:
    return "-" if value is None else format(value, spec)


def _params(params: Any) -> str:
    return ", ".join(f"{k} = {v:g}" if isinstance(v, int | float) else f"{k} = {v}" for k, v in params.items()) or "none searched"


def _summary_lines(title: str, s: EvalSummary | None) -> list[str]:
    if s is None:
        return [f"### {title}", "", "Not evaluated.", ""]
    interval = "-" if s.ci_low is None else f"{_number(s.ci_low)} to {_number(s.ci_high)} ({(s.confidence or 0) * 100:g}% level)"
    shares = ", ".join(f"under {float(k):.0%}: {v:.1%}" for k, v in s.share_files_wer_below.items()) or "-"
    rows = [
        f"### {title}",
        "",
        "| item | value |",
        "|---|---|",
        f"| split | {s.split} |",
        f"| files / words | {s.n_files} / {s.n_words} |",
        f"| WER (all words together) | {_number(s.wer)} |",
        f"| CER | {_number(s.cer)} |",
        f"| mean of the per-file WER | {_number(s.wer_mean_per_file)} |",
        f"| files with a WER ... | {shares} |",
    ]
    if s.verdict is not None or s.baseline_wer is not None:  # a summary of the baseline itself has no comparison
        rows += [
            f"| baseline WER | {_number(s.baseline_wer)} |",
            f"| difference from the baseline (positive = better) | {_number(s.delta_wer_vs_baseline)} |",
            f"| interval of the difference | {interval} |",
            f"| verdict against the baseline | {s.verdict or '-'} ({s.n_groups if s.n_groups is not None else '-'} groups) |",
        ]
    return [*rows, ""]


def render_markdown(card: ModelCard, entry_id: str, experiment_id: str, test_evaluations: int, has_review: bool, has_checksums: bool) -> str:
    """The text of `MODEL_CARD.md`: plain English, the verdict first, then the numbers, how the model was made, and how to check the files."""
    status = {"model": "WINNER", "baseline_retained": "THE BASELINE STAYS (no model was added)"}[card.kind]
    test_note = {"confirmed": "confirmed by the Test", "not_confirmed": "NOT confirmed by the Test (flagged, not dropped)", "not_evaluated": "the Test was not evaluated"}[card.test_status]
    lines = [
        f"# Model card: {entry_id} (experiment {experiment_id})",
        "",
        f"**Result: {status}; {test_note}.**",
        "",
        "## In plain words",
        "",
        card.selector_reason,
        "",
    ]
    if card.warnings:
        lines += ["## Flags", ""] + [f"- {w}" for w in card.warnings] + [""]
    lines += ["## Numbers", ""]
    lines += _summary_lines("Val (the choice was made on this set)" if card.kind == "model" else "Val (the baseline)", card.val)
    lines += _summary_lines("Test (looked at once, as the final confirmation)", card.test)
    lines += ["## All finalists on Val", "", "| rank | trial | parameters | Val WER | difference | verdict | Val loss | steps | note |", "|---|---|---|---|---|---|---|---|---|"]
    for i, f in enumerate(card.finalists, 1):
        lines.append(f"| {i} | {f.trial_id} | {_params(f.params)} | {_number(f.val_wer)} | {_number(f.delta_wer)} | {f.verdict} | {_number(f.val_loss)} | {f.steps} | {f.note} |")
    lines += ["", "## How it was made", ""]
    lines += [
        f"- base model: {card.base_model_name} (`{card.base_model_path}`)",
        f"- trial: {card.trial_id if card.trial_id is not None else '-'}, searched parameters: {_params(card.searched_params)}, steps: {card.steps if card.steps is not None else '-'}, Val loss: {_number(card.val_loss)}",
        f"- seed: {card.seed}; evaluation profile (eval_version): {card.eval_version}",
        f"- decoding: {', '.join(f'{k} {v}' for k, v in card.decode_settings.items())}",
        f"- data / split / subsample fingerprints: {card.data_fingerprint or '-'} / {card.split_fingerprint or '-'} / {card.subsample_fingerprint or '-'}",
        f"- configuration of the experiment: `{card.config_ref}` (relative to the experiments folder)",
        f"- code version: {card.code_version}; written at: {card.generated_at}",
        "- libraries: " + (", ".join(f"{k} {v}" for k, v in card.library_versions.items()) or "-"),
        "",
        "## How often the Test set was looked at",
        "",
    ]
    if card.test_fingerprint is None:
        lines.append("This entry was not evaluated on a Test set, so it does not count towards any Test set.")
    else:
        lines.append(
            f"This Test set (fingerprint `{card.test_fingerprint}`) has been looked at {test_evaluations} time(s) by registered entries, this one included. "
            "The count covers registered entries only, so it is a lower bound. Comparisons between experiments are made on Val; the Test numbers are reported but not used to pick among experiments."
        )
    lines += ["", "## Files of this entry", ""]
    if card.kind == "model":
        lines.append("- `model/`: the model folder. To use it, set `model.path` to this folder.")
    else:
        lines.append("- no `model/` folder: the baseline stayed.")
    lines += ["- `model_card.json`: the same facts as data (read by the report builder).", "- `MODEL_CARD.md`: this file."]
    if has_review:
        lines.append("- `review_list.txt`: the files to listen to (worst first, then random), with the word-level differences and the audio paths at the time of the run.")
    if has_checksums:
        lines.append("- `SHA256SUMS`: the checksum of every file of this entry; `ModelRegistry.verify('" + entry_id + "')` checks them (use it after copying the registry to another machine).")
    return "\n".join(lines) + "\n"


def render_review(review: ReviewList, title: str) -> str:
    """The review list as plain text (kept next to the card so that a copied registry still has the listening list)."""
    lines = [title, "Listen to the audio files to judge the text yourself. [replaced a->b], [-missing-], [+extra+]", ""]
    for e in review.entries:
        lines += [f"[{e.reason}] {e.sample_id}  WER {e.wer:.2f}  group {e.group_id}", f"  audio: {e.audio_path}", f"  REF : {e.reference}", f"  HYP : {e.hypothesis}", f"  DIFF: {e.diff}", ""]
    return "\n".join(lines)
