"""Helpers shared by the pipeline stages: config slices, the prepared data of a run, the evaluation items, and the scoring of one model on one split.

How to run: this module is not a script; the stage modules import it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.orchestrator.stagehelpers import pipeline_value; print(pipeline_value.__name__)"
Example output:
    pipeline_value
`evaluate_model` is the ONE place where a model is scored on a split of the prepared data. It REFUSES the Test split (`pipeline.test_split`) unless the caller says
`allow_test=True`; only the selection stage does that, through the one-time Test look (DECISIONS 57).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stt.config import AppConfig
from stt.evaluator import EvalItem, EvalRequest, EvalResult
from stt.orchestrator.errors import OrchestratorError
from stt.orchestrator.types import PreparedData, StageContext

logger = logging.getLogger("stt.orchestrator")
BASELINE_RESULT_FILE = "eval_result_baseline.json"  # reports/: the codec text of the baseline Val result, written by the evaluate stage, read by select_and_register


def section(config: AppConfig, name: str) -> dict[str, Any]:
    """A config section as plain data (an empty dict when it is not loaded)."""
    value = config.sections.get(name)
    return {} if value is None else dict(value.model_dump(mode="json"))


def pipeline_value(config: AppConfig, key: str, default: str) -> str:
    """A key of the `pipeline` section as text."""
    return str(getattr(config.sections.get("pipeline"), key, default))


def prepared_data(ctx: StageContext) -> PreparedData:
    """The prepared data of this run (made once and shared by the stages of the run)."""
    if "prepared" not in ctx.shared:
        ctx.shared["prepared"] = ctx.services.data.prepare(ctx.experiment.paths.reports)
    prepared: PreparedData = ctx.shared["prepared"]
    return prepared


def make_items(prepared: PreparedData, split: str, removed: frozenset[str], model_tokens: Callable[[str], int]) -> list[EvalItem]:
    """The EvalItems of a split with the per-file attributes (duration, reference words, label tokens, numeric extra fields, text flags)."""
    items = []
    for n in prepared.kept[split]:
        s = n.sample
        if s.id in removed:
            continue
        attributes: dict[str, float | int | bool | str] = {
            "duration": s.duration_sec,
            "ref_words": len(n.eval_text.split()),
            "label_tokens": model_tokens(n.train_text),
        }
        attributes.update({f"extra_{k}": v for k, v in s.extra.items() if isinstance(v, int | float | bool)})
        attributes.update({f"flag_{f}": True for f in n.flags})
        items.append(EvalItem(s.id, s.group_id, n.eval_text, s.audio_path, attributes))
    return items


def evaluate_model(ctx: StageContext, split: str, path: Path | None, label: str, allow_test: bool = False) -> EvalResult:
    """Score the model in the folder `path` (None = the base model of `model.path`) on `split` of the prepared data.

    Errors: OrchestratorError (exit code 2) for the Test split without `allow_test`, or for an unknown split; evaluation errors (exit code 4).
    """
    test_split = pipeline_value(ctx.config, "test_split", "test")
    if split == test_split and not allow_test:
        raise OrchestratorError(
            f"Refused to score the Test split ('{split}') here: the Test is evaluated ONCE, only for the winner and the baseline, by the select_and_register stage. "
            f"Use the Val split ('{pipeline_value(ctx.config, 'eval_split', 'val')}').",
            exit_code=2,
        )
    prepared = prepared_data(ctx)
    if split not in prepared.kept:
        raise OrchestratorError(f"The split '{split}' does not exist. Known splits: {', '.join(sorted(prepared.kept))}. Set pipeline.eval_split or --split.", exit_code=2)
    model = ctx.services.models.load(path)
    features = ctx.services.models.features(model).build(prepared.kept[split], split)
    items = make_items(prepared, split, frozenset(features.report.removed_label_ids), model.tokenizer.count_tokens)
    request = EvalRequest(items, features.dataset, model, split, prepared.eval_version, prepared.normalize, ctx.services.evaluation.decode_settings(model, items))
    logger.info("Evaluating the %s model (%s) on %d files of '%s'.", label, model.info.name, len(items), split)
    return ctx.services.evaluation.evaluate(request)
