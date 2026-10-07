"""Evaluator: transcribe an evaluation set, score the transcripts against the references (C7).

How to run: from Python (PowerShell, project root; the full working example is in docs/COMPONENTS.md, C7):
    $env:PYTHONPATH = "src"
    python -c "from stt.evaluator import EvalConfig, default_metric_registry; print(default_metric_registry().names())"
`Evaluator(config, transcriber, seed).evaluate(request) -> EvalResult`. The same Evaluator, with the same settings and the same eval profile,
measures the baseline and every trained model. It writes no file and prints nothing. Importing this module does not import torch.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence

from stt.evaluator.config import EvalConfig
from stt.evaluator.errors import EvaluatorConfigError, EvaluatorError
from stt.evaluator.registry import MetricRegistry, default_metric_registry
from stt.evaluator.review import build_review_list
from stt.evaluator.types import (
    DURATION_ATTRIBUTE,
    DecodeSettings,
    EvalItem,
    EvalRequest,
    EvalResult,
    FileScore,
    Metric,
    Transcriber,
)
from stt.modelloader import LoadedModel

logger = logging.getLogger("stt.evaluator")


def make_decode_settings(config: EvalConfig, model: LoadedModel, items: Sequence[EvalItem]) -> DecodeSettings:
    """The decoding settings of a run: beams and batch size from the config, the token cap from the config or the model's label limit,
    and, when `eval.generation.max_tokens_per_audio_sec` is set, a per-file cap from the `duration` attribute of each item.

    Errors: EvaluatorError when the per-second cap is set and an item has no numeric `duration` attribute.
    """
    generation = config.generation
    maximum = generation.max_new_tokens or model.tokenizer.max_label_tokens
    caps: dict[str, int] | None = None
    if generation.max_tokens_per_audio_sec is not None:
        caps = {}
        for item in items:
            seconds = item.attributes.get(DURATION_ATTRIBUTE)
            if isinstance(seconds, bool) or not isinstance(seconds, int | float) or seconds <= 0:
                raise EvaluatorError(
                    f"eval.generation.max_tokens_per_audio_sec is set but item '{item.sample_id}' has no positive numeric attribute "
                    f"'{DURATION_ATTRIBUTE}'. Add it to the EvalItem attributes, or remove the setting."
                )
            caps[item.sample_id] = max(1, min(maximum, math.ceil(generation.max_tokens_per_audio_sec * seconds)))
    return DecodeSettings(generation.num_beams, maximum, config.batch_size, generation.max_tokens_per_audio_sec, caps)


class Evaluator:
    """Scores one model on one evaluation set. The transcriber is injected (a fake one in the unit tests)."""

    def __init__(
        self,
        config: EvalConfig,
        transcriber: Transcriber,
        seed: int,
        metric_registry: MetricRegistry | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        registry = metric_registry or default_metric_registry()
        self._config = config
        self._transcriber = transcriber
        self._seed = seed
        self._clock = clock
        self._metrics: dict[str, Metric] = {name: registry.create(name) for name in config.metrics}  # unknown names fail here
        if "wer" not in self._metrics:
            raise EvaluatorConfigError("eval.metrics must contain 'wer' (the primary metric). Add it to the list.")

    def evaluate(self, request: EvalRequest) -> EvalResult:
        """Transcribe, normalize the hypotheses with the eval profile, score every file and return the result.

        Errors: EvaluatorError (exit code 4) for an empty item list, hypotheses that do not match the items, or all hypotheses empty
        (the likely cause is a wrong language setting or wrong features); EvaluatorConfigError (exit code 2) for an unknown metric.
        """
        items = list(request.items)
        if not items:
            raise EvaluatorError("There is nothing to evaluate: the item list is empty. Check the split and the SampleFilter report.")
        started = self._clock()
        if request.on_progress is None:
            raw = self._transcriber.transcribe(request.model, request.dataset, request.settings)
        else:  # the callback is passed only when there is one, so a transcriber without the optional parameter keeps working
            raw = self._transcriber.transcribe(request.model, request.dataset, request.settings, progress=request.on_progress)
        elapsed = self._clock() - started
        if len(raw) != len(items) or any(h.sample_id != i.sample_id for h, i in zip(raw, items, strict=False)):
            bad = next((i.sample_id for h, i in zip(raw, items, strict=False) if h.sample_id != i.sample_id), None)
            raise EvaluatorError(
                f"The transcriber returned {len(raw)} hypotheses for {len(items)} items"
                + (f", and the first mismatch is item '{bad}'." if bad else ".")
                + " The dataset and the EvalItem list must hold the same samples in the same order."
            )
        hypotheses = [request.normalize(h.text) for h in raw]
        if not any(h.strip() for h in hypotheses):
            raise EvaluatorError(
                "Every hypothesis is empty after normalization. The likely cause is a wrong model.language or wrong features: "
                "check model.language and the FeaturePipeline report, and read the raw outputs."
            )
        scored: list[FileScore] = []
        excluded: list[str] = []
        sum_errors = dict.fromkeys(self._metrics, 0)
        sum_units = dict.fromkeys(self._metrics, 0)
        for item, hyp_raw, hyp in zip(items, raw, hypotheses, strict=True):
            if not item.reference.strip():
                excluded.append(item.sample_id)
                continue
            scores = {name: metric.score(item.reference, hyp) for name, metric in self._metrics.items()}
            for name, score in scores.items():
                sum_errors[name] += score.errors
                sum_units[name] += score.units
            wer = scores["wer"]
            scored.append(
                FileScore(
                    sample_id=item.sample_id,
                    group_id=item.group_id,
                    reference=item.reference,
                    hypothesis=hyp,
                    hypothesis_raw=hyp_raw.text,
                    ref_words=wer.units,
                    word_errors=wer.errors,
                    wer=wer.errors / wer.units,
                    cer=(scores["cer"].errors / max(1, scores["cer"].units)) if "cer" in scores else None,
                    empty_output=not hyp.strip(),
                    hit_length_cap=hyp_raw.hit_length_cap,
                    attributes=dict(item.attributes),
                    extra={n: s.errors / max(1, s.units) for n, s in scores.items() if n not in ("wer", "cer")},
                )
            )
        if not scored:
            raise EvaluatorError("Every reference is empty after normalization, so nothing can be scored. Check the eval profile and the data.")
        words = sum(f.ref_words for f in scored)
        totals = {name: sum_errors[name] / max(1, sum_units[name]) for name in self._metrics}
        thresholds = {t: sum(f.wer < t for f in scored) / len(scored) for t in self._config.thresholds.wer_below}
        flags = {
            "empty_output": sum(f.empty_output for f in scored),
            "hit_length_cap": sum(f.hit_length_cap for f in scored),
            "excluded_empty_reference": len(excluded),
        }
        paths = {i.sample_id: i.audio_path for i in items}
        result = EvalResult(
            split_name=request.split_name,
            eval_version=request.eval_version,
            model_info=request.model.info,
            wer=totals["wer"],
            wer_mean_per_file=sum(f.wer for f in scored) / len(scored),
            cer=totals.get("cer"),
            n_files=len(scored),
            n_words=words,
            share_files_wer_below=thresholds,
            files=tuple(scored),
            decode_settings=_without_caps(request.settings),
            seconds=elapsed,
            flags=flags,
            excluded_ids=tuple(excluded),
            review=build_review_list(scored, paths, self._config.review, self._seed),
            extra_metrics={n: v for n, v in totals.items() if n not in ("wer", "cer")},
        )
        if flags["hit_length_cap"]:
            logger.warning("%d of %d outputs hit the length cap (a looping model?); they are flagged in the result.", flags["hit_length_cap"], len(scored))
        logger.info(result.summary())
        return result

    def check_sanity(self, baseline: EvalResult) -> None:
        """Fail when `eval.sanity.max_baseline_wer` is set and the baseline WER is above it (call it on the baseline only)."""
        limit = self._config.sanity.max_baseline_wer
        if limit is not None and baseline.wer > limit:
            raise EvaluatorError(
                f"The baseline WER {baseline.wer:.4f} is above eval.sanity.max_baseline_wer {limit:.4f}. "
                "Check model.language, the audio and the reference texts, or raise the limit."
            )


def _without_caps(settings: DecodeSettings) -> DecodeSettings:
    return DecodeSettings(settings.num_beams, settings.max_new_tokens, settings.batch_size, settings.max_tokens_per_audio_sec, None)


__all__ = ["Evaluator", "make_decode_settings"]
