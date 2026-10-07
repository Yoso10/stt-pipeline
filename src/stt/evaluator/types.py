"""Typed data objects of the Evaluator component and its Protocols.

How to run: this module is not a script; the other modules and later components import it. Example
(PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.evaluator.types import FileScore; print(list(FileScore.__dataclass_fields__))"

No torch or transformers import here. Additions to the brief (DECISIONS 41): `EvalResult.review`, `EvalResult.excluded_ids`,
`EvalResult.extra_metrics`, `FileScore.extra`, `DecodeSettings.token_caps`, optional numbers in `ComparisonResult`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from stt.featurepipeline import FeatureDataset
from stt.modelloader import LoadedModel, ModelInfo

Attribute = float | int | bool | str
Verdict = Literal["improved", "no_clear_difference", "worse", "not_comparable", "insufficient_groups"]
DURATION_ATTRIBUTE = "duration"  # the key of `EvalItem.attributes` that holds the audio length in seconds


@dataclass(frozen=True)
class EvalItem:
    """One file to score. `reference` is its `eval_text`; `attributes` are per-file facts passed through unchanged."""

    sample_id: str
    group_id: str
    reference: str
    audio_path: Path
    attributes: Mapping[str, Attribute] = field(default_factory=dict)


@dataclass(frozen=True)
class DecodeSettings:
    """How the model transcribes. `max_new_tokens` is the cap for every file (the model's label limit by default);
    `token_caps` (sample id -> cap) is the shorter per-file cap derived from the audio length, or None."""

    num_beams: int
    max_new_tokens: int
    batch_size: int
    max_tokens_per_audio_sec: float | None = None
    token_caps: Mapping[str, int] | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class RawHypothesis:
    """What the Transcriber returns for one file: the text as decoded, and whether the output hit its token cap."""

    sample_id: str
    text: str
    hit_length_cap: bool = False


class Transcriber(Protocol):
    """Transcribes every example of a dataset. The Hugging Face adapter is the only implementation that needs the model."""

    def transcribe(
        self, model: LoadedModel, dataset: FeatureDataset, settings: DecodeSettings, progress: Callable[[int, int], None] | None = None
    ) -> list[RawHypothesis]:
        """Return one hypothesis per example, in dataset order. `progress(done, total)` is optional (Wave 3 task 7): when given, the implementation calls it after every batch with the number of
        examples transcribed so far; a failing callback must not stop the transcription. The Evaluator passes it ONLY when the request has one."""
        ...


@dataclass(frozen=True)
class MetricScore:
    """Errors and reference units (words or characters) of one file for one metric."""

    errors: int
    units: int


class Metric(Protocol):
    """An error-rate metric. A new metric is a new class in the registry."""

    name: str

    def score(self, reference: str, hypothesis: str) -> MetricScore:
        """Score one file. Both texts are already normalized."""
        ...


@dataclass(frozen=True)
class EvalRequest:
    """Everything one evaluation needs. `normalize` is the eval profile and is applied to the hypotheses."""

    items: Sequence[EvalItem]
    dataset: FeatureDataset
    model: LoadedModel
    split_name: str
    eval_version: int
    normalize: Callable[[str], str]
    settings: DecodeSettings
    on_progress: Callable[[int, int], None] | None = field(default=None, compare=False, repr=False)  # optional (Wave 3 task 7): (examples transcribed, total) after every batch


@dataclass(frozen=True)
class FileScore:
    """The score of one file (kept for every file: the table shows where the model fails)."""

    sample_id: str
    group_id: str
    reference: str
    hypothesis: str
    hypothesis_raw: str
    ref_words: int
    word_errors: int
    wer: float
    cer: float | None
    empty_output: bool
    hit_length_cap: bool
    attributes: Mapping[str, Attribute]
    extra: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ReviewEntry:
    """One file for the owner to listen to. `diff` is the word-level difference: `[a->b]` replaced, `[-a-]` missing, `[+b+]` extra."""

    sample_id: str
    group_id: str
    audio_path: Path
    reference: str
    hypothesis: str
    wer: float
    diff: str
    reason: Literal["worst", "random"]


@dataclass(frozen=True)
class ReviewList:
    """The human review list: the worst files first, then random files."""

    entries: tuple[ReviewEntry, ...]


@dataclass(frozen=True)
class EvalResult:
    """The scored result of one evaluation. `wer` is the corpus WER (all words together), the primary metric.

    `flags` counts `empty_output`, `hit_length_cap` and `excluded_empty_reference`; `excluded_ids` lists the files whose reference was
    empty after normalization (they are not scored). `share_files_wer_below` maps each threshold to the share of files under it.
    """

    split_name: str
    eval_version: int
    model_info: ModelInfo
    wer: float
    wer_mean_per_file: float
    cer: float | None
    n_files: int
    n_words: int
    share_files_wer_below: Mapping[float, float]
    files: tuple[FileScore, ...]
    decode_settings: DecodeSettings
    seconds: float
    flags: Mapping[str, int]
    excluded_ids: tuple[str, ...]
    review: ReviewList
    extra_metrics: Mapping[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """The result as plain text (one short block)."""
        cer = "-" if self.cer is None else f"{self.cer:.4f}"
        shares = ", ".join(f"under {t:.0%}: {s:.1%}" for t, s in self.share_files_wer_below.items())
        flags = ", ".join(f"{k} {v}" for k, v in self.flags.items())
        return (
            f"{self.split_name} (eval_version {self.eval_version}, model {self.model_info.name}): {self.n_files} files, {self.n_words} words, "
            f"WER {self.wer:.4f}, mean per-file WER {self.wer_mean_per_file:.4f}, CER {cer}; files with WER {shares}; "
            f"flags: {flags}; {self.seconds:.1f} s"
        )


@dataclass(frozen=True)
class ComparisonResult:
    """Baseline against candidate. `delta_wer` = baseline minus candidate (positive = the candidate is better).

    With `one_sided` the interval is `[ci_low, ci_high]` of two one-sided bounds at the same `confidence`: `ci_low` decides "improved",
    `ci_high` decides "worse". The numbers are None for `not_comparable`; the interval is None for `insufficient_groups`.
    """

    delta_wer: float | None
    ci_low: float | None
    ci_high: float | None
    confidence: float
    one_sided: bool
    n_groups: int
    min_effect_met: bool
    verdict: Verdict
    explanation: str


@dataclass(frozen=True)
class GroupErrors:
    """Per group: word errors of the baseline and of the candidate, and the reference words (the unit of the bootstrap)."""

    baseline_errors: int
    candidate_errors: int
    words: int


class SignificanceTest(Protocol):
    """Decides whether the candidate is better, from per-group errors. Implementation: paired_group_bootstrap."""

    def interval(self, groups: Sequence[GroupErrors], seed: int) -> tuple[float, float]:
        """Return the (low, high) bounds of `delta_wer` as defined in `ComparisonResult`."""
        ...
