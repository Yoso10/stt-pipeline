"""Typed data objects of the ModelRegistry component.

How to run: this module is not a script; the other modules and later components import it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.registry.types import VerifyReport; print(VerifyReport(3, (), (), True))"
Example output:
    VerifyReport(files_checked=3, missing=(), mismatching=(), ok=True)

Everything in a card is plain data (numbers, text, lists, mappings) so that `model_card.json` can be read by anyone and by the ReportBuilder later. No torch and
no Hugging Face import. Additions to the brief (DECISIONS 53): `EvalSummary`, `FinalistRow`, the split of the fingerprints into fields, `ModelCard.kind`/`verdict`/`test_status`,
`ModelCard.test_evaluations`, `VerifyReport` as the content of `RegistryVerifyError`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from stt.selector import SelectionResult

Kind = Literal["model", "baseline_retained"]


@dataclass(frozen=True)
class EvalSummary:
    """The summary of one evaluated split in a card: numbers only, no per-file data. `delta_wer_vs_baseline`, the interval, `verdict` and `n_groups`
    describe the comparison with the baseline (None for the baseline itself or when there was none)."""

    split: str
    model: str
    n_files: int
    n_words: int
    wer: float
    cer: float | None
    wer_mean_per_file: float
    share_files_wer_below: Mapping[str, float]
    baseline_wer: float | None = None
    delta_wer_vs_baseline: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    confidence: float | None = None
    verdict: str | None = None
    n_groups: int | None = None


@dataclass(frozen=True)
class FinalistRow:
    """One finalist as the Selector saw it (all finalists are kept in the card of the entry, so the history is complete)."""

    trial_id: int
    params: Mapping[str, float | int | str]
    val_wer: float
    delta_wer: float | None
    verdict: str
    val_loss: float
    steps: int
    eligible: bool
    note: str


@dataclass(frozen=True)
class ModelCard:
    """The metadata of an entry (written as `model_card.json` and rendered as `MODEL_CARD.md`). The caller fills it (see `card_from_selection`).

    `test_fingerprint` identifies the Test set (None when the Test was not evaluated); `test_evaluations` is how many registered entries, this one included, were
    evaluated on that Test set at registration time (the registry sets it; the caller leaves it 0). `generated_at` is passed in (the registry never reads the clock).
    """

    kind: Kind
    verdict: str
    base_model_name: str
    base_model_path: str
    data_fingerprint: str | None
    split_fingerprint: str | None
    subsample_fingerprint: str | None
    test_fingerprint: str | None
    config_ref: str
    eval_version: int
    val: EvalSummary | None
    test: EvalSummary | None
    test_status: str
    decode_settings: Mapping[str, float | int | str | None]
    searched_params: Mapping[str, float | int | str]
    finalists: Sequence[FinalistRow]
    library_versions: Mapping[str, str]
    seed: int
    code_version: str
    generated_at: str
    selector_reason: str
    warnings: Sequence[str] = field(default_factory=tuple)
    trial_id: int | None = None
    steps: int | None = None
    val_loss: float | None = None
    test_evaluations: int = 0


@dataclass(frozen=True)
class RegisterRequest:
    """What to register. `finalist_dirs` are the output folders of the finalists (the model folders and the checkpoints are inside them)."""

    selection: SelectionResult
    card: ModelCard
    experiment_id: str
    finalist_dirs: Sequence[Path]
    review_text: str | None = None


@dataclass(frozen=True)
class RegistryEntry:
    """A stored entry. `test_evaluations` counts the registered entries, this one included, that were evaluated on the same Test set when it was registered."""

    entry_id: str
    experiment_id: str
    kind: Kind
    model_dir: Path | None
    card_path: Path
    checksums_path: Path | None
    test_evaluations: int


@dataclass(frozen=True)
class VerifyReport:
    """The result of `verify`: how many files were checked, which are missing, which differ, and whether everything is fine."""

    files_checked: int
    missing: tuple[str, ...]
    mismatching: tuple[str, ...]
    ok: bool
