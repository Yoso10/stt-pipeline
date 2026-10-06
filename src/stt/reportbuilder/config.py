"""The `report` configuration section, owned by the ReportBuilder component.

How to run: this module is not a script. The composition root registers the section (the Orchestrator task that wires the `report` command does it):
    from stt.config import SectionRegistry, load_config
    from stt.reportbuilder.config import ReportConfig
    config = load_config(path, registry=SectionRegistry({"report": ReportConfig}))
    report = config.section("report", ReportConfig)
Example YAML with every key and its DEFAULT: configs/report_default.yaml. The keys and DEFAULTS are those of the C14 brief; the NEW keys are marked in their description.
"""

from __future__ import annotations

from pydantic import Field

from stt.config import SectionModel


class ProgressConfig(SectionModel):
    """`report.progress`: the live progress page."""

    refresh_seconds: int = Field(default=10, ge=1, description="DEFAULT 10. The page asks the browser to reload itself every this many seconds.")
    every_n_records: int = Field(default=1, ge=1, description="DEFAULT 1. The page is written again after every this many metrics records.")


class HintsConfig(SectionModel):
    """`report.hints`: which diagnosis rules run and their thresholds. All thresholds are DEFAULT values to calibrate on real runs; a hint is advice, never a fact."""

    enabled_rules: list[str] | None = Field(default=None, description="DEFAULT none = every registered rule. Registered: overfitting, no_learning, unstable_training, gain_concentrated, short_clip_weakness, length_cap_or_empty, edge_of_range, failed_or_pruned.")
    overfitting_last_k_evals: int = Field(default=3, ge=2, description="DEFAULT 3. overfitting: the Val loss rose at each of the last K evaluations while the training loss fell over the same span.")
    no_learning_min_rel_improvement: float = Field(default=0.01, ge=0.0, description="DEFAULT 0.01. no_learning: the best Val loss is less than this share better than the first one.")
    unstable_spike_factor: float = Field(default=2.0, gt=1.0, description="DEFAULT 2.0. unstable_training: a training loss above this many times the median of the 5 records before it.")
    gain_top_group_fraction: float = Field(default=0.1, gt=0.0, le=1.0, description="DEFAULT 0.1. gain_concentrated: the best this share of the groups (lessons) ...")
    gain_share: float = Field(default=0.5, gt=0.0, le=1.0, description="DEFAULT 0.5. ... account for at least this share of the total word-error reduction.")
    short_clip_quantile: float = Field(default=0.25, gt=0.0, lt=0.5, description="DEFAULT 0.25. short_clip_weakness: the shortest this share of the files ...")
    short_clip_wer_ratio: float = Field(default=1.5, gt=1.0, description="DEFAULT 1.5. ... has a WER at least this many times that of the other files.")
    length_cap_share: float = Field(default=0.05, ge=0.0, le=1.0, description="DEFAULT 0.05. length_cap_or_empty: at least this share of the files hit the length cap or are empty.")
    failed_share: float = Field(default=0.3, ge=0.0, le=1.0, description="DEFAULT 0.3. failed_or_pruned: at least this share of the started trials failed or were pruned.")


class ReportConfig(SectionModel):
    """The `report` section."""

    title: str | None = Field(default=None, description="DEFAULT none = the experiment name. The title of the page.")
    attributes: list[str] = Field(
        default_factory=lambda: ["duration", "label_tokens", "ref_words", "extra_meta_quality_score", "flag_had_digits", "flag_had_latin", "flag_had_brackets"],
        description=(
            "DEFAULT duration, label_tokens, ref_words, extra_meta_quality_score (only if present), flag_had_digits, flag_had_latin, flag_had_brackets. The per-file attributes whose rank correlation with the "
            "per-file WER is shown. They are the keys the Orchestrator gives the Evaluator (`duration`, `ref_words`, `label_tokens`, `extra_<column>`, `flag_<text flag>`); the brief's names (duration_sec, "
            "meta_quality_score, had_digits ...) are the same things before that naming. A `flag_*` attribute that a file does not carry counts as 0. Open to change after the owner sees the first output."
        ),
    )
    min_points_for_correlation: int = Field(default=30, ge=2, description="DEFAULT 30. Fewer points: the per-file correlation row says `too few points` and shows no number; the search correlation is shown with a caveat.")
    max_points_per_chart: int = Field(default=2000, ge=10, description="DEFAULT 2000. More points in a scatter or a curve are thinned deterministically (every k-th) and the page says so.")
    groups_shown: int = Field(default=15, ge=1, description="NEW, DEFAULT 15. How many groups (lessons) the group table lists: the worst ones by the winner's WER.")
    histogram_bins: int = Field(default=20, ge=2, description="NEW, DEFAULT 20. Bins of the per-file WER histogram.")
    loss_noise_range: float = Field(
        default=0.01, ge=0.0,
        description="NEW, DEFAULT 0.01 (owner decision 2026-10-07). A Val-loss chart whose plotted values vary by less than this (in loss units) gets the note `Differences this small are noise`; the axis stays automatic.",
    )
    absolute_audio_paths: bool = Field(
        default=False,
        description="NEW, DEFAULT false (owner decision 2026-10-07). false: the listening list shows each audio path relative to the data folder (when the file is inside it); true: the full path as stored.",
    )
    hints: HintsConfig = HintsConfig()
    progress: ProgressConfig = ProgressConfig()
