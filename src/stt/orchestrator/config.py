"""The `pipeline` and `smoke` configuration sections, owned by the Orchestrator component.

How to run: this module is not a script. The composition root (`wiring.section_registry()`) registers the sections. Example YAML with every
key and its DEFAULT: configs/pipeline_default.yaml (and the complete smoke setup in configs/smoke.yaml). `run.prevent_sleep` is a key of the
RunManager's `run` section that the Orchestrator reads.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from stt.config import SectionModel


class PipelineConfig(SectionModel):
    """The `pipeline` section."""

    stages: list[str] = Field(
        default_factory=lambda: ["prepare", "evaluate", "search", "finalists", "select_and_register"],
        min_length=1,
        description=(
            "DEFAULT prepare, evaluate, search, finalists, select_and_register: the stages of `python -m stt run` (evaluate = the baseline on Val; later: export, report). "
            "Registered stages: prepare, train, evaluate, search, finalists, select_and_register. The other commands (prepare, train, evaluate, search) run their own fixed stages."
        ),
    )
    skip_completed: bool = Field(default=True, description="DEFAULT true. A stage with a completion marker and an unchanged config slice is skipped.")
    eval_split: str = Field(default="val", description="DEFAULT val (NEW key). The split the evaluate stage scores. Model choice and checks use Val.")
    test_split: str = Field(default="test", description="DEFAULT test (NEW key). The Test split: the evaluate stage REFUSES it; it is scored once, for the winner, through the Selector (Wave 2).")


class SmokeConfig(SectionModel):
    """The `smoke` section: the synthetic data of the smoke run."""

    n_samples: int = Field(default=12, ge=3, description="DEFAULT 12. Number of synthetic clips (each is its own group).")
    duration_sec: float = Field(default=3.0, gt=0, le=30, description="DEFAULT 3. Length of each clip in seconds.")
    steps: int = Field(default=3, ge=1, description="DEFAULT 3. Training steps of the smoke run (sets train.hyperparams.max_steps).")
    model_dir: Path = Field(default=Path("tests/assets/tiny_whisper"), description="DEFAULT tests/assets/tiny_whisper. The tiny random-weight test model (tools/make_test_model.py creates it).")
    work_dir: Path = Field(default=Path("data/smoke"), description="DEFAULT data/smoke (NEW key). Where the synthetic audio and manifest are written (git-ignored data folder).")


class LearningCurveConfig(SectionModel):
    """The `learning_curve` section (keys and DEFAULTS of the C13 brief)."""

    presets: list[str] = Field(
        default_factory=lambda: ["smoke", "tiny", "half", "full"],
        min_length=1,
        description="DEFAULT smoke, tiny, half, full. Subsampler preset names (built in or from `subsample.presets`): the Train sizes of the curve. Every point uses scope train_only and the same fixed Val set.",
    )
    params_from: str | None = Field(
        default=None,
        description="DEFAULT none = the fixed `train.hyperparams` of the config. An experiment id (for example exp_0003): its selected winner's searched values (or its best finalist's) replace the searched keys. The command line option `--params-from` sets it.",
    )


class BenchmarkConfig(SectionModel):
    """The `benchmark` section (key and DEFAULT of the C13 brief)."""

    steps: int = Field(default=3, ge=1, description="DEFAULT 3. Training steps to measure. The first steps are slower than the later ones: raise it for a steadier number. The Val probe is steps x eval.batch_size clips.")
