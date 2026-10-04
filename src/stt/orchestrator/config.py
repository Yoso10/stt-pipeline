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
        default_factory=lambda: ["prepare", "train", "evaluate"],
        min_length=1,
        description="DEFAULT prepare, train, evaluate (the minimal chain; the brief's later default adds search, select_and_register, export, report).",
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
