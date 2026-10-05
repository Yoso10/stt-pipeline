"""The `search` configuration section, owned by the HyperparamSearcher component.

How to run: this module is not a script. The composition root registers the section:
    from stt.config import SectionRegistry, load_config
    from stt.searcher.config import SearchConfig
    config = load_config(path, registry=SectionRegistry({"search": SearchConfig}))
    search = config.section("search", SearchConfig)
Example YAML with every key and its DEFAULT: configs/search_default.yaml. The seed comes from `runtime.seed`. Naming `parameters` in a YAML REPLACES the
default space (it is the space that is searched); every other training value stays fixed at `train.hyperparams`. `max_steps` and `seed` cannot be searched.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from stt.config import SectionModel
from stt.searcher.errors import SearchConfigError
from stt.searcher.types import ParamRange, ParamValue, PruningSpec, SearchSpec
from stt.trainercore import TrainHyperparams

SEARCHABLE = ("learning_rate", "grad_accum_steps", "batch_size", "warmup_steps", "weight_decay", "max_grad_norm")
INTEGER_PARAMS = ("grad_accum_steps", "batch_size", "warmup_steps")


class ParamConfig(SectionModel):
    """One searched parameter."""

    kind: Literal["log_float", "float", "int", "categorical"] = Field(description="log_float and float: a range (low, high); int: whole numbers (low, high); categorical: a list of choices.")
    low: float | None = Field(default=None, description="Lower bound of a range.")
    high: float | None = Field(default=None, description="Upper bound of a range.")
    choices: list[float | int | str] = Field(default_factory=list, description="The values of a categorical parameter.")
    points: int = Field(default=3, ge=2, description="NEW, DEFAULT 3. How many values the grid strategy takes from a float or log_float range.")

    @model_validator(mode="after")
    def _consistent(self) -> ParamConfig:
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError("a categorical parameter needs a non-empty list of choices")
        else:
            if self.low is None or self.high is None:
                raise ValueError(f"a {self.kind} parameter needs low and high")
            if self.low >= self.high:
                raise ValueError(f"low ({self.low}) must be below high ({self.high})")
            if self.kind == "log_float" and self.low <= 0:
                raise ValueError("a log_float range needs low above 0")
        return self

    def to_range(self) -> ParamRange:
        """The typed range."""
        return ParamRange(self.kind, self.low, self.high, tuple(self.choices), self.points)


class PruningConfig(SectionModel):
    """`search.pruning`."""

    enabled: bool = Field(default=True, description="DEFAULT true. A trial that is clearly worse after a few Val evaluations is stopped.")
    min_evals: int = Field(default=3, ge=1, description="DEFAULT 3. No trial is pruned before it has had this many Val evaluations.")


class FinalistsConfig(SectionModel):
    """`search.finalists`."""

    k: int = Field(default=3, ge=1, description="DEFAULT 3. The best k parameter sets are returned for the retraining on the full data.")


class ScreeningConfig(SectionModel):
    """`search.screening`: the Subsampler preset the search runs on (the Orchestrator reads it; to revise after the step times, O11)."""

    preset: str = Field(default="tiny", description="DEFAULT tiny. A Subsampler preset name.")


def _default_parameters() -> dict[str, ParamConfig]:
    return {
        "learning_rate": ParamConfig(kind="log_float", low=5e-6, high=5e-5),
        "grad_accum_steps": ParamConfig(kind="categorical", choices=[2, 4]),
    }


class SearchConfig(SectionModel):
    """The `search` section."""

    strategy: str = Field(default="optuna_tpe", description="DEFAULT optuna_tpe (the brief's DEFAULT; the owner decides after the comparison with grid, O10). Registered: grid, optuna_tpe.")
    parameters: dict[str, ParamConfig] = Field(default_factory=_default_parameters, description="DEFAULT learning_rate (log range 5e-6 to 5e-5) and grad_accum_steps (2 or 4). Naming this key replaces the default space.")
    n_trials: int = Field(default=6, ge=1, description="DEFAULT 6 (a small budget, thinned from the brief's 10 because of the step time, O28). Trials in all, earlier ones of a resumed study included.")
    time_budget_minutes: float | None = Field(default=None, gt=0, description="DEFAULT none. Stop starting trials after this many minutes (a running trial is told to stop).")
    pruning: PruningConfig = PruningConfig()
    parallel_trials: int = Field(default=1, ge=1, description="DEFAULT 1 (one after another). More than 1 runs worker processes, each with total threads divided by this number.")
    max_failed_ratio: float = Field(default=0.5, ge=0.0, le=1.0, description="DEFAULT 0.5. The search stops when more than this share of the finished trials failed.")
    min_trials_before_abort: int = Field(default=4, ge=1, description="DEFAULT 4. The failure ratio is checked only after this many finished trials.")
    finalists: FinalistsConfig = FinalistsConfig()
    finalist_repeats: int = Field(default=1, ge=1, description="DEFAULT 1. Seeds per finalist (a later stage; the Orchestrator reads it).")
    screening: ScreeningConfig = ScreeningConfig()
    study_dir: Path | None = Field(default=None, description="Set by the RunManager: the `search/` folder of the experiment. Empty = nothing is stored (no resume).")

    @field_validator("parameters")
    @classmethod
    def _known_names(cls, value: dict[str, ParamConfig]) -> dict[str, ParamConfig]:
        if not value:
            raise ValueError("search.parameters must name at least one parameter")
        for name in value:
            if name in ("max_steps", "seed"):
                raise ValueError(f"'{name}' is not searched (a generous max_steps plus early stopping decides the length; the seed is fixed). Searchable: {', '.join(SEARCHABLE)}")
            if name not in SEARCHABLE:
                raise ValueError(f"unknown parameter '{name}'. Searchable: {', '.join(SEARCHABLE)}")
        return value


def build_spec(config: SearchConfig, runtime_seed: int, total_threads: int) -> SearchSpec:
    """The typed spec of the config. `total_threads` is the thread budget of the whole search (for example `resolve_threads(train_config)`)."""
    return SearchSpec(
        parameters={name: p.to_range() for name, p in config.parameters.items()},
        strategy=config.strategy,
        n_trials=config.n_trials,
        time_budget_minutes=config.time_budget_minutes,
        pruning=PruningSpec(config.pruning.enabled, config.pruning.min_evals),
        parallel_trials=config.parallel_trials,
        max_failed_ratio=config.max_failed_ratio,
        min_trials_before_abort=config.min_trials_before_abort,
        finalists_k=config.finalists.k,
        study_dir=config.study_dir,
        seed=runtime_seed,
        total_threads=max(1, total_threads),
    )


def apply_params(base: TrainHyperparams, params: dict[str, ParamValue] | object) -> TrainHyperparams:
    """The hyperparameters of a trial: `base` (from `train.hyperparams`) with the searched values replaced (whole numbers for the integer parameters).

    Errors: SearchConfigError for a name that cannot be searched.
    """
    values = {}
    for name, value in dict(params).items():  # type: ignore[call-overload]
        if name not in SEARCHABLE:
            raise SearchConfigError(f"Cannot apply the parameter '{name}': searchable parameters are {', '.join(SEARCHABLE)}.")
        values[name] = int(value) if name in INTEGER_PARAMS else float(value)
    return replace(base, **values)
