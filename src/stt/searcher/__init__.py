"""stt.searcher: propose hyperparameter sets, hand them to a trial runner, record every outcome, return the ranked candidates (C8 HyperparamSearcher).

How to run: from Python (PowerShell, project root; the working example is in docs/COMPONENTS.md, C8):
    $env:PYTHONPATH = "src"
    python -c "from stt.searcher import default_registry; print(default_registry().names())"
Example output:
    ['grid', 'optuna_tpe']
Importing this package does not import Optuna, torch or transformers (the Optuna adapter is imported only for `optuna_tpe`).
"""

from stt.searcher.config import (
    SEARCHABLE,
    FinalistsConfig,
    ParamConfig,
    PruningConfig,
    ScreeningConfig,
    SearchConfig,
    apply_params,
    build_spec,
)
from stt.searcher.errors import SearchConfigError, SearchError, SearchInterrupted
from stt.searcher.grid import GridStrategy, grid_points
from stt.searcher.pruning import MedianRule
from stt.searcher.registry import StrategyRegistry, default_registry
from stt.searcher.searcher import HyperparamSearcher
from stt.searcher.study import StudyStore
from stt.searcher.types import (
    ParamRange,
    PruningSpec,
    SearchReport,
    SearchResult,
    SearchSpec,
    SearchStrategy,
    Suggestion,
    Trial,
    TrialOutcome,
    TrialRecord,
    TrialRunner,
)

__all__ = [
    "SEARCHABLE",
    "FinalistsConfig",
    "GridStrategy",
    "HyperparamSearcher",
    "MedianRule",
    "ParamConfig",
    "ParamRange",
    "PruningConfig",
    "PruningSpec",
    "ScreeningConfig",
    "SearchConfig",
    "SearchConfigError",
    "SearchError",
    "SearchInterrupted",
    "SearchReport",
    "SearchResult",
    "SearchSpec",
    "SearchStrategy",
    "StrategyRegistry",
    "StudyStore",
    "Suggestion",
    "Trial",
    "TrialOutcome",
    "TrialRecord",
    "TrialRunner",
    "apply_params",
    "build_spec",
    "default_registry",
    "grid_points",
]
