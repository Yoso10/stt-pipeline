"""stt.selector: decide which model is the winner, or that the baseline stays (C9 Selector).

How to run: from Python (PowerShell, project root; the working example is in docs/COMPONENTS.md, C9):
    $env:PYTHONPATH = "src"
    python -c "from stt.selector import SelectConfig; print(SelectConfig().min_verdict)"
Example output:
    improved
Importing this package imports no torch, no transformers and no file-access code; the Selector reads and writes no file.
"""

from stt.selector.config import SelectConfig
from stt.selector.errors import SelectorError, TestAlreadyEvaluatedError
from stt.selector.guard import TestOnceGuard
from stt.selector.selector import Judgement, Selector
from stt.selector.types import (
    Candidate,
    ComparisonRow,
    RankedCandidate,
    SelectionRequest,
    SelectionResult,
    ValComparator,
)

__all__ = [
    "Candidate",
    "ComparisonRow",
    "Judgement",
    "RankedCandidate",
    "SelectConfig",
    "SelectionRequest",
    "SelectionResult",
    "Selector",
    "SelectorError",
    "TestAlreadyEvaluatedError",
    "TestOnceGuard",
    "ValComparator",
]
