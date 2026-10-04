"""Reusable validation helpers for the config sections of later components.

How to run: this module is not a script. A component uses it in its own section schema, for example:
    from stt.config import SectionModel
    from stt.config.validators import ExistingPath
    class DataSection(SectionModel):
        path: ExistingPath            # fails with a clear message when the folder does not exist
Example message:
    Path does not exist: data/missing. Check the spelling, or create or copy the folder there.
These helpers belong to the Config brief's edge cases ("path that must exist does not exist",
"split ratios not summing to 1"); the keys themselves belong to the components.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator


def _must_exist(path: Path) -> Path:
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}. Check the spelling, or create or copy the folder there.")
    return path


ExistingPath = Annotated[Path, AfterValidator(_must_exist)]
"""A path that must exist when the configuration is loaded."""


def check_ratios_sum_to_one(ratios: Mapping[str, float], tolerance: float = 1e-6) -> None:
    """Raise ValueError (for use inside a pydantic validator) unless the ratios sum to 1.

    Inputs: name -> ratio. The message lists the values and their sum.
    """
    total = sum(ratios.values())
    if not math.isclose(total, 1.0, abs_tol=tolerance):
        listed = ", ".join(f"{k}={v}" for k, v in ratios.items())
        raise ValueError(f"Split ratios must sum to 1, but {listed} sums to {total}. Adjust the values so that they add up to 1.")
