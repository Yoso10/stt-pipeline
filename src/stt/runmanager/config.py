"""The `run` configuration section, owned by the RunManager component.

How to run: this module is not a script. The composition root registers the section:
    from stt.config import SectionRegistry, load_config
    from stt.runmanager.config import RunConfig
    config = load_config(path, registry=SectionRegistry({"run": RunConfig}))
    run = config.section("run", RunConfig)
Example YAML with every key and its DEFAULT: configs/run_default.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from stt.config import SectionModel

Level = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class LogConfig(SectionModel):
    """`run.log`: one file per experiment with every detail, and only the important lines on the console."""

    console_level: Level = Field(default="INFO", description="DEFAULT INFO. Lowest level shown on the console.")
    file_level: Level = Field(default="DEBUG", description="DEFAULT DEBUG. Lowest level written to logs/run.log.")


class RunConfig(SectionModel):
    """The `run` section."""

    experiments_dir: Path = Field(default=Path("experiments"), description="DEFAULT experiments. The folder of the numbered experiments (keep the path short on Windows).")
    parent: str | None = Field(default=None, description="DEFAULT none. The experiment this one extends, for example exp_0003 (its config is the base of the diff).")
    id_digits: int = Field(default=4, ge=1, le=9, description="DEFAULT 4. Digits of the experiment number: exp_0001.")
    config_readonly: bool = Field(default=True, description="DEFAULT true. config_full.yaml is made read-only when the experiment is created.")
    lock_heartbeat_seconds: float = Field(default=30.0, gt=0, description="DEFAULT 30. The running process touches run.lock this often.")
    lock_stale_minutes: float = Field(default=10.0, gt=0, description="DEFAULT 10. A heartbeat older than this makes the lock stale: the run fails with the removal instruction.")
    log: LogConfig = LogConfig()
