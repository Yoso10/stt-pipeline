"""The `registry` configuration section, owned by the ModelRegistry component.

How to run: this module is not a script. The composition root registers the section:
    from stt.config import SectionRegistry, load_config
    from stt.registry.config import RegistryConfig
    config = load_config(path, registry=SectionRegistry({"registry": RegistryConfig}))
    registry = config.section("registry", RegistryConfig)
Example YAML with every key and its DEFAULT: configs/registry_default.yaml. The registry is ONE self-contained folder (entries, cards, index): copy the whole
folder to a removable drive to move it. It lives next to the project (default `registry`), outside the experiments folder, so no experiment cleanup can reach it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from stt.config import SectionModel


class RegistryConfig(SectionModel):
    """The `registry` section."""

    dir: Path = Field(default=Path("registry"), description="DEFAULT registry (a sibling of the experiments folder, covered by .gitignore). The folder of the registry; copy it whole to move it.")
    keep_finalists: bool = Field(default=False, description="DEFAULT false. true keeps the folders of the other finalists and the checkpoints after a registration.")
    cleanup: bool = Field(default=True, description="DEFAULT true. Delete the other finalists and the checkpoints named in the request after a successful registration.")
    transfer: Literal["move", "copy"] = Field(default="move", description="DEFAULT move (rename on the same drive, copy and delete across drives). copy keeps the original folder.")
    checksums: bool = Field(default=True, description="DEFAULT true. Write SHA256SUMS for every entry (detects a corrupted copy after moving the registry).")
