"""The `export` configuration section, owned by the ModelExporter component.

How to run: this module is not a script. The composition root registers the section (the Orchestrator task that wires the `export` command does it):
    from stt.config import SectionRegistry, load_config
    from stt.modelexporter.config import ExportConfig
    config = load_config(path, registry=SectionRegistry({"export": ExportConfig}))
    export = config.section("export", ExportConfig)
Example YAML with every key and its DEFAULT: configs/export_default.yaml. The keys and DEFAULTS are those of the C11 brief.
"""

from __future__ import annotations

from pydantic import Field

from stt.config import SectionModel


class ExportConfig(SectionModel):
    """The `export` section."""

    formats: list[str] = Field(default_factory=lambda: ["hf_safetensors"], min_length=1, description="DEFAULT hf_safetensors. The exporters that run after a winner is registered (registered: hf_safetensors).")
    verify: bool = Field(default=True, description="DEFAULT true. Verify every export (hf_safetensors: the folder loads through the ModelLoader and its architecture and parameter count match). false: the export is recorded as `unverified`.")
    verify_samples: int = Field(default=5, ge=1, description="DEFAULT 5. How many files a transcription check uses (for exporters that verify at the `transcribe` level; the caller picks the files).")
    max_wer_diff: float = Field(default=0.02, ge=0.0, description="DEFAULT 0.02. The absolute WER difference allowed between an exported model and the registered one on the check files (transcribe level).")
