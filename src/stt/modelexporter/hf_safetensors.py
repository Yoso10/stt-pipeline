"""The `hf_safetensors` exporter: the format the registry already stores (the adapter module; the only module of the component that touches the model loader).

How to run: this module is not a script; the service uses it. Example (PowerShell, project root; verifies a stored model folder at the `load` level, offline):
    $env:PYTHONPATH = "src"
    python -c "from pathlib import Path; from stt.modelexporter.hf_safetensors import HfSafetensorsExporter, safetensors_parameter_count; print(safetensors_parameter_count(Path('tests/assets/tiny_whisper')))"
Example output:
    216704
v1 rules (C11 brief): the exporter does NOT copy (its result points to the entry's own model folder, so an export never modifies the registered model and uses no extra disk); it verifies at the `load`
level: (1) the folder is a complete Whisper folder (`check_model_folder`), (2) it loads through the ModelLoader offline, (3) the number of parameters that were loaded equals the number the weights file
DECLARES in its header (read with the standard library, no framework: a truncated or partly loaded file would differ), (4) the mel bins of the loaded model equal the `num_mel_bins` of `config.json`, and
(5) when the base model's folder exists on this machine, the architecture fields of `config.json` (mel bins, width, layers, vocabulary) equal the base model's (a fine-tuned model keeps its architecture); without the
base folder this step is skipped and the detail says so. The brief says the parameter count and the mel bins must `match the card`: the model card does not hold them, so they are compared with the weights header,
the config and the base model instead (DECISIONS 69).
"""

from __future__ import annotations

import json
import math
import struct
import time
from pathlib import Path
from typing import Any

from stt.modelexporter.types import ExportResult, VerifyInput, VerifyResult
from stt.modelloader import ModelConfig, ModelLoaderError, check_model_folder, create_loader

ARCHITECTURE_KEYS = ("num_mel_bins", "d_model", "encoder_layers", "decoder_layers", "encoder_attention_heads", "decoder_attention_heads", "vocab_size")


def safetensors_parameter_count(folder: Path) -> int | None:
    """The number of parameters the `.safetensors` files of `folder` declare in their headers (None when there is no such file). Standard library only: 8 bytes of header length, then JSON with a
    shape per tensor. Errors: ValueError for a header that cannot be read."""
    files = sorted(folder.glob("*.safetensors"))
    if not files:
        return None
    total = 0
    for path in files:
        with path.open("rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            header: dict[str, Any] = json.loads(handle.read(length))
        total += sum(math.prod(entry["shape"]) for name, entry in header.items() if name != "__metadata__")
    return total


def _config(folder: Path) -> dict[str, Any]:
    return dict(json.loads((folder / "config.json").read_text(encoding="utf-8")))


class HfSafetensorsExporter:
    """The identity exporter of v1."""

    name = "hf_safetensors"
    requires: tuple[str, ...] = ("transformers", "safetensors")

    def export(self, source: Path, target_dir: Path) -> ExportResult:
        """Point at the entry's own model folder (nothing is copied, nothing is written; `target_dir` is not used)."""
        started = time.monotonic()
        files = sorted(p for p in source.rglob("*") if p.is_file())
        return ExportResult(self.name, source, files, time.monotonic() - started)

    def verify(self, result: ExportResult, check: VerifyInput) -> VerifyResult:
        """The `load`-level verification (see the module docstring). Never raises for a failed check: it returns `ok` false with what differed."""
        folder = result.target_dir
        passed: list[str] = []
        try:
            check_model_folder(folder)
            passed.append("the folder is a complete Whisper folder")
        except ModelLoaderError as exc:
            return VerifyResult("load", False, f"the folder is not a complete model folder: {exc}")
        try:
            info = create_loader(ModelConfig(path=folder)).load().info
        except Exception as exc:  # any failure to load is a failed verification, with the reason
            return VerifyResult("load", False, f"the folder does not load through the ModelLoader: {type(exc).__name__}: {exc}")
        passed.append(f"loads through the ModelLoader ({info.num_parameters} parameters, {info.num_mel_bins} mel bins)")
        problems: list[str] = []
        try:
            declared = safetensors_parameter_count(folder)
        except (OSError, ValueError, KeyError) as exc:
            declared = None
            problems.append(f"the weights header cannot be read ({type(exc).__name__}: {exc})")
        if declared is not None:
            if declared == info.num_parameters:
                passed.append(f"the loaded parameter count equals the {declared} the weights file declares")
            else:
                problems.append(f"the model loaded {info.num_parameters} parameters but the weights file declares {declared}: it may be truncated or only partly loaded")
        config = _config(folder)
        if config.get("num_mel_bins") == info.num_mel_bins:
            passed.append("the mel bins match config.json")
        else:
            problems.append(f"the loaded model has {info.num_mel_bins} mel bins but config.json says {config.get('num_mel_bins')}")
        base = check.base_model_dir
        if base is not None and (base / "config.json").is_file():
            base_config = _config(base)
            differing = [f"{key}: {config.get(key)} against {base_config.get(key)} (base)" for key in ARCHITECTURE_KEYS if config.get(key) != base_config.get(key)]
            if differing:
                problems.append("the architecture differs from the base model: " + "; ".join(differing))
            else:
                passed.append(f"the architecture equals the base model's ({base.name})")
        else:
            passed.append("the base model folder is not on this machine: the architecture was not compared")
        if problems:
            return VerifyResult("load", False, "FAILED: " + "; ".join(problems) + ". Passed: " + "; ".join(passed) + ".")
        return VerifyResult("load", True, "; ".join(passed) + ".")
