"""The `check` command: says in plain words what is fine and what is not, and lists ALL problems at once (standard library only).

How to run (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -m stt check
    python -m stt check --config configs/smoke.yaml
Example output:
    OK       Python 3.12.14 (tested with 3.12)
    OK       package torch 2.14.1
    ...
    INFO     code version 0.0.0 (the manifest MANIFEST.sha256 is not there: made at home by tools/make_manifest.py)
    Everything checked is fine.            (exit code 0)   or   2 problem(s) found: ...   (exit code 2)
Checks: the Python version, every required package (read from the installed metadata, nothing heavy is imported), the model and data folders of the
config when one is given, the free disk space, the memory, the number of cores, the code version and the manifest. The Python version, the disk, the memory
and the cores are information only (no hard-coded thresholds); a missing package, a missing model folder or a missing data folder is a problem.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from stt.config import AppConfig
from stt.orchestrator.codeversion import MANIFEST_NAME, code_version, manifest_hash

TESTED_PYTHON = "3.12"
REQUIRED_PACKAGES = ("pydantic", "PyYAML", "torch", "transformers", "safetensors", "accelerate", "numpy", "scipy", "soundfile", "pyarrow")


@dataclass(frozen=True)
class CheckLine:
    """One line of the check: `level` is OK, INFO or PROBLEM."""

    level: str
    text: str


def memory_gb() -> float | None:
    """Total physical memory in GB (Windows and Linux), or None when it cannot be read."""
    try:
        if sys.platform == "win32":
            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]

            status = Status()
            status.length = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
            return status.total / 1e9
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (OSError, ValueError, AttributeError):
        return None


def run_checks(config: AppConfig | None, project_root: Path, disk_path: Path) -> list[CheckLine]:
    """All the checks. `config` may be None (then the model and data folders are not checked)."""
    lines: list[CheckLine] = []
    version = sys.version_info
    lines.append(CheckLine("OK", f"Python {version.major}.{version.minor}.{version.micro} (the pipeline was tested with {TESTED_PYTHON})"))
    for package in REQUIRED_PACKAGES:
        try:
            lines.append(CheckLine("OK", f"package {package} {metadata.version(package)}"))
        except metadata.PackageNotFoundError:
            lines.append(CheckLine("PROBLEM", f"package {package} is NOT installed: install it from the offline wheels (docs/RUNBOOK.md, step 'install')."))
    if config is None:
        lines.append(CheckLine("INFO", "no --config given: the model folder and the data folder were not checked."))
    else:
        for key, folder in (("model.path", getattr(config.sections.get("model"), "path", None)), ("data.source.path", _data_path(config))):
            if folder is None:
                lines.append(CheckLine("INFO", f"{key} is not set in this config."))
            elif Path(folder).is_dir():
                lines.append(CheckLine("OK", f"{key} exists: {folder}"))
            else:
                lines.append(CheckLine("PROBLEM", f"{key} does not exist: {folder}. Copy the folder there or fix the path in the YAML."))
    free = shutil.disk_usage(disk_path if disk_path.exists() else project_root).free / 1e9
    lines.append(CheckLine("INFO", f"free disk space where the experiments go: {free:.1f} GB (a checkpoint of a model needs about 3 times the model size)"))
    memory = memory_gb()
    lines.append(CheckLine("INFO", f"memory: {memory:.1f} GB" if memory is not None else "memory: could not be read"))
    lines.append(CheckLine("INFO", f"cores: {os.cpu_count()}"))
    suffix = "" if manifest_hash(project_root) is not None else f" (the manifest {MANIFEST_NAME} is not there: it is made at home by tools/make_manifest.py; verify with `python tools/make_manifest.py --verify`)"
    lines.append(CheckLine("INFO", f"code version {code_version(project_root)}{suffix}"))
    return lines


def _data_path(config: AppConfig) -> object:
    source = getattr(config.sections.get("data"), "source", None)
    return getattr(source, "path", None)


def format_report(lines: list[CheckLine]) -> tuple[str, int]:
    """The text of the check and the exit code: 0 when there is no PROBLEM, else 2 with the whole list."""
    text = [f"{line.level:8} {line.text}" for line in lines]
    problems = [line for line in lines if line.level == "PROBLEM"]
    if problems:
        text.append(f"{len(problems)} problem(s) found (all listed above with PROBLEM).")
        return "\n".join(text), 2
    text.append("Everything checked is fine.")
    return "\n".join(text), 0
