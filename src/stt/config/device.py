"""Choose the compute device from `runtime.device` (DECISIONS 87): `cpu` (the default), `cuda` or `auto`.

How to run: this module is not a script; the composition root calls `resolve_device` once per command. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.config.device import resolve_device; print(resolve_device('cpu'))"
Example output:
    cpu
Rules: `cpu` never looks at torch. `auto` is `cuda` when torch reports a usable GPU, otherwise `cpu`, and the choice is logged at INFO.
`cuda` with no usable GPU FAILS with a ConfigError (exit code 2) that names the key, what was found (torch version, CUDA build, whether CUDA is
available, the number of devices) and how to fix it. There is never a silent fall back to the CPU. torch is imported only inside the default probe,
so importing this module (and `stt.config`) does not import torch.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from stt.config.errors import ConfigError

logger = logging.getLogger("stt.config")

DEVICES = ("cpu", "cuda", "auto")


@dataclass(frozen=True)
class TorchFacts:
    """What torch says about the GPU: `version` is None when torch is not installed."""

    version: str | None
    cuda_build: str | None  # the CUDA version torch was built for; None for a CPU-only build
    cuda_available: bool
    device_count: int
    first_device_name: str | None = None


def torch_facts() -> TorchFacts:
    """Ask torch (the default probe; imports torch here and only here)."""
    try:
        import torch
    except ImportError:
        return TorchFacts(None, None, False, 0)
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    name = str(torch.cuda.get_device_name(0)) if count else None
    return TorchFacts(str(torch.__version__), torch.version.cuda, available, count, name)


def describe(facts: TorchFacts) -> str:
    """One line with what was found."""
    if facts.version is None:
        return "torch is not installed"
    build = f"built for CUDA {facts.cuda_build}" if facts.cuda_build else "a CPU-only build (no CUDA)"
    name = f", first device: {facts.first_device_name}" if facts.first_device_name else ""
    return f"torch {facts.version} ({build}), cuda available: {facts.cuda_available}, device count: {facts.device_count}{name}"


def resolve_device(requested: str, probe: Callable[[], TorchFacts] | None = None) -> str:
    """Return `"cpu"` or `"cuda"` for the setting `requested` (`runtime.device`).

    `probe` is injectable for tests (default: `torch_facts`). Errors: ConfigError (exit code 2) for an unknown value, and for `cuda` when no GPU
    is usable; the message names `runtime.device`, what was found and how to fix it.
    """
    if requested not in DEVICES:
        raise ConfigError(f"runtime.device '{requested}' is not valid. Use one of: {', '.join(DEVICES)}.")
    if requested == "cpu":
        logger.debug("runtime.device is cpu.")
        return "cpu"
    facts = (probe or torch_facts)()
    usable = facts.cuda_available and facts.device_count > 0
    if requested == "auto":
        chosen = "cuda" if usable else "cpu"
        logger.info("runtime.device is auto: using %s (%s).", chosen, describe(facts))
        return chosen
    if not usable:
        raise ConfigError(
            f"runtime.device is 'cuda' but no usable GPU was found: {describe(facts)}. "
            "Fix: (1) this machine needs an NVIDIA GPU with a working driver (check with nvidia-smi); "
            "(2) the installed torch must be a CUDA build (a '+cpu' or CUDA-less torch cannot use a GPU; install the CUDA wheel of the same torch version); "
            "(3) or set runtime.device to 'cpu' (or 'auto') knowingly. The run was stopped on purpose: it does NOT fall back to the CPU by itself."
        )
    logger.info("runtime.device is cuda: using the GPU (%s).", describe(facts))
    return "cuda"
