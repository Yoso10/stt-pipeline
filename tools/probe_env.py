r"""probe_env: say in plain words what this machine has (read-only; no network; it always ends with exit code 0).

How to run (PowerShell, project folder; the project's Python is better, any Python 3 works):
    .\.venv\Scripts\python.exe tools\probe_env.py
    .\.venv\Scripts\python.exe tools\probe_env.py --model models\whisper-large-v3-turbo-ivrit
Example output (home laptop):
    Python            3.12.14
    torch             2.14.1+cpu (CUDA build: none)
    CPU               8 logical threads, 4 physical cores
    RAM               15.77 GB total, 3.9 GB free
    Disk              22.3 GB free on the drive of C:\Users\...\stt-pipeline
    Model folder      OK: models\whisper-large-v3-turbo-ivrit passes the model-folder check
What it changes: nothing. A line that cannot be read says "unknown". Read the numbers against docs/WORK_RUNBOOK.md (the RAM and disk needs).
"""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GB = 1e9


def memory() -> tuple[str, str]:
    """(total, free) RAM as text; 'unknown' when it cannot be read."""
    try:
        if os.name == "nt":

            class Status(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong), ("free_page", ctypes.c_ulonglong), ("total_virtual", ctypes.c_ulonglong),
                    ("free_virtual", ctypes.c_ulonglong), ("free_extended", ctypes.c_ulonglong),
                ]

            status = Status()
            status.length = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
            return f"{status.total / GB:.2f} GB", f"{status.free / GB:.2f} GB"
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        free = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
        return f"{total / GB:.2f} GB", f"{free / GB:.2f} GB"
    except Exception:  # noqa: BLE001  a probe must never stop
        return "unknown", "unknown"


def physical_cores() -> str:
    """The number of physical cores (Windows: asked from the system; elsewhere unknown), as text."""
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum"],
                capture_output=True, text=True, timeout=30,
            ).stdout.strip()
            if out.isdigit():
                return out
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def torch_line() -> str:
    """torch version and CUDA build, or why torch is not there."""
    try:
        import torch

        build = torch.version.cuda
        return f"{torch.__version__} (CUDA build: {build if build else 'none'}; CUDA available: {torch.cuda.is_available()})"
    except ImportError:
        return "NOT INSTALLED in this Python (use the project's .venv)"
    except Exception as exc:  # noqa: BLE001
        return f"cannot be read ({type(exc).__name__}: {exc})"


def model_line(folder: Path) -> str:
    """Whether the model folder passes the pipeline's own check (`check_model_folder`)."""
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from stt.modelloader.whisper_folder import check_model_folder

        check_model_folder(folder, "he")
        return f"OK: {folder} passes the model-folder check"
    except Exception as exc:  # noqa: BLE001
        return f"PROBLEM: {exc}"


def main() -> int:
    """Print the probe. Always returns 0."""
    parser = argparse.ArgumentParser(description="Say what this machine has (read-only).")
    parser.add_argument("--model", type=Path, default=Path("models") / "whisper-large-v3-turbo-ivrit", help="the model folder to check")
    try:
        args = parser.parse_args()
        model = args.model if args.model.is_absolute() else ROOT / args.model
        total, free = memory()
        try:
            disk = f"{shutil.disk_usage(ROOT).free / GB:.1f} GB free on the drive of {ROOT}"
        except Exception:  # noqa: BLE001
            disk = "unknown"
        print(f"Python            {platform.python_version()} ({platform.system()} {platform.release()})")
        print(f"torch             {torch_line()}")
        print(f"CPU               {os.cpu_count()} logical threads, {physical_cores()} physical cores")
        print(f"RAM               {total} total, {free} free")
        print(f"Disk              {disk}")
        print(f"Model folder      {model_line(model)}")
    except SystemExit:
        pass
    except Exception as exc:  # noqa: BLE001
        print(f"probe_env could not finish ({type(exc).__name__}: {exc}); this is not an error of the machine")
    return 0


if __name__ == "__main__":
    sys.exit(main())
