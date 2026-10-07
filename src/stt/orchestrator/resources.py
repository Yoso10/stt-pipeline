"""What a run costs in memory and disk: the size of a folder, the peak memory of this process, and the benchmark's resources text (Wave 3 task 7; DECISIONS 76, 78, O118).

How to run: this module is not a script; the `benchmark` stage uses it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.orchestrator.resources import human_bytes; print(human_bytes(453_000_000))"
Example output:
    453 MB
`peak_memory_bytes()` reads the PEAK WORKING SET of this process (the most physical memory it used) through the Windows API `GetProcessMemoryInfo`, called with the standard-library `ctypes`
(no new dependency). Anywhere else, or when the call fails, it returns None and the benchmark says so plainly. Standard library only; nothing is written or deleted.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def folder_size(path: Path) -> int:
    """The total size in bytes of the files under `path` (symbolic links are not followed). Reads only."""
    total = 0
    for folder, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(folder) / name).lstat().st_size
            except OSError:
                continue
    return total


def human_bytes(size: float) -> str:
    """A size for people: `812 KB`, `453 MB` or `1.8 GB`."""
    if size >= 1_000_000_000:
        return f"{size / 1e9:.1f} GB"
    if size >= 1_000_000:
        return f"{size / 1e6:.0f} MB"
    return f"{max(size, 0) / 1e3:.0f} KB"


def peak_memory_bytes() -> int | None:
    """The peak working set of this process in bytes on Windows (standard-library `ctypes`), else None. Never raises."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    except Exception:  # noqa: BLE001  a measurement that cannot be made is reported as `not measured`, never an error
        return None


def resources_text(folder_bytes: int, checkpoints: int, largest_checkpoint_bytes: int, peak_bytes: int | None, trial_folders: int) -> list[str]:
    """The `Resources` lines of the benchmark: the disk of the measuring trial folder, the planned disk of the whole `run`, and the peak memory.

    A run keeps up to TWO checkpoints per trial (the latest and the best): when the measuring training kept fewer, the planned disk also shows the upper bound."""
    lines = ["Resources (measured on the benchmark's own training):", f"  disk of the measuring trial folder: {human_bytes(folder_bytes)} ({checkpoints} checkpoint folder(s), the largest {human_bytes(largest_checkpoint_bytes)})"]
    planned = trial_folders * folder_bytes
    upper = trial_folders * (folder_bytes + max(0, 2 - checkpoints) * largest_checkpoint_bytes)
    lines.append(f"  planned disk of `python -m stt run`: about {human_bytes(planned)} ({trial_folders} trial folders: the search trials and the finalists)")
    if upper > planned:
        lines.append(f"  a longer training keeps the latest AND the best checkpoint: plan up to about {human_bytes(upper)}")
    lines.append(
        f"  peak memory of this process: {human_bytes(peak_bytes)} (the peak working set)" if peak_bytes is not None
        else "  peak memory: NOT MEASURED (only Windows is supported through the standard library, or the call failed): watch python.exe in Task Manager (Details) while it runs"
    )
    return lines
