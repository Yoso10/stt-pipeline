r"""verify_clone: check that a fresh `git clone` is ready to run on this machine (read-only; exit code 0 = everything is fine, 1 = something must be fixed).

How to run (PowerShell, in the project folder, with the project's Python):
    .\.venv\Scripts\python.exe tools\verify_clone.py
    .\.venv\Scripts\python.exe tools\verify_clone.py --config configs\my_work_cpu.yaml --model models\whisper-large-v3-turbo-ivrit
    .\.venv\Scripts\python.exe tools\verify_clone.py --no-tests            # skip the quick unit tests (about one minute)
Linux: ./.venv/bin/python tools/verify_clone.py
Example output (home laptop):
    OK    Python 3.12.14 (the project was tested with 3.12)
    OK    package torch 2.14.1+cpu
    ...
    OK    model folder models\whisper-large-v3-turbo-ivrit passes the pipeline's own check
    OK    data: manifest.csv has the columns audio, text, group
    INFO  RAM 16.93 GB total, 7.26 GB free; disk 23.9 GB free
    OK    quick unit tests: Ran 243 tests ... OK
    Everything checked is fine.
What it checks: the Python and torch versions, that every package of requirements.txt can be imported, the model folder (`check_model_folder`), the data folder and the three columns of your config
(the config must be filled in: the data folder and three column names), the free RAM and disk (information, with a warning below 8 GB of free RAM), and a quick subset of the unit tests, offline.
It changes nothing and needs no network.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PACKAGES = {  # name in requirements.txt -> module to import
    "pydantic": "pydantic", "PyYAML": "yaml", "torch": "torch", "transformers": "transformers", "safetensors": "safetensors", "accelerate": "accelerate",
    "optuna": "optuna", "numpy": "numpy", "scipy": "scipy", "soundfile": "soundfile", "pyarrow": "pyarrow",
}
QUICK_TESTS = ["tests.test_config", "tests.test_textnormalizer", "tests.test_splitter", "tests.test_samplefilter", "tests.test_subsampler", "tests.test_datasource", "tests.test_selector", "tests.test_device"]
MIN_FREE_RAM_GB = 8.0  # the requirement written in docs/WORK_RUNBOOK.md (a warning, not a failure)
GB = 1e9
problems: list[str] = []
missing: list[str] = []  # what is absent, in plain words (printed at the end with where to get it)
DEFAULT_MODEL = Path("models") / "whisper-large-v3-turbo-ivrit"  # the model folder checked when the config cannot be read


def report(kind: str, text: str) -> None:
    print(f"{kind:<5} {text}")
    if kind == "FAIL":
        problems.append(text)


def ram() -> tuple[float, float] | None:
    try:
        if os.name == "nt":

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong), ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]

            status = Status()
            status.length = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
            return status.total / GB, status.free / GB
        page = os.sysconf("SC_PAGE_SIZE")
        return page * os.sysconf("SC_PHYS_PAGES") / GB, page * os.sysconf("SC_AVPHYS_PAGES") / GB
    except Exception:  # noqa: BLE001
        return None


def check_python_and_packages() -> None:
    report("OK", f"Python {platform.python_version()} (the project was tested with 3.12)")
    for name, module in PACKAGES.items():
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, "__version__", "")
            report("OK", f"package {name} {version}".rstrip())
        except Exception as exc:  # noqa: BLE001
            report("FAIL", f"package {name} cannot be imported ({type(exc).__name__}: {exc}). Install from the wheel bundle: pip install --no-index --find-links wheels -r requirements.lock")
    try:
        import torch

        build = "a CPU-only build" if not torch.version.cuda else f"built for CUDA {torch.version.cuda}"
        report("INFO", f"torch {torch.__version__} is {build}; CUDA available: {torch.cuda.is_available()}")
    except Exception:  # noqa: BLE001
        pass


def check_config_and_data(config_path: Path, model_override: Path | None) -> None:
    try:
        from stt.config import load_config
        from stt.datasource import DataConfig
        from stt.modelloader import ModelConfig
        from stt.modelloader.whisper_folder import check_model_folder
        from stt.orchestrator.wiring import section_registry
    except Exception as exc:  # noqa: BLE001
        report("FAIL", f"the project code cannot be imported ({type(exc).__name__}: {exc}). Run this script with the project's Python: .\\.venv\\Scripts\\python.exe")
        return
    config = None
    if not config_path.is_file():
        report("FAIL", f"the config {config_path} does not exist. Copy configs\\work_cpu.yaml to configs\\my_work_cpu.yaml and fill in the 4 values (data folder, three column names).")
        missing.append("a filled-in config (copy configs\\work_cpu.yaml to configs\\my_work_cpu.yaml and fill in the 4 values)")
    else:
        try:
            config = load_config(config_path, registry=section_registry(), env_file=None, environ={}, apply_environment=False)
            report("OK", f"config {config_path} loads (the data folder exists)")
        except Exception as exc:  # noqa: BLE001
            first = str(exc).strip().splitlines()
            report("FAIL", f"the config {config_path} is not ready: {' | '.join(first[:4])}. Fill in the <<< FILL IN >>> places (data folder, three column names).")
            missing.append("the 4 values in your config (the data folder and the three column names), or the data folder itself")
    # the model folder is checked even when the config is not ready: the path is the one of the config, or the shipped default
    model = model_override if model_override is not None else (config.section("model", ModelConfig).path if config is not None else DEFAULT_MODEL)
    model = model if Path(model).is_absolute() else ROOT / model
    try:
        check_model_folder(Path(model), "he")
        report("OK", f"model folder {model} passes the pipeline's own check")
    except Exception as exc:  # noqa: BLE001
        report("FAIL", f"model folder: {exc}")
        missing.append(f"the model folder {model} (not in git: copy it from home, or fetch it with tools\\fetch_model.py on a machine with internet)")
    if config is None:
        return
    data = config.section("data", DataConfig)
    folder = Path(data.source.path)
    if data.source.type == "csv":
        manifest = folder / "manifest.csv"
        if not manifest.is_file():
            report("FAIL", f"data: {manifest} does not exist (a csv source needs a file named manifest.csv in {folder}).")
            missing.append(f"the data: {manifest} (not in git: bring your data folder, docs/WORK_RUNBOOK.md, TOMORROW step 3)")
            return
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        wanted = {"audio": data.columns.audio, "text": data.columns.text, "group": data.columns.group}
        missing_columns = [f"{role} '{name}'" for role, name in wanted.items() if name not in header]
        if missing_columns:
            report("FAIL", f"data: manifest.csv has the columns {', '.join(header)}; missing: {', '.join(missing_columns)}. Fix data.columns in your config.")
        else:
            report("OK", f"data: manifest.csv has the columns {', '.join(wanted.values())}")
    else:
        files = list(folder.rglob("*.parquet"))
        report("OK" if files else "FAIL", f"data: {len(files)} parquet file(s) in {folder}")


def check_machine() -> None:
    memory = ram()
    disk = shutil.disk_usage(ROOT).free / GB
    if memory is None:
        report("INFO", f"RAM unknown; disk {disk:.1f} GB free")
        return
    report("INFO", f"RAM {memory[0]:.2f} GB total, {memory[1]:.2f} GB free; disk {disk:.1f} GB free on the drive of {ROOT}")
    if memory[1] < MIN_FREE_RAM_GB:
        report("WARN", f"only {memory[1]:.1f} GB of RAM is free; the turbo training peaks at 6.4 to 6.7 GB. Close other programs (docs/WORK_RUNBOOK.md).")


def run_quick_tests() -> None:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run([sys.executable, "-m", "unittest", *QUICK_TESTS], cwd=ROOT, env=env, capture_output=True, text=True)
    tail = [line for line in (result.stderr + result.stdout).strip().splitlines() if line.strip()]
    ran = next((line for line in tail if line.startswith("Ran ")), "")
    if result.returncode == 0:
        report("OK", f"quick unit tests: {ran} ... OK")
    else:
        report("FAIL", f"quick unit tests failed: {' | '.join(tail[-6:])}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that a fresh clone is ready (read-only).")
    parser.add_argument("--config", type=Path, default=None, help="your filled-in config (default: configs/my_work_cpu.yaml, else configs/work_cpu.yaml)")
    parser.add_argument("--model", type=Path, default=None, help="the model folder (default: model.path of the config)")
    parser.add_argument("--no-tests", action="store_true", help="skip the quick unit tests")
    args = parser.parse_args()
    config = args.config or (ROOT / "configs" / "my_work_cpu.yaml" if (ROOT / "configs" / "my_work_cpu.yaml").is_file() else ROOT / "configs" / "work_cpu.yaml")
    check_python_and_packages()
    check_config_and_data(config if config.is_absolute() else ROOT / config, args.model)
    check_machine()
    if not args.no_tests:
        run_quick_tests()
    print()
    if problems:
        print(f"{len(problems)} problem(s) found. Fix them, then run this script again.")
        if missing:
            print("WHAT IS MISSING (and where to get it: docs/WORK_RUNBOOK.md, the page TOMORROW):")
            for item in missing:
                print(f"  - {item}")
        return 1
    print("Everything checked is fine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
