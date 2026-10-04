"""Download all packages as wheels for the TARGET Python and Windows, test an offline install.

How to run (from the project root, with the project virtual environment active):
    python tools/fetch_wheels.py --requirements requirements.txt --out wheels/
Options: --python-version 3.12 (target Python, default 3.12)   --platform win_amd64 (default)
         --lock requirements.lock (pinned result)   --venv-python PATH (a python.exe of the target
         version used to build the clean test environment; default: the current python, which must
         have the target version)
What it does: 1) pip download of every package and dependency into wheels/ (internet needed here,
2) creates a clean temporary virtual environment, 3) installs from wheels/ with NO network
(--no-index --find-links), 4) runs pip check, 5) writes requirements.lock (exact versions).
Example output:
    Downloaded 52 wheels to wheels
    Offline install test: OK (pip check clean)
    Wrote requirements.lock (52 pinned packages)
    Next: copy wheels/ and requirements.lock to the closed machine; there run
    python -m pip install --no-index --find-links wheels -r requirements.lock
Exit codes: 0 OK, 1 usage or environment problem, 2 a wheel is missing for the target, 3 the
offline install test failed. Running again reuses a complete wheels/ folder of the same input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import INFO_FILE, FetchError, read_info  # noqa: E402

# A runner executes a command and returns (exit code, combined output). Tests pass a fake.
Runner = Callable[[Sequence[str]], tuple[int, str]]


class WheelMissingError(FetchError):
    """No wheel exists for the target Python and Windows (the O1 signal)."""


class OfflineInstallError(FetchError):
    """The clean offline install test failed."""


@dataclass(frozen=True)
class WheelsResult:
    """Outcome of fetch_wheels."""

    out: Path
    reused: bool
    wheel_count: int
    lock_file: Path
    pinned: int


def subprocess_runner(cmd: Sequence[str]) -> tuple[int, str]:
    """Real runner: run `cmd`, return (exit code, stdout+stderr)."""
    proc = subprocess.run(list(cmd), capture_output=True, text=True, encoding="utf-8", errors="replace")
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _python_version_of(python: str, runner: Runner) -> str:
    code, text = runner([python, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"])
    if code != 0:
        raise FetchError(f"Cannot run '{python}': {text.strip()[:200]}")
    return text.strip().splitlines()[-1]


def _explain_download_failure(output: str, python_version: str) -> FetchError:
    match = re.search(r"No matching distribution found for ([^\s]+)", output)
    pkg = match.group(1) if match else None
    if pkg is None:
        match = re.search(r"Could not find a version that satisfies the requirement ([^\s]+)", output)
        pkg = match.group(1) if match else None
    if pkg is not None:
        return WheelMissingError(
            f"No wheel for package '{pkg}' for Python {python_version} on Windows. "
            "Try another Python version (O1) or ask before replacing the package."
        )
    tail = " | ".join(output.strip().splitlines()[-3:])
    return FetchError(f"pip download failed: {tail}. Check the internet connection, then run again.")


def fetch_wheels(
    requirements: Path,
    out: Path,
    lock_file: Path,
    python_version: str = "3.12",
    platform: str = "win_amd64",
    venv_python: str | None = None,
    runner: Runner = subprocess_runner,
) -> WheelsResult:
    """Download wheels, test an offline install in a clean venv, write the lock file.

    Inputs: requirements file, output folder, lock file path, target Python version and
    platform tag, the python used to create the test venv (must match `python_version`),
    and a command runner.
    Outputs: WheelsResult. Raises WheelMissingError (no wheel), OfflineInstallError (the offline
    install failed) or FetchError (anything else).
    """
    if not requirements.is_file():
        raise FetchError(f"Requirements file not found: {requirements}.")
    python = venv_python or sys.executable
    actual = _python_version_of(python, runner)
    if actual != python_version:
        raise FetchError(
            f"The Python used for the test venv is {actual}, but the target is {python_version}. "
            f"Run this tool with a Python {python_version} (or pass --venv-python PATH)."
        )

    req_hash = _sha256(requirements)
    existing = read_info(out) if out.exists() else None
    reused = bool(
        existing
        and existing.get("requirements_sha256") == req_hash
        and existing.get("python_version") == python_version
        and existing.get("platform") == platform
    )
    if not reused:
        partial = out.with_name(out.name + ".partial")
        for stale in (partial, out):
            if stale.exists():
                shutil.rmtree(stale)
        partial.mkdir(parents=True)
        try:
            cmd = [
                python, "-m", "pip", "download",
                "-r", str(requirements), "--dest", str(partial),
                "--python-version", python_version, "--platform", platform,
                "--implementation", "cp", "--abi", f"cp{python_version.replace('.', '')}",
                "--only-binary=:all:", "--disable-pip-version-check",
            ]  # fmt: skip
            code, output = runner(cmd)
            if code != 0:
                raise _explain_download_failure(output, python_version)
            names = sorted(p.name for p in partial.glob("*.whl"))
            info = {
                "requirements_sha256": req_hash,
                "python_version": python_version,
                "platform": platform,
                "wheels": names,
            }
            (partial / INFO_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")
            partial.rename(out)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        existing = read_info(out)

    wheel_count = len(existing["wheels"]) if existing else 0  # type: ignore[arg-type]
    pinned = _offline_install_test(requirements, out, lock_file, python, runner)
    return WheelsResult(out, reused, wheel_count, lock_file, pinned)


def _offline_install_test(
    requirements: Path, wheels: Path, lock_file: Path, python: str, runner: Runner
) -> int:
    """Install from `wheels` into a clean temporary venv with no network; write the lock file."""
    tmp = Path(tempfile.mkdtemp(prefix="wheels_test_"))
    try:
        venv = tmp / "venv"
        code, output = runner([python, "-m", "venv", str(venv)])
        if code != 0:
            raise FetchError(f"Could not create the test virtual environment: {output.strip()[-300:]}")
        vpy = str(_venv_python(venv))
        code, output = runner([
            vpy, "-m", "pip", "install", "--no-index", "--find-links", str(wheels),
            "-r", str(requirements), "--disable-pip-version-check",
        ])  # fmt: skip
        if code != 0:
            tail = " | ".join(output.strip().splitlines()[-3:])
            raise OfflineInstallError(
                f"Offline install test FAILED: {tail}. "
                "Run fetch_wheels again (it includes dependencies) and read the named package."
            )
        code, output = runner([vpy, "-m", "pip", "check", "--disable-pip-version-check"])
        if code != 0:
            raise OfflineInstallError(f"Offline install OK but pip check found problems: {output.strip()[-300:]}")
        code, output = runner([vpy, "-m", "pip", "freeze", "--disable-pip-version-check"])
        if code != 0:
            raise FetchError(f"pip freeze failed: {output.strip()[-300:]}")
        lines = sorted(line for line in output.splitlines() if line.strip() and "==" in line)
        lock_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return len(lines)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None, runner: Runner = subprocess_runner) -> int:
    """Command-line entry. Returns the exit code. `runner` is replaced in tests."""
    parser = argparse.ArgumentParser(description="Build and test an offline wheel bundle.")
    parser.add_argument("--requirements", type=Path, default=Path("requirements.txt"))
    parser.add_argument("--out", type=Path, default=Path("wheels"))
    parser.add_argument("--lock", type=Path, default=Path("requirements.lock"))
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument("--platform", default="win_amd64")
    parser.add_argument("--venv-python", default=None)
    args = parser.parse_args(argv)
    try:
        result = fetch_wheels(
            args.requirements, args.out, args.lock, args.python_version, args.platform, args.venv_python, runner
        )
    except WheelMissingError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    except OfflineInstallError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 3
    except FetchError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    verb = "Reused" if result.reused else "Downloaded"
    print(f"{verb} {result.wheel_count} wheels in {result.out}")
    print("Offline install test: OK (pip check clean)")
    print(f"Wrote {result.lock_file} ({result.pinned} pinned packages)")
    print("Next: copy the wheels folder and requirements.lock to the closed machine; there run")
    print(f"  python -m pip install --no-index --find-links {result.out.as_posix()} -r {result.lock_file.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
