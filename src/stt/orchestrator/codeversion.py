"""The code version number (standard library only).

How to run: this module is not a script; `check` and the experiment records use it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.orchestrator.codeversion import code_version; print(code_version())"
Example output:
    0.0.0
Option (a), owner decision 2026-10-05 (O71): the version is the constant `stt.__version__`. Option (b), the hash of the sources, is prepared:
when a `MANIFEST.sha256` file exists in the project root (made by `tools/make_manifest.py`, a Wave 3 tool that does not exist yet), the
first 8 characters of its hash are appended (`0.0.0+1a2b3c4d`). Verifying the sources against the manifest is the job of that tool and of `check`
once the tool exists; today `check` only says whether the manifest is there.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import stt

MANIFEST_NAME = "MANIFEST.sha256"


def manifest_hash(root: Path) -> str | None:
    """The first 8 hex characters of the SHA-256 of `<root>/MANIFEST.sha256`, or None when the file does not exist."""
    path = root / MANIFEST_NAME
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def code_version(root: Path | None = None) -> str:
    """`stt.__version__`, plus `+<8 characters>` of the manifest hash when a manifest exists in `root`."""
    suffix = manifest_hash(root) if root is not None else None
    return stt.__version__ if suffix is None else f"{stt.__version__}+{suffix}"
