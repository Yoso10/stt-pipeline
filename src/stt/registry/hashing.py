"""SHA256SUMS: write, read and check the checksums of an entry (standard library only).

How to run: this module is not a script; the registry uses it. Example (PowerShell, project root; checks the files of a folder against its SHA256SUMS):
    $env:PYTHONPATH = "src"
    python -c "from pathlib import Path; from stt.registry.hashing import check_sums; print(check_sums(Path('registry/model_0001')))"
The file `SHA256SUMS` has one line per file: the hash, two spaces, and the path relative to the entry folder with forward slashes (the format of `sha256sum -c`),
UTF-8, sorted by path. It lists every file of the entry except itself. It detects a corrupted or changed copy after the registry was moved between machines on a
removable drive. It is a transfer-integrity check; it is independent of how the ModelLoader identifies a model (by name and path).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from stt.registry.types import VerifyReport

SUMS_NAME = "SHA256SUMS"
CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    """The SHA-256 of a file (read in 1 MB pieces, so large model files do not fill the memory)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def list_files(folder: Path, exclude: tuple[str, ...] = (SUMS_NAME,)) -> list[str]:
    """The files under `folder` as sorted relative paths with forward slashes, without the excluded names (at the top level)."""
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and p.relative_to(folder).as_posix() not in exclude)


def render_sums(folder: Path) -> str:
    """The text of SHA256SUMS for the files under `folder`."""
    return "".join(f"{sha256_file(folder / rel)}  {rel}\n" for rel in list_files(folder))


def write_sums(folder: Path) -> Path:
    """Write `SHA256SUMS` into `folder` and return its path."""
    path = folder / SUMS_NAME
    path.write_text(render_sums(folder), encoding="utf-8", newline="\n")
    return path


def read_sums(path: Path) -> dict[str, str]:
    """Read a SHA256SUMS file into {relative path: hash}. Errors: ValueError for a line that is not `<64 hex characters>  <path>`."""
    sums: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        digest, _, rel = line.partition("  ")
        if len(digest) != 64 or not rel or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"line {number} of {path} is not '<sha256>  <path>'")
        sums[rel] = digest
    return sums


def check_sums(folder: Path) -> VerifyReport:
    """Check the files of `folder` against its SHA256SUMS: missing files, changed files, and files that are not listed (reported as mismatching)."""
    listed = read_sums(folder / SUMS_NAME)
    missing = tuple(sorted(rel for rel in listed if not (folder / rel).is_file()))
    mismatching = [rel for rel, digest in sorted(listed.items()) if (folder / rel).is_file() and sha256_file(folder / rel) != digest]
    extra = [f"{rel} (not listed in {SUMS_NAME})" for rel in list_files(folder) if rel not in listed]
    bad = tuple(mismatching + extra)
    return VerifyReport(len(listed), missing, bad, not missing and not bad)
