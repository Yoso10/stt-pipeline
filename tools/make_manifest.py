"""Write (and verify) MANIFEST.sha256: the SHA-256 of every file of a release, so that the code on the closed machine can be proven to be the code that was tested at home (home-side tool).

How to run (from the project root, any Python 3, standard library only; no network, no other package):
    python tools/make_manifest.py --root . --out MANIFEST.sha256              # write the manifest
    python tools/make_manifest.py --root . --verify MANIFEST.sha256           # verify a tree against a manifest
Options: --exclude PATTERN (repeatable; a file pattern such as "configs/work.yaml" or "*.log", matched against the path relative to --root, added to the built-in exclusions).
Example output (write):
    Wrote MANIFEST.sha256: 241 files, manifest hash 1a2b3c4d.
    Next: copy the WHOLE folder (the manifest included) to the closed machine; `python -m stt check` finds it and prints the code version `0.0.0+1a2b3c4d`.
    Verify any time with: python tools/make_manifest.py --root . --verify MANIFEST.sha256
Example output (verify, a file was changed):
    CHANGED  src/stt/orchestrator/cli.py
    VERIFY FAILED: 1 changed, 0 missing (240 files match). Copy the release again from home.
What goes in: every file under --root EXCEPT the local and runtime folders (.git, .venv, venv, data, models, wheels, experiments, registry at the root; __pycache__, .pytest_cache, .mypy_cache and
*.egg-info anywhere), compiled files (*.pyc), `.env`, the manifest file itself and your --exclude patterns. The file lists `<sha256>  <path>` per line (the format of `sha256sum -c`), paths relative to --root with
forward slashes, sorted, one newline at the end of every line, no date and no machine data: the same files always give the same manifest, byte for byte. The hashes are of the raw bytes of the files, so make the
manifest from the very folder you copy (a copy or a fresh `git clone` that converts line endings gives different bytes and a different manifest).
Verify: CHANGED or MISSING files fail (exit code 1). A file under --root that is NOT in the manifest is only listed as `EXTRA` (exit code 0): for example a config you copied for the work machine.
Exit codes: 0 = written / verified, 1 = verification failed, 2 = a usage error (the folder or the manifest does not exist, a malformed manifest line, an unsafe path in the manifest).
Edge cases: the output is written through a temporary file and renamed, so an interrupted run never leaves a half manifest. An existing manifest is replaced (that is the purpose; the old one is a derived file).
Home-side tool: the pipeline never imports it. It reads files and writes only the manifest.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

MANIFEST_NAME = "MANIFEST.sha256"
ROOT_ONLY_DIRS = frozenset({".git", ".venv", "venv", "data", "models", "wheels", "experiments", "registry"})
ANY_DEPTH_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache"})
ANY_DEPTH_DIR_PATTERNS = ("*.egg-info",)
EXCLUDED_FILE_PATTERNS = ("*.pyc", ".env")
CHUNK = 1 << 20


class ManifestError(Exception):
    """A usage error the owner can act on (exit code 2). The message says what is wrong and how to fix it."""


@dataclass(frozen=True)
class VerifyReport:
    """The result of a verification: the file counts and the problems."""

    matching: int
    changed: tuple[str, ...]
    missing: tuple[str, ...]
    extra: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True when no listed file is changed or missing (extra files do not fail)."""
        return not self.changed and not self.missing


def file_sha256(path: Path) -> str:
    """The SHA-256 (hex) of the raw bytes of `path`, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _skip_dir(relative: str, name: str) -> bool:
    if "/" not in relative and name in ROOT_ONLY_DIRS:
        return True
    return name in ANY_DEPTH_DIRS or any(fnmatch.fnmatch(name, pattern) for pattern in ANY_DEPTH_DIR_PATTERNS)


def collect_files(root: Path, excludes: Sequence[str] = (), skip: Sequence[Path] = ()) -> list[str]:
    """The release files under `root` as sorted relative posix paths (see the module header for what is left out). `skip` holds absolute paths that are never listed (the manifest itself)."""
    skipped = {p.resolve() for p in skip}
    found: list[str] = []
    for folder, dirs, files in os.walk(root):
        here = Path(folder)
        prefix = here.relative_to(root).as_posix()
        prefix = "" if prefix == "." else prefix + "/"
        dirs[:] = sorted(d for d in dirs if not _skip_dir(prefix + d, d))
        for name in files:
            relative = prefix + name
            if (here / name).resolve() in skipped or name == MANIFEST_NAME and prefix == "":
                continue
            if any(fnmatch.fnmatch(name, p) for p in EXCLUDED_FILE_PATTERNS) or any(fnmatch.fnmatch(relative, p) for p in excludes):
                continue
            found.append(relative)
    return sorted(found)


def build_manifest(root: Path, excludes: Sequence[str] = (), skip: Sequence[Path] = ()) -> str:
    """The text of the manifest of `root`: one `<sha256>  <path>` line per file, sorted, deterministic."""
    return "".join(f"{file_sha256(root / relative)}  {relative}\n" for relative in collect_files(root, excludes, skip))


def parse_manifest(text: str) -> dict[str, str]:
    """The manifest as {relative path: sha256}. Errors: ManifestError for a malformed line, a duplicate path, or a path that is absolute or leaves the folder."""
    entries: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), 1):
        digest, separator, relative = line.partition("  ")
        if len(digest) != 64 or separator != "  " or not relative or any(c not in "0123456789abcdef" for c in digest):
            raise ManifestError(f"Manifest line {number} is not '<64 hex characters>  <path>': {line[:80]!r}. The file is damaged or not a manifest; make it again with tools/make_manifest.py.")
        parts = relative.split("/")
        if relative.startswith("/") or ":" in relative or ".." in parts or "\\" in relative:
            raise ManifestError(f"Manifest line {number} has an unsafe path {relative!r} (absolute, drive or '..'). Nothing is read. Make the manifest again with tools/make_manifest.py.")
        if relative in entries:
            raise ManifestError(f"Manifest line {number} lists {relative!r} twice. Make the manifest again with tools/make_manifest.py.")
        entries[relative] = digest
    return entries


def verify_manifest(root: Path, manifest: Path, excludes: Sequence[str] = ()) -> VerifyReport:
    """Compare the tree under `root` with `manifest`. Errors: ManifestError when the manifest is missing, unreadable or malformed."""
    if not manifest.is_file():
        raise ManifestError(f"The manifest {manifest} does not exist. Make it with: python tools/make_manifest.py --root {root} --out {manifest}")
    try:
        entries = parse_manifest(manifest.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise ManifestError(f"The manifest {manifest} is not UTF-8 text ({exc.reason}). Make it again with tools/make_manifest.py.") from exc
    changed, missing, matching = [], [], 0
    for relative, digest in sorted(entries.items()):
        target = root / relative
        if not target.is_file():
            missing.append(relative)
        elif file_sha256(target) != digest:
            changed.append(relative)
        else:
            matching += 1
    extra = [r for r in collect_files(root, excludes, [manifest]) if r not in entries]
    return VerifyReport(matching, tuple(changed), tuple(missing), tuple(extra))


def write_manifest(root: Path, out: Path, excludes: Sequence[str] = ()) -> tuple[int, str]:
    """Write the manifest of `root` to `out` through a temporary file and a rename. Returns (the number of files, the first 8 characters of the manifest's own hash)."""
    text = build_manifest(root, excludes, [out])
    temporary = out.with_name(out.name + ".tmp")
    temporary.write_bytes(text.encode("utf-8"))  # bytes: the newlines stay "\n" on every machine
    temporary.replace(out)
    return text.count("\n"), hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry. Returns the exit code."""
    parser = argparse.ArgumentParser(description="Write or verify MANIFEST.sha256 (the SHA-256 of every file of a release).")
    parser.add_argument("--root", type=Path, default=Path("."), help="the release folder (default: the current folder)")
    parser.add_argument("--out", type=Path, default=None, help=f"the manifest to write (default: <root>/{MANIFEST_NAME})")
    parser.add_argument("--verify", nargs="?", const=True, default=None, type=Path, help=f"verify the tree against this manifest instead of writing one (default file: <root>/{MANIFEST_NAME})")
    parser.add_argument("--exclude", action="append", default=[], help="an extra file pattern to leave out, matched against the path relative to --root (repeatable)")
    args = parser.parse_args(argv)
    root: Path = args.root
    try:
        if not root.is_dir():
            raise ManifestError(f"The folder {root} does not exist. Pass the release folder with --root.")
        if args.verify is not None:
            manifest = root / MANIFEST_NAME if args.verify is True else args.verify
            report = verify_manifest(root, manifest, args.exclude)
            for name in report.changed:
                print(f"CHANGED  {name}")
            for name in report.missing:
                print(f"MISSING  {name}")
            for name in report.extra:
                print(f"EXTRA    {name}  (not in the manifest; not a failure)")
            if report.ok:
                print(f"VERIFIED: {report.matching} files match {manifest}" + (f"; {len(report.extra)} extra file(s) not in the manifest." if report.extra else "."))
                return 0
            print(f"VERIFY FAILED: {len(report.changed)} changed, {len(report.missing)} missing ({report.matching} files match). Copy the release again from home.")
            return 1
        out = args.out if args.out is not None else root / MANIFEST_NAME
        count, short = write_manifest(root, out, args.exclude)
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {out}: {count} files, manifest hash {short}.")
    print(f"Next: copy the WHOLE folder (the manifest included) to the closed machine; `python -m stt check` finds it and prints the code version with +{short}.")
    print(f"Verify any time with: python tools/make_manifest.py --root {root} --verify {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
