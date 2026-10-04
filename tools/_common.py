"""Shared helpers of the home-side tools (crash-safe folder download, error types).

How to run: this module is not a script. It is imported by tools/fetch_model.py and
tools/fetch_data.py. Home-side only: the pipeline never imports anything from tools/.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

INFO_FILE = "fetch_info.json"

# A downloader copies the repository `repo` (of type `repo_type`) into the folder `target`,
# restricted to files that match `allow_patterns`. The real one wraps huggingface_hub;
# tests pass a fake.
Downloader = Callable[[str, str, Path, list[str]], None]


class FetchError(Exception):
    """A tool failed for a reason the owner can act on. The message says what and how to fix."""


class RepoNotFoundError(FetchError):
    """The repository does not exist or is not public."""


class RepoNotPublicError(FetchError):
    """The repository exists but needs a login (gated or private)."""


@dataclass(frozen=True)
class FetchResult:
    """What a fetch produced. `reused` is True when a complete earlier download was kept."""

    out: Path
    reused: bool
    info: dict[str, object]


def list_files(folder: Path) -> list[dict[str, object]]:
    """Return every file under `folder` (relative posix path and size in bytes), sorted."""
    return [
        {"path": p.relative_to(folder).as_posix(), "bytes": p.stat().st_size}
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name != INFO_FILE
    ]


def read_info(folder: Path) -> dict[str, object] | None:
    """Return the parsed fetch_info.json of `folder`, or None when missing or unreadable."""
    path = folder / INFO_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def fetch_repo(
    repo: str,
    repo_type: str,
    out: Path,
    allow_patterns: list[str],
    downloader: Downloader,
    required: Callable[[Path], list[str]],
) -> FetchResult:
    """Download `repo` into `out` crash-safely and write fetch_info.json.

    Inputs: repo name, repo type ("model" or "dataset"), destination folder, file patterns to
    download, a downloader, and `required(folder)` which returns the list of MISSING required
    files (empty list = complete).
    Output: FetchResult. A complete earlier download of the same repo is reused, never
    overwritten. A folder without fetch_info.json is treated as interrupted and rebuilt.
    Errors: FetchError (or a subclass) with an actionable message; nothing half-written is left
    at `out`.
    """
    existing = read_info(out) if out.exists() else None
    if existing is not None:
        if existing.get("repo") != repo:
            raise FetchError(
                f"{out} already holds '{existing.get('repo')}', not '{repo}'. "
                "Choose another --out folder (nothing was changed)."
            )
        return FetchResult(out=out, reused=True, info=existing)

    partial = out.with_name(out.name + ".partial")
    for stale in (partial, out):  # interrupted earlier run: rebuild
        if stale.exists():
            shutil.rmtree(stale)
    partial.mkdir(parents=True)
    try:
        downloader(repo, repo_type, partial, allow_patterns)
        shutil.rmtree(partial / ".cache", ignore_errors=True)  # Hub bookkeeping, not data
        missing = required(partial)
        if missing:
            raise FetchError(
                f"'{repo}' was downloaded but is missing required files: {', '.join(missing)}. "
                "Check that the repository is the right one; nothing was stored."
            )
        files = list_files(partial)
        info: dict[str, object] = {
            "repo": repo,
            "repo_type": repo_type,
            "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total_bytes": sum(int(str(f["bytes"])) for f in files),
            "files": files,
        }
        (partial / INFO_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")
        partial.rename(out)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return FetchResult(out=out, reused=False, info=info)


def hub_downloader(repo: str, repo_type: str, target: Path, allow_patterns: list[str]) -> None:
    """Real downloader: huggingface_hub.snapshot_download, anonymous, no code executed.

    Raises RepoNotFoundError, RepoNotPublicError or FetchError (network or other failure).
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        snapshot_download(
            repo_id=repo,
            repo_type=repo_type,
            local_dir=str(target),
            allow_patterns=allow_patterns,
            token=False,  # anonymous on purpose: only public items are supported
        )
    except GatedRepoError as exc:
        raise RepoNotPublicError(
            f"'{repo}' requires a login (gated). Not public: obtain the files another way."
        ) from exc
    except RepositoryNotFoundError as exc:
        raise RepoNotFoundError(
            f"'{repo}' ({repo_type}) was not found, or it is private. "
            "Check the exact name on the Hub; do not guess a similar one."
        ) from exc
    except Exception as exc:  # network down, DNS, disk full, ...
        raise FetchError(
            f"Download of '{repo}' failed: {type(exc).__name__}: {exc}. "
            "Check the internet connection and the free disk space, then run again."
        ) from exc


def hub_dataset_files(repo: str) -> list[tuple[str, int]]:
    """Real lister: (path, size in bytes) of every file of a public dataset repo (no download)."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        info = HfApi().dataset_info(repo, files_metadata=True, token=False)
    except GatedRepoError as exc:
        raise RepoNotPublicError(f"'{repo}' requires a login (gated).") from exc
    except RepositoryNotFoundError as exc:
        raise RepoNotFoundError(f"'{repo}' (dataset) was not found, or it is private.") from exc
    except Exception as exc:
        raise FetchError(f"Listing '{repo}' failed: {type(exc).__name__}: {exc}.") from exc
    return [(s.rfilename, int(s.size or 0)) for s in (info.siblings or [])]
