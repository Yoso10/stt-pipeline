"""Download a dataset (Parquet files) from the Hugging Face Hub into a local folder (home-side).

How to run (from the project root, with the project virtual environment active):
    1. See the size first, nothing is downloaded:
       python tools/fetch_data.py --repo portal-daf-yomi/daf-yomi-talmud-whisper-training --dry-run
    2. Download:
       python tools/fetch_data.py --repo portal-daf-yomi/daf-yomi-talmud-whisper-training --out data/daf_yomi_raw
Example output of --dry-run:
    portal-daf-yomi/...: 9 files, 812.4 MB (parquet: 8 files, 812.3 MB)
Running the download again reuses the finished folder. Exit code 0 = success.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    Downloader,
    FetchError,
    FetchResult,
    fetch_repo,
    hub_dataset_files,
    hub_downloader,
)

DATA_PATTERNS = ["*.parquet", "*.json", "*.md", "*.csv"]

Lister = Callable[[str], list[tuple[str, int]]]


def missing_data_files(folder: Path) -> list[str]:
    """Return what is missing from `folder`: at least one .parquet file must exist."""
    if not any(folder.rglob("*.parquet")):
        return ["at least one .parquet file"]
    return []


def fetch_data(repo: str, out: Path, downloader: Downloader = hub_downloader) -> FetchResult:
    """Download dataset `repo` into `out`; return FetchResult. Raises FetchError on failure."""
    return fetch_repo(repo, "dataset", out, DATA_PATTERNS, downloader, missing_data_files)


def dry_run_summary(repo: str, lister: Lister = hub_dataset_files) -> str:
    """Return a one-line size summary of what fetch_data would download. Downloads nothing."""
    files = lister(repo)
    wanted = [(n, s) for n, s in files if any(Path(n).match(p) for p in DATA_PATTERNS)]
    parquet = [(n, s) for n, s in wanted if n.endswith(".parquet")]
    total = sum(s for _, s in wanted) / 1e6
    pq_total = sum(s for _, s in parquet) / 1e6
    return (
        f"{repo}: {len(wanted)} files, {total:.1f} MB "
        f"(parquet: {len(parquet)} files, {pq_total:.1f} MB)"
    )


def main(
    argv: list[str] | None = None,
    downloader: Downloader = hub_downloader,
    lister: Lister = hub_dataset_files,
) -> int:
    """Command-line entry. Returns the exit code. `downloader` and `lister` are replaced in tests."""
    parser = argparse.ArgumentParser(description="Download a Hub dataset into a local folder.")
    parser.add_argument("--repo", required=True, help="Hub dataset name")
    parser.add_argument("--out", type=Path, help="destination folder (not needed with --dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="only print the download size")
    args = parser.parse_args(argv)
    try:
        if args.dry_run:
            print(dry_run_summary(args.repo, lister))
            return 0
        if args.out is None:
            parser.error("--out is required unless --dry-run is given")
        result = fetch_data(args.repo, args.out, downloader)
    except FetchError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    verb = "Reused (already complete)" if result.reused else "Created"
    n = len(result.info["files"])  # type: ignore[arg-type]
    mb = int(str(result.info["total_bytes"])) / 1e6
    print(f"{verb}: {result.out}  ({n} files, {mb:.1f} MB)")
    print(f"Next: set  data.source.path: {result.out.as_posix()}  in your config YAML.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
