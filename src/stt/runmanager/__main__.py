"""Read-only helper: list the experiments, or show where everything of one experiment is.

How to run (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -m stt.runmanager list --dir experiments
    python -m stt.runmanager show exp_0003 --dir experiments
Example output of `show`:
    experiment exp_0003: status completed, parent exp_0001, created 2026-10-05T12:00:00
    config   : experiments/exp_0003/config_full.yaml  (the difference from the parent: config_diff.yaml)
    status   : experiments/exp_0003/status.json
    log      : experiments/exp_0003/logs/run.log  (last lines below)
    metrics  : experiments/exp_0003/metrics/trial_001/metrics.csv  (last record below)
    reports  : experiments/exp_0003/reports/report.html
It changes nothing. The `stt` command of the Orchestrator will wrap it later (`stt status`).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from stt.runmanager.config import RunConfig
from stt.runmanager.errors import RunManagerError
from stt.runmanager.store import ExperimentStore


def _tail(path: Path, lines: int) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def main(argv: Sequence[str] | None = None) -> int:
    """Run the helper. Returns 0 on success and 2 when the experiment cannot be read."""
    parser = argparse.ArgumentParser(prog="python -m stt.runmanager", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["list", "show"])
    parser.add_argument("experiment", nargs="?", help="for `show`: the experiment id, for example exp_0003")
    parser.add_argument("--dir", type=Path, default=Path("experiments"), help="the experiments folder (run.experiments_dir); default: experiments")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # Hebrew paths and log lines on a Windows console
        sys.stdout.reconfigure(encoding="utf-8")
    store = ExperimentStore(RunConfig(experiments_dir=args.dir))
    try:
        if args.command == "list":
            for s in store.list_experiments():
                print(f"{s.experiment_id}  {s.status:10}  parent {s.parent_id or '-':9}  created {s.created_at or '-'}")
            return 0
        if not args.experiment:
            parser.error("`show` needs an experiment id, for example: show exp_0003")
        handle = store.open(args.experiment)
        status, info, where = handle.status(), handle.run_info(), handle.locations()
    except RunManagerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"experiment {handle.experiment_id}: status {status.status}" + (f" ({status.reason})" if status.reason else "") + f", parent {info.parent_id or '-'}, created {info.created_at}")
    print("stages   : " + (", ".join(f"{n} {m.status}" for n, m in status.stages.items()) or "none finished"))
    print(f"config   : {where.config_full}  (the difference from the parent or the defaults: {where.config_diff.name})")
    print(f"status   : {where.status}")
    print(f"log      : {where.log}  (last lines below)")
    for line in _tail(where.log, 3):
        print("           " + line)
    for m in where.metrics_files:
        print(f"metrics  : {m}  (last record below)")
        for line in _tail(m, 1):
            print("           " + line)
    for r in where.reports:
        print(f"reports  : {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
