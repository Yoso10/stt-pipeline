"""The `cleanup` command: PROPOSE what could be deleted to free disk space. It never deletes anything (DECISIONS 56 and 73).

How to run (project root):
    python -m stt cleanup --config my.yaml                       # the proposal is written next to the experiments folder
    python -m stt cleanup --older-than-days 30 --out proposal.txt
    python -m stt cleanup --include-failed                       # also list failed experiments (they are NOT listed by default)
Example output (stderr, then one JSON on stdout):
    Proposal written to: C:\\work\\cleanup_proposal_20261007T201600.txt
    Nothing was deleted. Checkpoint folders: 3 (1.8 GB); candidate folders: 7 (2.9 GB, includes the checkpoints); not proposed: 2 experiments.
What it proposes (the folders; all of them are rebuilt by running the experiment again):
    A. the `checkpoints` folders inside the candidates of COMPLETED experiments (the models stay);
    B. the whole candidate folders (trial and finalist models, with their checkpoints) of COMPLETED experiments (this includes A, so the sizes are not added);
    C. only with `--include-failed`: the whole folder of FAILED experiments.
Rules (owner, LOCKED): it changes nothing except writing its own proposal file (a file that exists is never overwritten); it never proposes anything inside the model registry or a folder that
contains it; it never proposes anything of an experiment that is not completed (created, running, stopped, unreadable) or that has a lock file (a run is active or crashed). The C12 RunManager
supplies the candidate folders (`ExperimentStore.propose_cleanup`); this module only filters them, measures them and writes the text. There is no flag that deletes: the owner deletes by hand.
Errors: OrchestratorError (exit code 2) for a missing experiments folder, an unreadable `--older-than-days`, or an existing output file.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from stt.orchestrator.errors import OrchestratorError
from stt.runmanager import CleanupPolicy, ExperimentSummary

CHECKPOINTS = "checkpoints"
CANDIDATES = "candidates"
FAILED = "failed"
REASONS = {
    CHECKPOINTS: "checkpoint folder of a completed experiment; the model next to it stays; it is needed only to resume that training",
    CANDIDATES: "trial or finalist model folder of a completed experiment (with its checkpoints); the numbers stay in the experiment's reports; it is needed only to use that exact model again",
    FAILED: "the whole folder of a FAILED experiment; a rerun with --experiment would resume it, so keep it if you may want that",
}


class CleanupCatalog(Protocol):
    """The experiments folder as `cleanup` needs it (the real `ExperimentStore` fits; `wiring.experiment_catalog` makes it)."""

    def list_experiments(self) -> list[ExperimentSummary]:
        """All experiments, lowest number first."""
        ...

    def propose_cleanup(self, policy: CleanupPolicy) -> list[Path]:
        """The folders the policy would remove. Changes nothing."""
        ...


@dataclass(frozen=True)
class CleanupItem:
    """One folder that could be deleted: which experiment, which kind, where, how big, and why."""

    experiment_id: str
    kind: str
    path: Path
    size_bytes: int
    reason: str


@dataclass(frozen=True)
class CleanupSkip:
    """Something that was NOT proposed and the reason (so that the owner can see the protections work)."""

    what: str
    why: str


@dataclass(frozen=True)
class CleanupProposal:
    """The result: the three lists, what was left out, and the settings that made it. `deleted` is always 0."""

    experiments_dir: Path
    made_at: str
    older_than_days: int | None
    include_failed: bool
    checkpoints: tuple[CleanupItem, ...]
    candidates: tuple[CleanupItem, ...]
    failed: tuple[CleanupItem, ...]
    skipped: tuple[CleanupSkip, ...]
    deleted: int = 0


def folder_size(path: Path) -> int:
    """The total size in bytes of the files under `path` (symbolic links are not followed). Reads only."""
    total = 0
    for folder, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(folder) / name).lstat().st_size
            except OSError:
                continue  # a file that vanished or cannot be read does not stop the proposal
    return total


def _inside(path: Path, other: Path) -> bool:
    """True when `path` is `other` or lies inside it."""
    return path == other or other in path.parents


def propose_cleanup(
    catalog: CleanupCatalog, experiments_dir: Path, registry_dir: Path | None, now: str, older_than_days: int | None = None, include_failed: bool = False
) -> CleanupProposal:
    """What could be deleted, with sizes and reasons. Changes nothing.

    Errors: OrchestratorError (exit code 2) when `experiments_dir` is not a folder or `older_than_days` is negative; RunManagerError (exit code 2) when `now` is not an ISO date.
    """
    if not experiments_dir.is_dir():
        raise OrchestratorError(f"The experiments folder {experiments_dir} does not exist. Point run.experiments_dir at your experiments folder (--set run.experiments_dir=...).", exit_code=2)
    if older_than_days is not None and older_than_days < 0:
        raise OrchestratorError(f"--older-than-days must be 0 or more, got {older_than_days}.", exit_code=2)
    base = experiments_dir.resolve()
    registry = registry_dir.resolve() if registry_dir is not None else None
    skipped: list[CleanupSkip] = []
    eligible_completed: set[str] = set()
    eligible_failed: set[str] = set()
    for summary in catalog.list_experiments():
        root = base / summary.experiment_id
        if (root / "run.lock").exists():
            skipped.append(CleanupSkip(summary.experiment_id, "it has a lock file (a run is active or crashed); never proposed"))
        elif registry is not None and _inside(registry, root):
            skipped.append(CleanupSkip(summary.experiment_id, f"the model registry {registry_dir} lies inside it; never proposed"))
        elif summary.status == "completed":
            eligible_completed.add(summary.experiment_id)
        elif summary.status == "failed" and include_failed:
            eligible_failed.add(summary.experiment_id)
        elif summary.status == "failed":
            skipped.append(CleanupSkip(summary.experiment_id, "its status is failed; only listed with --include-failed"))
        else:
            skipped.append(CleanupSkip(summary.experiment_id, f"its status is {summary.status}, not completed: it is not finished; never proposed"))

    def collect(policy: CleanupPolicy, kind: str, allowed: set[str]) -> tuple[CleanupItem, ...]:
        items = []
        for path in catalog.propose_cleanup(policy):
            resolved = path.resolve()
            experiment_id = resolved.relative_to(base).parts[0]
            if experiment_id not in allowed:
                continue
            if registry is not None and (_inside(resolved, registry) or _inside(registry, resolved)):
                skipped.append(CleanupSkip(str(path), "it is, contains or lies inside the model registry; never proposed"))
                continue
            items.append(CleanupItem(experiment_id, kind, path, folder_size(path), REASONS[kind]))
        return tuple(sorted(items, key=lambda i: (i.experiment_id, str(i.path))))

    # two policies, because the C12 `candidates` kind already covers the checkpoints inside the candidate folders
    checkpoints = collect(CleanupPolicy(checkpoints_of_completed=True, older_than_days=older_than_days, now=now), CHECKPOINTS, eligible_completed)
    candidates = collect(CleanupPolicy(candidates_of_completed=True, older_than_days=older_than_days, now=now), CANDIDATES, eligible_completed)
    failed = collect(CleanupPolicy(failed_experiments=True, older_than_days=older_than_days, now=now), FAILED, eligible_failed) if include_failed else ()
    if older_than_days is not None:
        proposed = {i.experiment_id for i in (*checkpoints, *candidates, *failed)}
        for experiment_id in sorted((eligible_completed | eligible_failed) - proposed):
            skipped.append(CleanupSkip(experiment_id, f"nothing to propose: it is younger than {older_than_days} days, its creation date is unreadable, or it has no candidate folders"))
    return CleanupProposal(experiments_dir, now, older_than_days, include_failed, checkpoints, candidates, failed, tuple(skipped))


def megabytes(size: int) -> str:
    """A size for people: `812 KB`, `3.4 MB` or `1.8 GB`."""
    if size >= 1_000_000_000:
        return f"{size / 1e9:.1f} GB"
    if size >= 1_000_000:
        return f"{size / 1e6:.1f} MB"
    return f"{max(size, 0) // 1000} KB"


def total_size(items: Sequence[CleanupItem]) -> int:
    """The sum of the sizes of `items`."""
    return sum(i.size_bytes for i in items)


def render_proposal(proposal: CleanupProposal) -> str:
    """The text of the proposal file (plain text; it starts with the How to run block)."""
    lines = [
        "HOW TO RUN: python -m stt cleanup [--config FILE] [--older-than-days N] [--include-failed] [--out FILE]   (it only proposes and writes this file)",
        "THIS FILE IS A PROPOSAL. NOTHING WAS DELETED, and the command has no option that deletes. To free the space, read the list and delete the folders you agree with yourself.",
        "Nothing inside the model registry and nothing of an experiment that is not completed or is locked is ever listed.",
        "",
        f"Experiments folder: {proposal.experiments_dir}",
        f"Made: {proposal.made_at}   Older than: " + ("no age filter" if proposal.older_than_days is None else f"{proposal.older_than_days} days") + "   Failed experiments: " + ("included" if proposal.include_failed else "not included (use --include-failed)"),
        "",
    ]

    def section(title: str, note: str, kind: str, items: Sequence[CleanupItem]) -> None:
        lines.append(f"{title}: {len(items)} folder(s), {megabytes(total_size(items))}")
        lines.append(f"  {note}")
        lines.append(f"  Why: {REASONS[kind]}.")
        for item in items:
            lines.append(f"  {megabytes(item.size_bytes):>9}  {item.path}")
        if not items:
            lines.append("  (nothing)")
        lines.append("")

    section("A. CHECKPOINT FOLDERS of completed experiments", "Smallest step: the trained models stay, only the files for resuming a training go.", CHECKPOINTS, proposal.checkpoints)
    section("B. WHOLE CANDIDATE FOLDERS of completed experiments", "Bigger step: the trial and finalist models go as well. B INCLUDES A: do not add the two totals.", CANDIDATES, proposal.candidates)
    if proposal.include_failed:
        section("C. FAILED EXPERIMENTS (whole folders)", "Only listed because --include-failed was given.", FAILED, proposal.failed)
    lines.append(f"NOT PROPOSED: {len(proposal.skipped)}")
    lines.extend(f"  {skip.what}: {skip.why}" for skip in proposal.skipped)
    if not proposal.skipped:
        lines.append("  (nothing was left out)")
    lines.append("")
    lines.append("Deleted by this command: 0")
    return "\n".join(lines) + "\n"


def write_proposal(text: str, target: Path) -> Path:
    """Write the proposal to `target` (the only thing `cleanup` writes). A file that exists is NEVER overwritten.

    Errors: OrchestratorError (exit code 2) when `target` exists or its folder does not exist.
    """
    if target.exists():
        raise OrchestratorError(f"The proposal file {target} already exists and is not overwritten. Choose another name with --out, or move the old file away yourself.", exit_code=2)
    if not target.parent.is_dir():
        raise OrchestratorError(f"The folder of the proposal file {target} does not exist. Create it, or choose another --out.", exit_code=2)
    with target.open("x", encoding="utf-8", newline="\n") as handle:  # "x": fails instead of overwriting, even in a race
        handle.write(text)
    return target
