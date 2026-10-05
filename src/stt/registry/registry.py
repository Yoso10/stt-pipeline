"""ModelRegistry: numbered, immutable entries on the file system with a card, checksums and an index (C10).

How to run: from Python (PowerShell, project root; the full working example is in docs/COMPONENTS.md, C10):
    $env:PYTHONPATH = "src"
    python -c "from stt.registry import ModelRegistry, RegistryConfig; print(ModelRegistry(RegistryConfig()).list_entries())"
Example output (an empty registry):
    []
Layout of the registry folder (ONE self-contained folder: copy it whole to move it, for example to a removable drive):
    registry/index.csv                 one row per entry
    registry/model_0001/model/         the winner's model folder (absent for a baseline_retained entry)
    registry/model_0001/model_card.json, MODEL_CARD.md, review_list.txt, SHA256SUMS
Registration is crash-safe: everything is built in a temporary folder `.tmp_model_NNNN` that reserves the number, the checksums are written and verified, the folder is
renamed into place at the end, the index is updated, and only then the other finalists and the checkpoints named in the request are deleted. A failure at any step puts
the model folder back where it was and removes the temporary folder: the registry is unchanged. Entries are never overwritten and never edited. No network, no
model loading, no evaluation, no torch or Hugging Face import.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import stat
from dataclasses import replace
from pathlib import Path

from stt.registry import hashing
from stt.registry.card import card_from_json, card_to_json, render_markdown
from stt.registry.config import RegistryConfig
from stt.registry.errors import RegistryError, RegistryVerifyError
from stt.registry.index import IndexStore, row_from_card
from stt.registry.types import ModelCard, RegisterRequest, RegistryEntry, VerifyReport

logger = logging.getLogger("stt.registry")
ENTRY = re.compile(r"^model_(\d{4,})$")
TEMP_PREFIX = ".tmp_"
CARD_JSON = "model_card.json"
CARD_MD = "MODEL_CARD.md"
REVIEW = "review_list.txt"
MODEL_FOLDER = "model"


def _writable_and_retry(function, path, exc):  # type: ignore[no-untyped-def]
    os.chmod(path, stat.S_IWRITE)  # model files may be read-only; a deliberate delete must not be blocked
    function(path)


def _remove_tree(path: Path) -> None:
    shutil.rmtree(path, onexc=_writable_and_retry)


def _same_drive(a: Path, b: Path) -> bool:
    return a.resolve().drive == b.resolve().drive and a.stat().st_dev == b.stat().st_dev


class ModelRegistry:
    """The registry of one folder (`registry.dir`)."""

    def __init__(self, config: RegistryConfig, index: IndexStore | None = None) -> None:
        self._config = config
        self._root = config.dir
        self._index = index or IndexStore()

    # ---- reading -------------------------------------------------------------------------------------------------
    def _entry_ids(self) -> list[str]:
        if not self._root.is_dir():
            return []
        return sorted(p.name for p in self._root.iterdir() if p.is_dir() and ENTRY.match(p.name))

    def _read_card(self, entry_id: str) -> tuple[str, ModelCard]:
        path = self._root / entry_id / CARD_JSON
        try:
            _, experiment_id, card = card_from_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RegistryError(f"The card {path} cannot be read ({type(exc).__name__}: {exc}). The entry is damaged: copy it again from the source registry.") from exc
        return experiment_id, card

    def _entry(self, entry_id: str) -> RegistryEntry:
        folder = self._root / entry_id
        if not folder.is_dir():
            known = ", ".join(self._entry_ids()) or "none yet"
            raise RegistryError(f"The registry entry '{entry_id}' does not exist in {self._root}. Existing entries: {known}.")
        experiment_id, card = self._read_card(entry_id)
        sums = folder / hashing.SUMS_NAME
        return RegistryEntry(entry_id, experiment_id, card.kind, folder / MODEL_FOLDER if card.kind == "model" else None, folder / CARD_JSON, sums if sums.is_file() else None, card.test_evaluations)

    def get(self, entry_id: str) -> RegistryEntry:
        """The entry `entry_id`. Errors: RegistryError naming the existing entries, or the damaged card."""
        return self._entry(entry_id)

    def list_entries(self) -> list[RegistryEntry]:
        """All entries, lowest number first (read from the cards, which are the source of truth)."""
        return [self._entry(entry_id) for entry_id in self._entry_ids()]

    def card(self, entry_id: str) -> ModelCard:
        """The card of an entry."""
        return self._read_card(entry_id)[1]

    def test_set_count(self, test_fingerprint: str) -> int:
        """How many registered entries were evaluated on this Test set (a lower bound: only registered entries are counted)."""
        return sum(1 for entry_id in self._entry_ids() if self._read_card(entry_id)[1].test_fingerprint == test_fingerprint)

    # ---- the index -----------------------------------------------------------------------------------------------------
    def _validated_rows(self) -> list[dict[str, str]]:
        ids = self._entry_ids()
        if not ids and not self._index.path(self._root).exists():
            return []
        rows = self._index.read(self._root)
        indexed = [r["entry_id"] for r in rows]
        if sorted(indexed) != ids:
            raise RegistryError(
                f"The registry index does not match the entries: the index lists {len(indexed)} ({', '.join(indexed) or 'none'}), the folder holds {len(ids)} ({', '.join(ids) or 'none'}). "
                "Run ModelRegistry.rebuild_index() to make the index again from the cards; it is never rebuilt silently."
            )
        return rows

    def rebuild_index(self) -> None:
        """Make `index.csv` again from the entry cards (the cards are the source of truth)."""
        self._root.mkdir(parents=True, exist_ok=True)
        rows = []
        for entry_id in self._entry_ids():
            experiment_id, card = self._read_card(entry_id)
            rows.append(row_from_card(card, entry_id, experiment_id))
        self._index.write(self._root, rows)

    def best(self, eval_version: int, test_fingerprint: str) -> RegistryEntry | None:
        """The entry of kind `model` with the lowest Val WER among the entries with this `eval_version` and this Test-set fingerprint (other comparisons are invalid).

        Returns None when none matches. Errors: RegistryError when the index is missing or does not match the entries (the fix is `rebuild_index()`).
        """
        rows = [r for r in self._validated_rows() if r["kind"] == "model" and r["eval_version"] == str(eval_version) and r["test_fingerprint"] == test_fingerprint and r["val_wer"]]
        if not rows:
            return None
        best = min(rows, key=lambda r: (float(r["val_wer"]), r["entry_id"]))
        return self._entry(best["entry_id"])

    # ---- verify ----------------------------------------------------------------------------------------------------------
    def verify(self, entry_id: str) -> VerifyReport:
        """Check the files of an entry against its SHA256SUMS. Returns the report when everything is fine.

        Errors: RegistryVerifyError (exit code 5) listing the missing and the changed files; RegistryError when the entry has no checksums or the file is damaged.
        """
        folder = self._root / entry_id
        if not folder.is_dir():
            raise RegistryError(f"The registry entry '{entry_id}' does not exist in {self._root}.")
        if not (folder / hashing.SUMS_NAME).is_file():
            raise RegistryError(f"The entry {entry_id} has no {hashing.SUMS_NAME} (it was registered with registry.checksums false), so it cannot be verified.")
        try:
            report = hashing.check_sums(folder)
        except ValueError as exc:
            raise RegistryError(f"The {hashing.SUMS_NAME} of {entry_id} is damaged: {exc}.") from exc
        if not report.ok:
            raise RegistryVerifyError(
                f"The entry {entry_id} does not match its {hashing.SUMS_NAME}: missing files {list(report.missing)}, changed or unlisted files {list(report.mismatching)}. "
                "Copy the entry again from the source registry.",
                report,
            )
        return report

    # ---- registering -------------------------------------------------------------------------------------------------------
    def register(self, request: RegisterRequest) -> RegistryEntry:
        """Store the selection result as the next entry (see the module docstring for the crash-safe order).

        Errors: RegistryError (exit code 4) when the request does not fit the selection, the winner folder is missing or empty, a finalist folder contains or lies inside
        the registry, the index is missing or damaged, a leftover temporary folder exists, or anything fails while registering (the registry is left unchanged).
        """
        sel, card = request.selection, request.card
        winner = sel.winner
        if (winner is None) != (card.kind == "baseline_retained") or card.verdict != sel.verdict:
            raise RegistryError(f"The card (kind {card.kind}, verdict {card.verdict}) does not match the selection (verdict {sel.verdict}).")
        if winner is not None and card.eval_version != winner.val.eval_version:
            raise RegistryError(f"The card says eval_version {card.eval_version} but the winner was evaluated with {winner.val.eval_version}.")
        winner_dir = winner.model_dir if winner is not None else None
        if winner_dir is not None and (not winner_dir.is_dir() or not any(winner_dir.iterdir())):
            raise RegistryError(f"The winner's model folder {winner_dir} is missing or empty, so it cannot be registered. Check the finalist retraining.")
        self._check_finalists(request)
        self._root.mkdir(parents=True, exist_ok=True)
        self._validated_rows()  # a damaged or stale index fails here, before anything is created
        leftovers = sorted(p.name for p in self._root.iterdir() if p.name.startswith(TEMP_PREFIX))
        if leftovers:
            raise RegistryError(
                f"A leftover temporary folder from an interrupted registration exists: {self._root / leftovers[0]}. It may hold a model folder: look inside, move any 'model' folder back "
                "to where it came from, then delete the temporary folder. Nothing was changed."
            )
        number = max([int(ENTRY.match(i).group(1)) for i in self._entry_ids() if ENTRY.match(i)] or [0]) + 1  # type: ignore[union-attr]
        entry_id = f"model_{number:04d}"
        temp = self._root / f"{TEMP_PREFIX}{entry_id}"
        final = self._root / entry_id
        if final.exists():
            raise RegistryError(f"The entry {final} already exists; entries are never overwritten.")
        try:
            temp.mkdir()  # exclusive: the number is reserved
        except OSError as exc:
            raise RegistryError(f"Cannot create {temp} ({exc}). Check that the folder is writable and the disk is not full.") from exc
        state = {"moved_from": None, "renamed": False}
        try:
            entry = self._build(request, entry_id, temp, final, winner_dir, state)
        except BaseException:
            self._rollback(temp, final, state)
            raise
        if self._config.transfer == "move" and winner_dir is not None and winner_dir.exists():
            try:
                _remove_tree(winner_dir)  # the source of a move across drives (the model is safe in the entry, verified)
            except OSError as exc:
                logger.warning("Could not delete the source %s after the move (%s); the entry is complete.", winner_dir, exc)
        self._cleanup(request, winner_dir)
        return entry

    def _check_finalists(self, request: RegisterRequest) -> None:
        root = self._root.resolve()
        for d in request.finalist_dirs:
            r = d.resolve()
            if r == root or root in r.parents or r in root.parents:
                raise RegistryError(f"The finalist folder {d} contains the registry {self._root} or lies inside it. Nothing was changed: the finalists must be outside the registry.")

    def _build(self, request: RegisterRequest, entry_id: str, temp: Path, final: Path, winner_dir: Path | None, state: dict[str, object]) -> RegistryEntry:
        card = request.card
        evaluated = self.test_set_count(card.test_fingerprint) + 1 if card.test_fingerprint is not None else 0
        counted = replace(card, test_evaluations=evaluated)
        if winner_dir is not None:
            target = temp / MODEL_FOLDER
            if self._config.transfer == "move" and _same_drive(winner_dir, temp):
                winner_dir.rename(target)  # atomic on one drive; undone by the rollback
                state["moved_from"] = winner_dir
            else:
                self._copy_verified(winner_dir, target)  # a move across drives is a verified copy; the source is deleted only after the whole registration succeeded
        has_review = request.review_text is not None
        (temp / CARD_JSON).write_text(card_to_json(counted, entry_id, request.experiment_id), encoding="utf-8", newline="\n")
        (temp / CARD_MD).write_text(render_markdown(counted, entry_id, request.experiment_id, evaluated, has_review, self._config.checksums), encoding="utf-8", newline="\n")
        if request.review_text is not None:
            (temp / REVIEW).write_text(request.review_text, encoding="utf-8", newline="\n")
        if self._config.checksums:
            hashing.write_sums(temp)
        rows = self._validated_rows()  # read BEFORE the new entry exists, so the index and the folder are compared as they were
        temp.rename(final)
        state["renamed"] = True
        if self._config.checksums:
            report = hashing.check_sums(final)  # verified in its final place BEFORE anything is deleted
            if not report.ok:
                raise RegistryVerifyError(f"The new entry {entry_id} does not match its {hashing.SUMS_NAME} right after it was written: {list(report.missing) + list(report.mismatching)}.", report)
        rows.append(row_from_card(counted, entry_id, request.experiment_id))
        self._index.write(self._root, rows)
        sums = final / hashing.SUMS_NAME
        return RegistryEntry(entry_id, request.experiment_id, card.kind, final / MODEL_FOLDER if winner_dir is not None else None, final / CARD_JSON, sums if sums.is_file() else None, evaluated)

    @staticmethod
    def _copy_verified(src: Path, dst: Path) -> None:
        shutil.copytree(src, dst)
        for rel in hashing.list_files(src, exclude=()):
            if hashing.sha256_file(src / rel) != hashing.sha256_file(dst / rel):
                raise RegistryError(f"The copy of {src / rel} is not identical to the original (a disk or drive problem). Nothing was registered; the original is untouched.")

    def _rollback(self, temp: Path, final: Path, state: dict[str, object]) -> None:
        """Undo a failed registration: the entry folder back to the temporary name, the model folder back to its place, the temporary folder removed."""
        try:
            if state.get("renamed") and final.exists():
                final.rename(temp)
            source = state.get("moved_from")
            if isinstance(source, Path) and (temp / MODEL_FOLDER).exists() and not source.exists():
                (temp / MODEL_FOLDER).rename(source)
            if temp.exists():
                _remove_tree(temp)
        except OSError as exc:  # the message names the folder: nothing is deleted that still holds the model
            logger.error("The rollback of the registration could not finish (%s). Look inside %s: it may still hold the model folder.", exc, temp)

    def _cleanup(self, request: RegisterRequest, winner_dir: Path | None) -> None:
        """After a successful registration: delete the other finalists and the checkpoints named in the request (never anything else)."""
        if not self._config.cleanup or self._config.keep_finalists:
            return
        winner_trial = winner_dir.parent.resolve() if winner_dir is not None else None
        for d in request.finalist_dirs:
            try:
                if winner_trial is not None and d.resolve() == winner_trial:
                    if (d / "checkpoints").is_dir():
                        _remove_tree(d / "checkpoints")
                elif d.is_dir():
                    _remove_tree(d)
            except OSError as exc:
                logger.warning("Could not delete %s after the registration (%s); the entry is complete, delete it by hand.", d, exc)
