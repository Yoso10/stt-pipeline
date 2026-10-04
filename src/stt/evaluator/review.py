"""The human review list: the worst files and random files, each with a word-level difference (standard library only).

How to run: this module is not a script; the Evaluator calls it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.evaluator.review import word_diff; print(word_diff('שלום עולם טוב', 'שלום עולם רע'))"
Example output:
    שלום עולם [טוב->רע]
`[a->b]` is a replaced word, `[-a-]` a word the model missed, `[+b+]` a word the model added. Listen to the files in the list
(`audio_path`) to judge the text yourself. Deterministic for a fixed seed.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from pathlib import Path

from stt.evaluator.config import ReviewConfig
from stt.evaluator.types import FileScore, ReviewEntry, ReviewList


def word_diff(reference: str, hypothesis: str) -> str:
    """The word-level difference of two texts as one line."""
    ref, hyp = reference.split(), hypothesis.split()
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=ref, b=hyp, autojunk=False).get_opcodes():
        if tag == "equal":
            parts.extend(ref[i1:i2])
        elif tag == "replace":
            parts.append(f"[{' '.join(ref[i1:i2])}->{' '.join(hyp[j1:j2])}]")
        elif tag == "delete":
            parts.append(f"[-{' '.join(ref[i1:i2])}-]")
        else:
            parts.append(f"[+{' '.join(hyp[j1:j2])}+]")
    return " ".join(parts)


def build_review_list(files: Sequence[FileScore], audio_paths: Mapping[str, Path], config: ReviewConfig, seed: int) -> ReviewList:
    """The `n_worst` files with the highest WER (ties by id) followed by `n_random` seeded random files among the others."""
    ordered = sorted(files, key=lambda f: (-f.wer, f.sample_id))
    worst = ordered[: config.n_worst]
    rest = sorted(ordered[config.n_worst :], key=lambda f: f.sample_id)
    random_files = random.Random(seed).sample(rest, min(config.n_random, len(rest)))
    entries = [(f, "worst") for f in worst] + [(f, "random") for f in random_files]
    return ReviewList(
        tuple(
            ReviewEntry(f.sample_id, f.group_id, audio_paths[f.sample_id], f.reference, f.hypothesis, f.wer, word_diff(f.reference, f.hypothesis), reason)  # type: ignore[arg-type]
            for f, reason in entries
        )
    )
