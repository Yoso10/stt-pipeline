"""The WER difference of two transcriptions of the same files (pure code; for exporters that verify at the `transcribe` level).

How to run: this module is not a script; an exporter's `verify` uses it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.modelexporter.compare import wer_difference; print(wer_difference(['a b c d'], ['a b c x']))"
Example output:
    0.25
The expected texts are the transcripts of the REGISTERED model; `actual` are those of the exported model. The result is the corpus WER of `actual` against `expected` (all words together), the word
counting of the Evaluator (`stt.evaluator.metrics`): this component computes no WER of its own and calls no model.
"""

from __future__ import annotations

from collections.abc import Sequence

from stt.evaluator.metrics import WordErrorRate


def wer_difference(expected: Sequence[str], actual: Sequence[str]) -> float:
    """The corpus WER of `actual` against `expected`. Errors: ValueError for lists of different length; 0.0 when there are no words."""
    if len(expected) != len(actual):
        raise ValueError(f"{len(expected)} expected texts but {len(actual)} transcriptions: the check needs one transcription per file.")
    metric = WordErrorRate()
    errors = words = 0
    for reference, hypothesis in zip(expected, actual, strict=True):
        score = metric.score(reference, hypothesis)
        errors, words = errors + score.errors, words + score.units
    return errors / words if words else 0.0
