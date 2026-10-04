"""Our own error-rate metrics: word error rate and character error rate (pure Python, no jiwer, no evaluate.load).

How to run: this module is not a script; the Evaluator uses it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.evaluator.metrics import WordErrorRate; print(WordErrorRate().score('שלום עולם טוב', 'שלום עולם'))"
Example output:
    MetricScore(errors=1, units=3)
WER = word edit distance (substitutions + deletions + insertions) divided by the number of reference words. CER is the same on
characters; spaces count as characters. The corpus value of a metric is the sum of the errors divided by the sum of the units (all
words together), computed by the Evaluator.
"""

from __future__ import annotations

from collections.abc import Sequence

from stt.evaluator.types import MetricScore


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Levenshtein distance between two sequences (substitution, deletion and insertion cost 1)."""
    start = 0
    while start < len(a) and start < len(b) and a[start] == b[start]:  # a shared start and end cost nothing: trim them
        start += 1
    end_a, end_b = len(a), len(b)
    while end_a > start and end_b > start and a[end_a - 1] == b[end_b - 1]:
        end_a -= 1
        end_b -= 1
    a, b = a[start:end_a], b[start:end_b]
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


class WordErrorRate:
    """`wer`: errors over the words of the reference (words are separated by whitespace)."""

    name = "wer"

    def score(self, reference: str, hypothesis: str) -> MetricScore:
        """Word edit distance and the number of reference words."""
        ref = reference.split()
        return MetricScore(edit_distance(ref, hypothesis.split()), len(ref))


class CharErrorRate:
    """`cer`: errors over the characters of the reference (spaces included)."""

    name = "cer"

    def score(self, reference: str, hypothesis: str) -> MetricScore:
        """Character edit distance and the number of reference characters."""
        return MetricScore(edit_distance(reference, hypothesis), len(reference))
