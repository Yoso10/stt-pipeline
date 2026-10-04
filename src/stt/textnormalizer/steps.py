"""The normalization steps. Each step is a small class with a `name` and `apply(text) -> text`.

How to run: this module is not a script; the registry builds the steps from the `text` config.
Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.textnormalizer.steps import CollapseWhitespace; print(repr(CollapseWhitespace().apply('  a   b ')))"
Example output:
    'a b'
A new step is a new class here plus one line in the registry (stt.textnormalizer.registry); no
existing code changes. Every step is deterministic and idempotent. No step corrects spelling,
changes words or uses a model.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence


def is_nikud(char: str) -> bool:
    """True for a Hebrew vowel point or cantillation mark (a nonspacing mark in the Hebrew block)."""
    return "֑" <= char <= "ׇ" and unicodedata.category(char) == "Mn"


class UnicodeClean:
    """Unicode normalization (NFC) and removal of the configured invisible and direction-control characters."""

    name = "unicode_clean"

    def __init__(self, invisible_chars: str) -> None:
        self._table = {ord(c): None for c in invisible_chars}

    def apply(self, text: str) -> str:
        """NFC, then delete the invisible characters, then NFC again (deleting can change combining order)."""
        cleaned = unicodedata.normalize("NFC", text).translate(self._table)
        return unicodedata.normalize("NFC", cleaned)


class StripTimestampTokens:
    """Remove Whisper-style timestamp tokens (the pattern comes from Config)."""

    name = "strip_timestamps"

    def __init__(self, pattern: re.Pattern[str]) -> None:
        self._pattern = pattern

    def apply(self, text: str) -> str:
        """Delete every token."""
        return self._pattern.sub("", text)


class RemovePunctuation:
    """Remove the configured punctuation characters; the configured space characters (maqaf) become a space."""

    name = "remove_punctuation"

    def __init__(self, chars: str, space_chars: str) -> None:
        table: dict[int, str | None] = {ord(c): None for c in chars}
        table.update({ord(c): " " for c in space_chars})
        self._table = table

    def apply(self, text: str) -> str:
        """Delete or replace the characters."""
        return text.translate(self._table)


class UnifyQuotes:
    """Map quote and geresh variants to one form (the mapping comes from Config)."""

    name = "unify_quotes"

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self._table = {ord(k): v for k, v in mapping.items()}

    def apply(self, text: str) -> str:
        """Apply the mapping."""
        return text.translate(self._table)


class RemoveNikud:
    """Remove Hebrew vowel points and cantillation marks (nonspacing marks of the Hebrew block).

    The maqaf, paseq and sof pasuq of that block are punctuation, not marks, and are kept.
    """

    name = "remove_nikud"

    def apply(self, text: str) -> str:
        """Delete the marks."""
        return "".join(c for c in text if not is_nikud(c))


class RemoveBracketMarkup:
    """Handle the configured bracket pairs: `chars_only` (remove the characters, keep the content),
    `with_content` (remove the pair and what is inside; nested pairs too) or `keep` (do nothing).

    An unbalanced bracket character is removed and the text kept (not in `keep` mode).
    """

    name = "remove_bracket_markup"

    def __init__(self, pairs: Sequence[tuple[str, str]], mode: str) -> None:
        self._mode = mode
        self._chars = {c for pair in pairs for c in pair}
        self._patterns = [
            re.compile(re.escape(o) + "[^" + re.escape(o) + re.escape(c) + "]*" + re.escape(c)) for o, c in pairs
        ]

    def apply(self, text: str) -> str:
        """Apply the mode."""
        if self._mode == "keep":
            return text
        if self._mode == "with_content":
            previous = None
            while previous != text:  # innermost pairs first, until nothing is left to remove
                previous = text
                for pattern in self._patterns:
                    text = pattern.sub("", text)
        return "".join(c for c in text if c not in self._chars)


class LowercaseLatin:
    """Lowercase Latin-script letters only (Hebrew and other scripts are untouched)."""

    name = "lowercase_latin"

    def apply(self, text: str) -> str:
        """Lowercase the Latin letters."""
        return "".join(c.lower() if c.isalpha() and unicodedata.name(c, "").startswith("LATIN") else c for c in text)


class CollapseWhitespace:
    """Single spaces, trimmed."""

    name = "collapse_whitespace"

    def apply(self, text: str) -> str:
        """Collapse every run of whitespace to one space and trim."""
        return " ".join(text.split())
