"""The `text` configuration section, owned by the TextNormalizer component.

How to run: this module is not a script. The composition root registers the section:
    from stt.config import SectionRegistry, load_config
    from stt.textnormalizer.config import TextConfig
    config = load_config(path, registry=SectionRegistry({"text": TextConfig}))
    text = config.section("text", TextConfig)
Example YAML with every key and its DEFAULT: configs/text_default.yaml. The default character lists are
general (standard Hebrew and Latin punctuation, invisible and direction-control characters); a dataset
changes them in YAML, never in code (and a changed eval profile needs a new `text.eval_version`).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from stt.config import SectionModel

DEFAULT_TRAIN_STEPS = ("unicode_clean", "strip_timestamps", "unify_quotes", "remove_bracket_markup", "collapse_whitespace")
DEFAULT_EVAL_STEPS = (
    "unicode_clean",
    "strip_timestamps",
    "remove_punctuation",
    "unify_quotes",
    "remove_nikud",
    "remove_bracket_markup",
    "lowercase_latin",
    "collapse_whitespace",
)
# Zero-width and direction-control characters, soft hyphen, byte order mark, Arabic letter mark.
DEFAULT_INVISIBLE = "​‌‍‎‏‪‫‬‭‮⁠⁦⁧⁨⁩­﻿؜"
# Standard Hebrew and Latin punctuation, including quote variants, geresh, gershayim, maqaf, hyphen and dashes.
DEFAULT_PUNCTUATION = ".,;:!?…\"'“”‘’׳״־-–—"
DEFAULT_SPACE_CHARS = "־"  # the Hebrew maqaf is replaced by a space, not removed
DEFAULT_QUOTE_MAP = {
    "'": "׳",
    "’": "׳",
    '"': "״",
    "“": "״",
    "”": "״",
}
DEFAULT_TIMESTAMP_PATTERN = r"<\|(\d+(?:\.\d+)?)\|>"


class ProfileConfig(SectionModel):
    """One profile: the ordered list of step names and an optional bracket mode of its own."""

    steps: tuple[str, ...] = Field(description="Ordered step names. The order is meaningful (timestamps are stripped before punctuation).")
    bracket_mode: Literal["chars_only", "with_content", "keep"] | None = Field(
        default=None, description="DEFAULT unset = use text.bracket.mode. Recipe for a non-speech marker: train keep, eval with_content."
    )


class ProfilesConfig(SectionModel):
    """The two profiles: `train` (gentle cleaning) and `eval` (aggressive cleaning, the measuring stick)."""

    train: ProfileConfig = ProfileConfig(steps=DEFAULT_TRAIN_STEPS)
    eval: ProfileConfig = ProfileConfig(steps=DEFAULT_EVAL_STEPS)


class TrainTextConfig(SectionModel):
    """Settings of the training label."""

    timestamps: str = Field(default="strip", description="DEFAULT strip. `keep` is not implemented in v1 (O3, O48).")

    @field_validator("timestamps")
    @classmethod
    def _only_strip(cls, value: str) -> str:
        if value == "keep":
            raise ValueError("text.train.timestamps 'keep' is not implemented in v1 (a documented later extension, O3/O48); use 'strip'.")
        if value != "strip":
            raise ValueError(f"text.train.timestamps must be 'strip' (got '{value}').")
        return value


class BracketConfig(SectionModel):
    """Bracket pairs and what to do with them."""

    pairs: tuple[tuple[str, str], ...] = Field(default=(("[", "]"), ("(", ")")), description="DEFAULT [] and (). Pairs of opening and closing characters.")
    mode: Literal["chars_only", "with_content", "keep"] = Field(
        default="chars_only", description="DEFAULT chars_only (O25): remove the bracket characters, keep the content."
    )

    @field_validator("pairs")
    @classmethod
    def _pairs_are_single_characters(cls, value: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
        for opening, closing in value:
            if len(opening) != 1 or len(closing) != 1:
                raise ValueError(f"each bracket pair needs two single characters, got ({opening!r}, {closing!r}).")
        return value


class PunctuationConfig(SectionModel):
    """Characters of the remove_punctuation step."""

    chars: str = Field(default=DEFAULT_PUNCTUATION, description="DEFAULT a standard Hebrew and Latin set. Every listed character is removed.")
    space_chars: str = Field(default=DEFAULT_SPACE_CHARS, description="DEFAULT the Hebrew maqaf. Listed characters are replaced by a space instead.")


class QuotesConfig(SectionModel):
    """Mapping of quote and geresh variants to one form."""

    map: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_QUOTE_MAP), description="DEFAULT: ' and the right single quote become geresh; \", curly double quotes become gershayim."
    )


class UnicodeConfig(SectionModel):
    """Settings of the unicode_clean step."""

    invisible_chars: str = Field(default=DEFAULT_INVISIBLE, description="DEFAULT zero-width, direction-control and similar characters, removed.")


class InventoryConfig(SectionModel):
    """Settings of the character inventory in the report."""

    top_n: int = Field(default=50, ge=1, description="DEFAULT 50. How many characters (most frequent first) the report lists.")


class TextConfig(SectionModel):
    """The `text` section."""

    eval_version: int = Field(default=1, ge=1, description="DEFAULT 1. Version of the eval profile; WER of different versions is not comparable.")
    profiles: ProfilesConfig = ProfilesConfig()
    train: TrainTextConfig = TrainTextConfig()
    bracket: BracketConfig = BracketConfig()
    punctuation: PunctuationConfig = PunctuationConfig()
    quotes: QuotesConfig = QuotesConfig()
    unicode: UnicodeConfig = UnicodeConfig()
    timestamp_pattern: str = Field(
        default=DEFAULT_TIMESTAMP_PATTERN,
        description="DEFAULT Whisper-style <|12.34|>. A regular expression with ONE capture group holding the time in seconds.",
    )
    inventory: InventoryConfig = InventoryConfig()

    @field_validator("timestamp_pattern")
    @classmethod
    def _pattern_has_one_group(cls, value: str) -> str:
        import re

        try:
            compiled = re.compile(value)
        except re.error as exc:
            raise ValueError(f"not a valid regular expression: {exc}") from exc
        if compiled.groups != 1:
            raise ValueError("text.timestamp_pattern needs exactly one capture group (the time in seconds).")
        return value
