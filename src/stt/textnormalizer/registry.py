"""The step registry: step name -> factory, so steps are selected by name from Config.

How to run: this module is not a script. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.textnormalizer.registry import default_registry; print(default_registry().names())"
Example output:
    ['collapse_whitespace', 'lowercase_latin', 'remove_bracket_markup', 'remove_nikud', 'remove_punctuation', 'strip_timestamps', 'unicode_clean', 'unify_quotes']
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from types import MappingProxyType

from stt.textnormalizer.config import TextConfig
from stt.textnormalizer.errors import TextNormalizerError
from stt.textnormalizer.steps import (
    CollapseWhitespace,
    LowercaseLatin,
    RemoveBracketMarkup,
    RemoveNikud,
    RemovePunctuation,
    StripTimestampTokens,
    UnicodeClean,
    UnifyQuotes,
)
from stt.textnormalizer.types import NormalizationStep

# A factory gets the whole text config and the name of the profile it is built for.
StepFactory = Callable[[TextConfig, str], NormalizationStep]


class StepRegistry:
    """A read-only name -> factory mapping."""

    def __init__(self, factories: Mapping[str, StepFactory]) -> None:
        self._factories: Mapping[str, StepFactory] = MappingProxyType(dict(factories))

    def names(self) -> list[str]:
        """The registered step names, sorted."""
        return sorted(self._factories)

    def create(self, step: str, config: TextConfig, profile: str) -> NormalizationStep:
        """Build the step `step` for `profile`. Errors: TextNormalizerError listing the registered names."""
        factory = self._factories.get(step)
        if factory is None:
            raise TextNormalizerError(
                f"Unknown step '{step}' in text.profiles.{profile}.steps. Registered steps: {', '.join(self.names())}. "
                "Fix the name in the YAML configuration."
            )
        return factory(config, profile)


def _bracket_step(config: TextConfig, profile: str) -> NormalizationStep:
    override = getattr(config.profiles, profile).bracket_mode
    return RemoveBracketMarkup(config.bracket.pairs, override or config.bracket.mode)


def default_registry() -> StepRegistry:
    """The registry with the eight built-in steps."""
    return StepRegistry(
        {
            "unicode_clean": lambda c, p: UnicodeClean(c.unicode.invisible_chars),
            "strip_timestamps": lambda c, p: StripTimestampTokens(re.compile(c.timestamp_pattern)),
            "remove_punctuation": lambda c, p: RemovePunctuation(c.punctuation.chars, c.punctuation.space_chars),
            "unify_quotes": lambda c, p: UnifyQuotes(c.quotes.map),
            "remove_nikud": lambda c, p: RemoveNikud(),
            "remove_bracket_markup": _bracket_step,
            "lowercase_latin": lambda c, p: LowercaseLatin(),
            "collapse_whitespace": lambda c, p: CollapseWhitespace(),
        }
    )
