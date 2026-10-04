"""TextNormalizer: builds the train and eval profiles from Config and normalizes samples.

How to run: this module is not a script; the composition root creates the normalizer. Example
(PowerShell, project root; the full working example is in docs/COMPONENTS.md, C2):
    $env:PYTHONPATH = "src"
    python -c "from stt.textnormalizer import TextConfig, build_profile; f = build_profile(TextConfig(), 'eval'); print(f('<|0.5|>שלום, עולם. <|2.0|>'))"
Example output:
    שלום עולם
`build_profile(config, name)` returns a plain `Callable[[str], str]`: the Evaluator applies the SAME eval
profile to references and to model outputs. The input samples are never changed.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence

from stt.datasource import AudioSample
from stt.textnormalizer.config import TextConfig
from stt.textnormalizer.errors import TextNormalizerError
from stt.textnormalizer.registry import StepRegistry, default_registry
from stt.textnormalizer.steps import is_nikud
from stt.textnormalizer.timestamps import parse_segments
from stt.textnormalizer.types import CharStat, NormalizedSample, NormalizeReport, NormalizeResult, NormalizationStep

PROFILE_NAMES = ("train", "eval")
MAX_PASSES = 10


def _is_hebrew_letter(c: str) -> bool:
    return "א" <= c <= "ת"


def _inventory_char(c: str) -> bool:
    """A character the inventory lists: not a Hebrew letter, not a digit, not whitespace."""
    return not (_is_hebrew_letter(c) or c.isspace() or unicodedata.category(c) == "Nd")


class TextNormalizer:
    """Applies the configured profiles. Errors: TextNormalizerError for an unknown step or profile (names listed)."""

    def __init__(self, config: TextConfig, registry: StepRegistry | None = None) -> None:
        self._config = config
        self._registry = registry or default_registry()
        self._timestamp = re.compile(config.timestamp_pattern)
        self._steps: dict[str, tuple[NormalizationStep, ...]] = {
            name: tuple(self._registry.create(step, config, name) for step in getattr(config.profiles, name).steps)
            for name in PROFILE_NAMES
        }
        self._bracket_chars = {c for pair in config.bracket.pairs for c in pair}

    def build_profile(self, name: str) -> Callable[[str], str]:
        """Return the function that applies profile `name` ('train' or 'eval') to a string."""
        if name not in self._steps:
            raise TextNormalizerError(f"Unknown profile '{name}'. Registered profiles: {', '.join(PROFILE_NAMES)}.")
        steps = self._steps[name]

        def once(text: str) -> str:
            for step in steps:
                text = step.apply(text)
            return text

        def apply(text: str) -> str:
            # Idempotence (rule 3): removing characters can create a pattern that an earlier step already
            # passed (for example "<|(5|>" becomes a timestamp token), so the steps repeat until nothing changes.
            result = once(text)
            for _ in range(MAX_PASSES):
                again = once(result)
                if again == result:
                    break
                result = again
            return result

        return apply

    def normalize(self, samples: Sequence[AudioSample]) -> NormalizeResult:
        """Return a NormalizedSample for every input sample (same order) and the report. Never drops a sample."""
        train, evaluate = self.build_profile("train"), self.build_profile("eval")
        results: list[NormalizedSample] = []
        raw_counts: Counter[str] = Counter()
        raw_texts_with: Counter[str] = Counter()
        train_counts: Counter[str] = Counter()
        eval_counts: Counter[str] = Counter()
        flag_counts: Counter[str] = Counter()
        changed = {"train": 0, "eval": 0}
        empty: dict[str, list[str]] = {"train": [], "eval": []}
        for sample in samples:
            raw = sample.text
            had_tokens, segments = parse_segments(raw, self._timestamp)
            train_text, eval_text = train(raw), evaluate(raw)
            masked = self._timestamp.sub("", raw)
            flags = self._flags(masked, had_tokens, segments is None and had_tokens, train_text, eval_text)
            results.append(NormalizedSample(sample, train_text, eval_text, segments, flags))
            flag_counts.update(flags)
            changed["train"] += train_text != raw
            changed["eval"] += eval_text != raw
            for profile, value in (("train", train_text), ("eval", eval_text)):
                if value == "":
                    empty[profile].append(sample.id)
            raw_counts.update(c for c in masked if _inventory_char(c))
            raw_texts_with.update({c for c in masked if _inventory_char(c)})
            train_counts.update(c for c in train_text if _inventory_char(c))
            eval_counts.update(c for c in eval_text if _inventory_char(c))
        everything = set(raw_counts) | set(train_counts) | set(eval_counts)
        ordered = sorted(everything, key=lambda c: (-raw_counts[c], -max(train_counts[c], eval_counts[c]), c))
        stats = tuple(
            CharStat(
                char=c,
                codepoint=f"U+{ord(c):04X}",
                name=unicodedata.name(c, f"<{unicodedata.category(c)}>"),
                raw=raw_counts[c],
                train=train_counts[c],
                eval=eval_counts[c],
                samples=raw_texts_with[c],
            )
            for c in ordered[: self._config.inventory.top_n]
        )
        report = NormalizeReport(
            samples=len(results),
            flag_counts={flag: flag_counts[flag] for flag in sorted(flag_counts)},
            changed=changed,
            empty_ids={k: tuple(v) for k, v in empty.items()},
            inventory=stats,
            inventory_total=len(everything),
            eval_version=self._config.eval_version,
            profile_steps={name: tuple(s.name for s in steps) for name, steps in self._steps.items()},
            notes=("Inventory counts for the raw text ignore the timestamp tokens (they contain digits, '.', '<', '|', '>').",),
        )
        return NormalizeResult(results, report)

    def _flags(self, masked: str, had_tokens: bool, bad_tokens: bool, train_text: str, eval_text: str) -> tuple[str, ...]:
        found = []
        if had_tokens:
            found.append("had_timestamps")
        if bad_tokens:
            found.append("bad_timestamps")
        if any(c in self._bracket_chars for c in masked):
            found.append("had_brackets")
        if any(unicodedata.category(c) == "Nd" for c in masked):
            found.append("had_digits")
        if any(c.isalpha() and unicodedata.name(c, "").startswith("LATIN") for c in masked):
            found.append("had_latin")
        if any(is_nikud(c) for c in masked):
            found.append("had_nikud")
        if train_text == "" or eval_text == "":
            found.append("empty_after_normalization")
        return tuple(found)


def build_profile(config: TextConfig, name: str, registry: StepRegistry | None = None) -> Callable[[str], str]:
    """Build profile `name` ('train' or 'eval') from Config as a plain `Callable[[str], str]`."""
    return TextNormalizer(config, registry).build_profile(name)
