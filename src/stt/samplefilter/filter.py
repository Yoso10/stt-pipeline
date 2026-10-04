"""SampleFilter: decides, per split, which samples are kept by the configured rules and reports why.

How to run: this module is not a script; the composition root creates the filter. Example (PowerShell,
project root; `normalized` is a list of NormalizedSample from the TextNormalizer, the full example is in
docs/COMPONENTS.md, C1b):
    $env:PYTHONPATH = "src"
    python -c "from stt.samplefilter import FilterConfig, SampleFilter; f = SampleFilter(FilterConfig()); print(f.filter([], 'train').report.summary())"
Example output:
    split train: mode report, seen 0, kept 0, would reject 0, would-be kept ratio 1.0000 ...
Structural rules (duration, text length) run on ALL splits; quality rules run on train only by default.
`report` mode keeps everything and only counts; `apply` removes. The input is never modified; no file is
read or written; no tokenizer is imported (a token counter, if wanted, is injected as a callable).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from stt.samplefilter.config import FilterConfig
from stt.samplefilter.errors import FilterError, FilterGateError
from stt.samplefilter.registry import RuleRegistry, default_registry
from stt.samplefilter.types import (
    FilterReport,
    FilterResult,
    RejectedSample,
    RuleOutcome,
    RuleParams,
    SampleRule,
    TokenCounter,
)
from stt.textnormalizer import NormalizedSample


@dataclass(frozen=True)
class _Active:
    rule: SampleRule
    kind: str
    applies_to: tuple[str, ...]
    threshold: float | None

    def applies(self, split: str) -> bool:
        return "*" in self.applies_to or split in self.applies_to


def _has_field(sample: NormalizedSample, field: str) -> bool:
    if field == "duration_sec":
        return True
    if field.startswith("extra:"):
        return field[len("extra:") :] in sample.sample.extra
    return hasattr(sample, field)


class SampleFilter:
    """Applies the configured rules per split.

    Inputs: the `filter` config, an optional rule registry, and an optional token counter
    (`Callable[[str], int]`, needed only when the `max_label_tokens` rule is enabled).
    Errors: FilterError for an unknown rule name, an enabled rule without its threshold, field or counter
    (raised when the filter is built); FilterError when a field a running rule needs is missing from the
    samples; FilterGateError when a split would keep less than `filter.min_kept_ratio` (the error carries
    the full result) or becomes empty in `apply` mode.
    """

    def __init__(self, config: FilterConfig, registry: RuleRegistry | None = None, token_counter: TokenCounter | None = None) -> None:
        self._config = config
        registry = registry or default_registry()
        for name in config.rules:
            registry.spec(name)  # unknown name: FAIL listing the registered names
        self._active: dict[str, _Active] = {}
        for name in registry.names():
            spec = registry.spec(name)
            overrides = config.rules.get(name)
            enabled = overrides.enabled if overrides is not None and overrides.enabled is not None else None
            if enabled is None:
                enabled = spec.default_enabled or (overrides is not None and (overrides.threshold is not None))
            if not enabled:
                continue
            kind = (overrides.kind if overrides else None) or spec.kind
            threshold = overrides.threshold if overrides and overrides.threshold is not None else spec.default_threshold
            params = RuleParams(
                threshold=threshold,
                field=overrides.field if overrides else None,
                measure_on=(overrides.measure_on if overrides else None) or spec.default_measure_on,
                token_counter=token_counter,
            )
            applies_to = (overrides.applies_to if overrides else None) or (("*",) if kind == "structural" else ("train",))
            self._active[name] = _Active(spec.factory(params), kind, applies_to, threshold)

    @property
    def active_rules(self) -> tuple[str, ...]:
        """Names of the enabled rules."""
        return tuple(self._active)

    def filter(self, samples: Sequence[NormalizedSample], split: str) -> FilterResult:
        """Judge every sample of `split`. See the class docstring for the modes, the gates and the errors."""
        running = {n: a for n, a in self._active.items() if a.applies(split)}
        skipped = tuple(n for n, a in self._active.items() if not a.applies(split))
        self._check_fields(running, samples, split)
        failures: dict[str, list[tuple[str, str]]] = {}
        failed_count = dict.fromkeys(running, 0)
        for sample in samples:
            for name, active in running.items():
                verdict = active.rule.check(sample)
                if not verdict.passed:
                    failures.setdefault(sample.sample.id, []).append((name, verdict.reason))
                    failed_count[name] += 1
        would = [
            RejectedSample(s.sample.id, tuple(n for n, _ in failures[s.sample.id]), tuple(r for _, r in failures[s.sample.id]))
            for s in samples
            if s.sample.id in failures
        ]
        applying = self._config.mode == "apply"
        kept = [s for s in samples if s.sample.id not in failures] if applying else list(samples)
        seen = len(samples)
        ratio = (seen - len(would)) / seen if seen else 1.0
        report = FilterReport(
            split=split,
            mode=self._config.mode,
            seen=seen,
            kept=len(kept),
            would_reject=len(would),
            kept_ratio=ratio,
            rules_run=tuple(RuleOutcome(n, a.kind, a.threshold, failed_count[n]) for n, a in running.items()),
            rules_not_applied=skipped,
        )
        result = FilterResult(kept=kept, rejected=would if applying else [], would_reject=would, report=report)
        breakdown = ", ".join(f"{n}: {c}" for n, c in sorted(failed_count.items(), key=lambda x: -x[1]) if c) or "no rule failed"
        if seen and ratio < self._config.min_kept_ratio:
            raise FilterGateError(
                f"Split '{split}' would keep {ratio:.1%} of its samples, below filter.min_kept_ratio {self._config.min_kept_ratio:.1%} "
                f"({len(would)} of {seen} would be removed; per rule: {breakdown}). Read the report, relax the dominant rule "
                "in report mode first; do not just lower min_kept_ratio.",
                result,
            )
        if applying and seen and not kept:
            worst = max(failed_count, key=lambda n: failed_count[n])
            raise FilterGateError(f"Split '{split}' is empty after filtering. The rule that removed the most samples: {worst} ({breakdown}).", result)
        return result

    def filter_splits(self, splits: Mapping[str, Sequence[NormalizedSample]]) -> dict[str, FilterResult]:
        """Run `filter` for every split (in the given order) and return the results by split name."""
        return {name: self.filter(samples, name) for name, samples in splits.items()}

    def _check_fields(self, running: Mapping[str, _Active], samples: Sequence[NormalizedSample], split: str) -> None:
        for name, active in running.items():
            for field in active.rule.required_fields:
                missing = [s.sample.id for s in samples if not _has_field(s, field)]
                if missing:
                    raise FilterError(
                        f"Rule '{name}' needs the field '{field}', but {len(missing)} sample(s) of split '{split}' lack it "
                        f"(for example {', '.join(missing[:3])}). Map that column in the DataSource configuration, "
                        f"or set filter.rules.{name}.enabled: false (or name another field in filter.rules.{name}.field)."
                    )
