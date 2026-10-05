"""Subsampler: select a smaller, deterministic, nested subset of the samples (C3b).

How to run: from Python (PowerShell, project root; the full working example is in docs/COMPONENTS.md, C3b):
    $env:PYTHONPATH = "src"
    python -c "from stt.subsampler import Subsampler; print(Subsampler)"
`Subsampler(registry).subsample(splits, spec) -> SubsampleResult`. Every unit (a sample or a whole group) gets a rank from the SHA-256 of
`seed|repeat_index|unit key`; units are taken in rank order until the requested size is reached (the unit that crosses the target is included, so the
overshoot is at most one unit and is reported). Therefore the same inputs and seed always give the same subset, the input order does not matter, and a
smaller subset is contained in a larger one (strategy, seed and repeat index unchanged). No sample is modified and no file is touched.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from collections.abc import Mapping, Sequence

from stt.subsampler.errors import SubsampleConfigError, SubsampleError
from stt.subsampler.strategies import StrategyRegistry, default_registry
from stt.subsampler.types import SplitSizes, SubsampleReport, SubsampleResult, SubsampleSize, SubsampleSpec, SubsampleStrategy
from stt.textnormalizer import NormalizedSample

EPSILON = 1e-9
Splits = Mapping[str, Sequence[NormalizedSample]]


def rank(seed: int, repeat_index: int, key: str) -> tuple[int, str]:
    """The rank of a unit: the first 8 bytes of SHA-256(`seed|repeat_index|key`), then the key (so equal hashes still order the same way)."""
    digest = hashlib.sha256(f"{seed}|{repeat_index}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big"), key


def data_fingerprint(splits: Splits) -> str:
    """SHA-256 over the sorted lines `split|id|group|duration` of all the input samples (independent of the input order)."""
    lines = sorted(f"{name}|{n.sample.id}|{n.sample.group_id}|{n.sample.duration_sec!r}" for name, samples in splits.items() for n in samples)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def spec_document(spec: SubsampleSpec) -> dict[str, object]:
    """The spec as plain data (stored in the JSON file and part of the fingerprint)."""
    return {
        "size": {"kind": spec.size.kind, "value": spec.size.value},
        "strategy": spec.strategy,
        "scope": spec.scope,
        "val": spec.val,
        "seed": spec.seed,
        "repeat_index": spec.repeat_index,
        "train_split": spec.train_split,
        "val_split": spec.val_split,
        "min_train_samples": spec.min_train_samples,
        "val_samples": spec.val_samples,
    }


def result_fingerprint(spec: SubsampleSpec, data_fp: str, chosen: Mapping[str, Sequence[str]]) -> str:
    """SHA-256 over the spec, the data fingerprint and the sorted chosen ids of every split."""
    body = {"spec": spec_document(spec), "data": data_fp, "ids": {name: sorted(ids) for name, ids in sorted(chosen.items())}}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def _minutes(n: NormalizedSample) -> float:
    return n.sample.duration_sec / 60.0


def _sizes(name: str, reduced: bool, before: Sequence[NormalizedSample], after: Sequence[NormalizedSample], strategy: SubsampleStrategy) -> SplitSizes:
    groups = lambda items: len({n.sample.group_id for n in items})  # noqa: E731
    return SplitSizes(name, reduced, len(before), groups(before), sum(map(_minutes, before)), len(after), groups(after), sum(map(_minutes, after)))


class Subsampler:
    """Selects subsets by rank. The strategies come from a registry (a new strategy is a new class)."""

    def __init__(self, registry: StrategyRegistry | None = None) -> None:
        self._registry = registry or default_registry()

    def subsample(self, splits: Splits, spec: SubsampleSpec) -> SubsampleResult:
        """Return the subsampled splits.

        Errors: SubsampleConfigError (exit code 2) for an unknown strategy, an unknown split name, or a minutes or samples size with scope all_splits;
        SubsampleError (exit code 3) for a size larger than what exists (both numbers are named), fewer Train samples than `min_train_samples`, or a split
        that would be empty.
        """
        strategy = self._registry.create(spec.strategy)  # an unknown name fails here
        data_fp = data_fingerprint(splits)
        if spec.scope not in ("train_only", "all_splits"):
            raise SubsampleConfigError(f"Unknown subsample.scope '{spec.scope}'. Registered scopes: all_splits, train_only.")
        if spec.train_split not in splits:
            raise SubsampleConfigError(f"The Train split '{spec.train_split}' is not one of the splits: {', '.join(splits)}.")
        copy = {name: list(samples) for name, samples in splits.items()}
        if spec.val_samples is not None and spec.scope != "train_only":
            raise SubsampleConfigError("subsample.val_samples (a fixed Val slice) needs scope train_only: with all_splits every split is reduced by the same fraction.")
        if spec.size.kind == "all" and spec.val_samples is None:
            return self._finish(splits, copy, spec, strategy, data_fp, set(), None)
        targets = self._targets(splits, spec)
        chosen_ids: dict[str, set[str]] = {}
        for name, target in targets.items():
            chosen_ids[name] = self._select(splits[name], target, spec, strategy, name)
        result_splits = {name: ([n for n in samples if n.sample.id in chosen_ids[name]] if name in chosen_ids else list(samples)) for name, samples in splits.items()}
        for name in chosen_ids:
            if not result_splits[name]:
                raise SubsampleError(f"The split '{name}' would be empty after the subsampling (it had {len(splits[name])} samples). Choose a larger size.")
        train = result_splits[spec.train_split]
        if spec.train_split in chosen_ids and len(train) < spec.min_train_samples:
            raise SubsampleError(f"Only {len(train)} Train samples were selected, fewer than subsample.min_train_samples ({spec.min_train_samples}). Choose a larger size, or lower the minimum knowingly.")
        return self._finish(splits, result_splits, spec, strategy, data_fp, set(chosen_ids), targets.get(spec.train_split))

    # ---- targets ---------------------------------------------------------------------------------------------------
    def _targets(self, splits: Splits, spec: SubsampleSpec) -> dict[str, tuple[str, float]]:
        """For every reduced split: (unit, target), the unit being `minutes` or `samples`."""
        size = spec.size
        train = splits[spec.train_split]
        if spec.val_samples is not None:
            if spec.val_split is None or spec.val_split not in splits:
                raise SubsampleConfigError(f"subsample.val_samples needs a Val split, but '{spec.val_split}' is not one of the splits: {', '.join(splits)}.")
            slice_target = {spec.val_split: ("samples", float(spec.val_samples))}
            if size.kind == "all":
                return slice_target
            return {**self._targets(splits, replace(spec, val_samples=None)), **slice_target}
        if spec.scope == "all_splits":
            if size.kind != "fraction":
                raise SubsampleConfigError(f"scope all_splits needs a size given as a fraction (for example 0.1) or all, not {size.describe()}: every split is reduced by the same fraction.")
            return {name: ("minutes", (size.value or 0.0) * sum(map(_minutes, samples))) for name, samples in splits.items()}
        if size.kind == "samples":
            targets = {spec.train_split: ("samples", float(size.value or 0))}
        elif size.kind == "minutes":
            targets = {spec.train_split: ("minutes", float(size.value or 0))}
        else:
            targets = {spec.train_split: ("minutes", (size.value or 0.0) * sum(map(_minutes, train)))}
        if spec.val == "follow":
            if spec.val_split is None or spec.val_split not in splits:
                raise SubsampleConfigError(f"subsample.val 'follow' needs a Val split, but '{spec.val_split}' is not one of the splits: {', '.join(splits)}.")
            total = sum(map(_minutes, train)) if size.kind != "samples" else float(len(train))
            share = min(1.0, targets[spec.train_split][1] / total) if total else 1.0
            targets[spec.val_split] = ("minutes", share * sum(map(_minutes, splits[spec.val_split])))
        return targets

    # ---- selection ---------------------------------------------------------------------------------------------------
    def _select(self, samples: Sequence[NormalizedSample], target: tuple[str, float], spec: SubsampleSpec, strategy: SubsampleStrategy, name: str) -> set[str]:
        unit, wanted = target
        weights: dict[str, float] = defaultdict(float)
        members: dict[str, list[str]] = defaultdict(list)
        for n in samples:
            key = strategy.key(n)
            weights[key] += 1.0 if unit == "samples" else _minutes(n)
            members[key].append(n.sample.id)
        available = sum(weights.values())
        if wanted > available + EPSILON:
            raise SubsampleError(f"Requested {wanted:.2f} {unit} of '{name}' but only {available:.2f} {unit} exist. Choose a smaller size, or the size all.")
        chosen: set[str] = set()
        total = 0.0
        for key in sorted(weights, key=lambda k: rank(spec.seed, spec.repeat_index, k)):
            if total >= wanted - EPSILON:
                break
            chosen.update(members[key])
            total += weights[key]
        return chosen

    # ---- report ----------------------------------------------------------------------------------------------------------
    def _finish(self, before: Splits, after: dict[str, list[NormalizedSample]], spec: SubsampleSpec, strategy: SubsampleStrategy, data_fp: str, reduced: set[str], train_target: tuple[str, float] | None) -> SubsampleResult:
        sizes = tuple(_sizes(name, name in reduced, before[name], after[name], strategy) for name in before)
        warnings = []
        train_after = after[spec.train_split]
        if spec.size.kind != "all" and len({n.sample.group_id for n in train_after}) == 1 and spec.strategy == "by_group":
            warnings.append("only ONE group was selected for Train: a training set with a single group is risky (choose a larger size).")
        unit = requested = achieved = overshoot = None
        if train_target is not None:
            unit, requested = train_target
            achieved = float(len(train_after)) if unit == "samples" else sum(map(_minutes, train_after))
            overshoot = achieved - requested
        requested_text = spec.size.describe() + (f", Val slice of {spec.val_samples} samples" if spec.val_samples is not None else "")
        report = SubsampleReport(True, requested_text, spec.strategy, spec.scope, spec.val, spec.seed, spec.repeat_index, sizes, unit, requested, achieved, overshoot, tuple(warnings))
        fingerprint = result_fingerprint(spec, data_fp, {name: [n.sample.id for n in samples] for name, samples in after.items()})
        return SubsampleResult(after, report, fingerprint)

    @staticmethod
    def disabled(splits: Splits, spec: SubsampleSpec) -> SubsampleResult:
        """`enabled: false`: the input unchanged (the same sample objects, in the same order), and the report says so."""
        kept = {name: list(samples) for name, samples in splits.items()}
        sizes = tuple(SplitSizes(name, False, len(s), len({n.sample.group_id for n in s}), sum(map(_minutes, s)), len(s), len({n.sample.group_id for n in s}), sum(map(_minutes, s))) for name, s in kept.items())
        report = SubsampleReport(False, "disabled", spec.strategy, spec.scope, spec.val, spec.seed, spec.repeat_index, sizes, None, None, None, None)
        return SubsampleResult(kept, report, result_fingerprint(spec, data_fingerprint(splits), {name: [n.sample.id for n in s] for name, s in kept.items()}))
