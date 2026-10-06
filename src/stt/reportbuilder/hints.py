"""The diagnosis rules (pure code, standard library only) and their registry.

How to run: this module is not a script; the report builder uses it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.reportbuilder.hints import default_rule_registry; print(default_rule_registry().names())"
Example output:
    ['edge_of_range', 'failed_or_pruned', 'gain_concentrated', 'length_cap_or_empty', 'no_learning', 'overfitting', 'short_clip_weakness', 'unstable_training']
Every rule looks at the data it is given and returns a `DiagnosisHint` (what it noticed, the numbers behind it, and what it cannot tell) or None. A hint is worded as a hint ("may", "could"),
never as a stated fact, and its thresholds come from `report.hints` (DEFAULT values to calibrate on real runs). A rule whose input is missing returns None. Rules never change the data and never
compute WER again (they add up the per-file word errors that the Evaluator stored). A new rule is a new class registered by name.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence

from stt.evaluator import EvalResult
from stt.reportbuilder.errors import ReportConfigError
from stt.reportbuilder.types import DiagnosisHint, DiagnosisRule, ReportData
from stt.trainercore import StepMetrics


def history_of_interest(data: ReportData) -> Sequence[StepMetrics] | None:
    """The retraining history of the winner, or of the best finalist when the baseline stays."""
    history = data.winner_history()
    if history is None and data.selection is not None and data.selection.table:
        history = data.final_histories.get(data.selection.table[0].trial_id)
    return history


def group_errors(result: EvalResult) -> dict[str, tuple[int, int]]:
    """Per group: (word errors, reference words), added up from the stored per-file scores."""
    out: dict[str, tuple[int, int]] = {}
    for f in result.files:
        errors, words = out.get(f.group_id, (0, 0))
        out[f.group_id] = (errors + f.word_errors, words + f.ref_words)
    return out


def _val_points(history: Sequence[StepMetrics]) -> list[tuple[int, float, float | None]]:
    return [(m.step, float(m.val_loss), m.train_loss) for m in history if m.val_loss is not None and math.isfinite(m.val_loss)]


class Overfitting:
    """The Val loss rose at each of the last K evaluations while the training loss fell."""

    name = "overfitting"

    def evaluate(self, data: ReportData) -> DiagnosisHint | None:
        history = history_of_interest(data)
        k = data.settings.hints.overfitting_last_k_evals
        if history is None:
            return None
        points = _val_points(history)
        if len(points) < k + 1:
            return None
        last = points[-(k + 1) :]
        rising = all(b[1] > a[1] for a, b in zip(last, last[1:], strict=False))
        train = [p[2] for p in last if p[2] is not None]
        falling = len(train) >= 2 and train[-1] < train[0]
        if not (rising and falling):
            return None
        return DiagnosisHint(
            self.name, "warning",
            f"The Val loss rose at each of the last {k} evaluations (steps {last[0][0]} to {last[-1][0]}) while the training loss fell: the model may be starting to overfit the training data.",
            {"val_loss_first": round(last[0][1], 4), "val_loss_last": round(last[-1][1], 4), "train_loss_first": round(train[0], 4), "train_loss_last": round(train[-1], 4)},
            "Only the checkpoint with the best Val loss is kept, so the returned model is not the last one; a noisy Val loss on a small Val set can look like this too.",
        )


class NoLearning:
    """The best Val loss is hardly better than the first one."""

    name = "no_learning"

    def evaluate(self, data: ReportData) -> DiagnosisHint | None:
        history = history_of_interest(data)
        if history is None:
            return None
        points = _val_points(history)
        if len(points) < 2:
            return None
        first, best = points[0][1], min(p[1] for p in points)
        improvement = (first - best) / abs(first) if first else 0.0
        if improvement >= data.settings.hints.no_learning_min_rel_improvement:
            return None
        return DiagnosisHint(
            self.name, "info",
            f"The Val loss improved by only {improvement:.2%} between the first evaluation and the best one: the training may not be changing the model much (learning rate too low, too few steps, or the base model is already close).",
            {"val_loss_first": round(first, 4), "val_loss_best": round(best, 4), "relative_improvement": round(improvement, 4)},
            "A flat Val loss does not mean the WER is flat: the loss and the WER can move differently.",
        )


class UnstableTraining:
    """A training loss far above its recent median."""

    name = "unstable_training"

    def evaluate(self, data: ReportData) -> DiagnosisHint | None:
        history = history_of_interest(data)
        if history is None:
            return None
        losses = [(m.step, float(m.train_loss)) for m in history if m.train_loss is not None and math.isfinite(m.train_loss)]
        factor = data.settings.hints.unstable_spike_factor
        spikes = [(s, v) for i, (s, v) in enumerate(losses) if i >= 5 and v > factor * statistics.median(x for _, x in losses[i - 5 : i])]
        if not spikes:
            return None
        return DiagnosisHint(
            self.name, "warning",
            f"The training loss jumped above {factor:g} times its recent median {len(spikes)} time(s), first at step {spikes[0][0]}: the training may be unstable (learning rate too high, or a few very hard examples).",
            {"spikes": len(spikes), "first_spike_step": spikes[0][0], "first_spike_loss": round(spikes[0][1], 4)},
            "A single batch of unusual audio can spike the loss without the training being unstable.",
        )


class GainConcentrated:
    """Most of the improvement comes from a few groups (lessons)."""

    name = "gain_concentrated"

    def evaluate(self, data: ReportData) -> DiagnosisHint | None:
        winner = data.winner_val()
        if data.baseline_val is None or winner is None or winner is data.baseline_val:
            return None
        base, win = group_errors(data.baseline_val), group_errors(winner)
        groups = sorted(set(base) & set(win))
        if len(groups) < 2:
            return None
        reductions = [base[g][0] - win[g][0] for g in groups]
        total = sum(reductions)
        if total <= 0:
            return None
        top = max(1, math.ceil(data.settings.hints.gain_top_group_fraction * len(groups)))
        share = sum(sorted((r for r in reductions if r > 0), reverse=True)[:top]) / total
        if share < data.settings.hints.gain_share:
            return None
        return DiagnosisHint(
            self.name, "info",
            f"{top} of {len(groups)} groups account for {share:.0%} of the total reduction in word errors: the improvement may come from a few lessons rather than from the whole set.",
            {"groups": len(groups), "top_groups": top, "share_of_gain": round(share, 3), "total_errors_removed": total},
            "With few groups the share is naturally high; look at the group table before concluding.",
        )


class ShortClipWeakness:
    """The shortest files have a clearly higher WER than the rest."""

    name = "short_clip_weakness"

    def evaluate(self, data: ReportData) -> DiagnosisHint | None:
        result = data.winner_val()
        if result is None:
            return None
        files = [f for f in result.files if isinstance(f.attributes.get("duration"), int | float)]
        if len(files) < 8:
            return None
        files.sort(key=lambda f: (float(f.attributes["duration"]), f.sample_id))  # type: ignore[arg-type]
        cut = max(1, math.ceil(data.settings.hints.short_clip_quantile * len(files)))
        short, rest = files[:cut], files[cut:]
        words_s, words_r = sum(f.ref_words for f in short), sum(f.ref_words for f in rest)
        if not words_s or not words_r:
            return None
        wer_s, wer_r = sum(f.word_errors for f in short) / words_s, sum(f.word_errors for f in rest) / words_r
        ratio = wer_s / wer_r if wer_r else math.inf
        if wer_r == 0 or ratio < data.settings.hints.short_clip_wer_ratio:
            return None
        return DiagnosisHint(
            self.name, "info",
            f"The shortest {len(short)} of {len(files)} files have a WER of {wer_s:.3f}, {ratio:.1f} times the {wer_r:.3f} of the others: the model may be weaker on short clips.",
            {"short_files": len(short), "wer_short": round(wer_s, 4), "wer_rest": round(wer_r, 4), "ratio": round(ratio, 3)},
            "Short clips have few words, so one wrong word moves their WER a lot; shortness can also go with other properties (a speaker, a topic).",
        )


class LengthCapOrEmpty:
    """Many outputs hit the length cap or are empty."""

    name = "length_cap_or_empty"

    def evaluate(self, data: ReportData) -> DiagnosisHint | None:
        result = data.winner_val()
        if result is None or not result.files:
            return None
        bad = [f for f in result.files if f.hit_length_cap or f.empty_output]
        share = len(bad) / len(result.files)
        if share < data.settings.hints.length_cap_share or not bad:
            return None
        capped, empty = sum(1 for f in result.files if f.hit_length_cap), sum(1 for f in result.files if f.empty_output)
        return DiagnosisHint(
            self.name, "warning",
            f"{len(bad)} of {len(result.files)} outputs ({share:.0%}) hit the length cap or are empty: the model may be looping or giving up on these files, which can inflate the WER.",
            {"files": len(result.files), "hit_length_cap": capped, "empty_output": empty, "share": round(share, 3)},
            "Check the listening list; the cap is `eval.generation.max_tokens_per_audio_sec`, a raised cap may be needed for slow speech.",
        )


class EdgeOfRange:
    """The best trial sits at the edge of a searched range."""

    name = "edge_of_range"

    def evaluate(self, data: ReportData) -> DiagnosisHint | None:
        search = data.search
        if search is None or search.best_params is None:
            return None
        edges = []
        for name, spec in search.space.items():
            value = search.best_params.get(name)
            low, high = spec.get("low"), spec.get("high")
            if spec.get("kind") == "categorical" or low is None or high is None or not isinstance(value, int | float):
                continue
            if math.isclose(value, low, rel_tol=1e-6) or math.isclose(value, high, rel_tol=1e-6):
                edges.append(f"{name} = {value:g} (range {low:g} to {high:g})")
        if not edges:
            return None
        return DiagnosisHint(
            self.name, "info",
            "The best trial is at the edge of the searched range of " + "; ".join(edges) + ": a better value may lie outside the range.",
            {"parameters": ", ".join(edges)},
            "With a grid of few points the best point is often at an edge by construction.",
        )


class FailedOrPruned:
    """Many trials failed or were pruned."""

    name = "failed_or_pruned"

    def evaluate(self, data: ReportData) -> DiagnosisHint | None:
        search = data.search
        if search is None or not search.trials:
            return None
        bad = [t for t in search.trials if t.status in ("failed", "pruned")]
        share = len(bad) / len(search.trials)
        if share < data.settings.hints.failed_share or not bad:
            return None
        failed = sum(1 for t in bad if t.status == "failed")
        return DiagnosisHint(
            self.name, "warning",
            f"{len(bad)} of {len(search.trials)} trials ({share:.0%}) failed or were pruned ({failed} failed): the search space or the settings may be too aggressive.",
            {"trials": len(search.trials), "failed": failed, "pruned": len(bad) - failed, "share": round(share, 3)},
            "Pruning is meant to stop poor trials early; a failed trial's reason is in the search report.",
        )


class RuleRegistry:
    """Name -> rule factory. A read-only mapping."""

    def __init__(self, factories: dict[str, Callable[[], DiagnosisRule]]) -> None:
        self._factories = dict(factories)

    def names(self) -> list[str]:
        """The registered names, sorted."""
        return sorted(self._factories)

    def create(self, name: str) -> DiagnosisRule:
        """The rule `name`. Errors: ReportConfigError listing the registered names."""
        factory = self._factories.get(name)
        if factory is None:
            raise ReportConfigError(f"Unknown diagnosis rule '{name}' in report.hints.enabled_rules. Registered rules: {', '.join(self.names())}.")
        return factory()


def default_rule_registry() -> RuleRegistry:
    """The registry with the eight rules of the brief."""
    classes: list[Callable[[], DiagnosisRule]] = [Overfitting, NoLearning, UnstableTraining, GainConcentrated, ShortClipWeakness, LengthCapOrEmpty, EdgeOfRange, FailedOrPruned]
    return RuleRegistry({c().name: c for c in classes})  # type: ignore[attr-defined]
