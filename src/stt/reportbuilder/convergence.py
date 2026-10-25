"""The convergence view of the report for a `train` + `evaluate` experiment (DECISIONS 90): did the training converge, was it long enough, did it overfit; and what changed.

How to run: this module is not a script; the ReportBuilder uses it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.reportbuilder.convergence import verdict_from_points; print(verdict_from_points([(25, 2.0), (50, 1.5), (75, 1.2), (100, 1.0)], [(25, 2.2), (100, 1.1)]).kind)"
Example output:
    improving
Pure functions and HTML text; no file access, no model, no network, no script. Every caption is written twice, in English and in Hebrew (the Hebrew in a right-to-left block).
THE VERDICT IS COMPUTED from the Val-loss points, never guessed. The three constants below are display rules, named here so that they can be read and changed in one place (they are not config keys):
`STILL_IMPROVING_MARGIN`, `OVERFIT_MARGIN` (both relative: 0.01 = 1%) and `ABSOLUTE_WER_TARGET_BAND` (the owner's absolute target, drawn behind the WER bars).
The Test split is never used here: the final Test appears only when `test_result.json` exists (the `final-test` command).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

from stt.reportbuilder import svg
from stt.reportbuilder.types import ClipRow, FinalTestView, ModelNumbers, ReportData, TrainedRunView
from stt.trainercore import StepMetrics

STILL_IMPROVING_MARGIN = 0.01  # the last Val point must be the lowest AND lower than the previous one by more than this share to say "still improving"
OVERFIT_MARGIN = 0.01  # the last Val point must be higher than the best one by more than this share (while the training loss fell) to say "overfitting"
GOOD_LOSS_FALL = 0.10  # a Val loss that fell by at least this share from the first to the last point counts as "a good loss"
ABSOLUTE_WER_TARGET_BAND = (0.10, 0.30)  # the owner's absolute target for the WER, drawn as a band behind the bars
CLIPS_SHOWN = 15  # clips in each of the two tables (the worst and the best)


@dataclass(frozen=True)
class ConvergenceVerdict:
    """kind: `improving` | `plateau` | `overfitting` | `too_few_points`; `step`: the best step (for overfitting, the step after which the Val loss rose)."""

    kind: str
    step: int | None
    english: str
    hebrew: str


def _e(value: object) -> str:
    return escape(str(value))


def _he(text: str) -> str:
    return f'<p class="rtl" dir="rtl" lang="he">{_e(text)}</p>'


def _both(english: str, hebrew: str, css: str = "small") -> str:
    """A caption in English and in Hebrew."""
    return f'<p class="{css}" lang="en">{_e(english)}</p>{_he(hebrew)}'


def val_points(history: Sequence[StepMetrics]) -> list[tuple[int, float]]:
    """(step, Val loss) of every evaluation, in step order (finite values only)."""
    seen: dict[int, float] = {}
    for m in history:
        if m.val_loss is not None and math.isfinite(m.val_loss):
            seen[m.step] = float(m.val_loss)
    return sorted(seen.items())


def train_points(history: Sequence[StepMetrics]) -> list[tuple[int, float]]:
    """(step, training loss) of every logged step, in step order (finite values only)."""
    seen: dict[int, float] = {}
    for m in history:
        if m.train_loss is not None and math.isfinite(m.train_loss):
            seen[m.step] = float(m.train_loss)
    return sorted(seen.items())


def _train_near(train: Sequence[tuple[int, float]], step: int) -> float | None:
    """The training loss at the logged step nearest to `step` (None without any point)."""
    return min(train, key=lambda p: abs(p[0] - step))[1] if train else None


def verdict_from_points(val: Sequence[tuple[int, float]], train: Sequence[tuple[int, float]]) -> ConvergenceVerdict:
    """The verdict from the data. Order of the rules: too few points; overfitting (the best Val point is before the last one, the last Val point is higher than the best by more than
    `OVERFIT_MARGIN`, and the training loss near the end is lower than near the best step); still improving (the last Val point is the lowest and fell by more than
    `STILL_IMPROVING_MARGIN` against the previous point); otherwise plateau."""
    if len(val) < 2:
        n = len(val)
        return ConvergenceVerdict(
            "too_few_points",
            None,
            f"Not enough points to judge: only {n} Val-loss point(s). Set train.eval_every_steps to about max_steps / 4 to see the curve.",
            f"אין מספיק נקודות כדי לשפוט: רק {n} נקודות ב-Val. הקטינו את train.eval_every_steps לכ-max_steps / 4 כדי לראות את העקומה.",
        )
    best_step, best_loss = min(val, key=lambda p: (p[1], p[0]))
    last_step, last_loss = val[-1]
    previous_loss = val[-2][1]
    train_best = _train_near(train, best_step)
    train_last = _train_near(train, last_step)
    if best_step != last_step and last_loss > best_loss * (1.0 + OVERFIT_MARGIN) and train_best is not None and train_last is not None and train_last < train_best:
        return ConvergenceVerdict(
            "overfitting",
            best_step,
            f"Overfitting: the Val loss rose after step {best_step} (from {best_loss:.4f} to {last_loss:.4f} at step {last_step}) while the training loss fell ({train_best:.4f} to {train_last:.4f}). "
            f"The best model is the one of step {best_step}; it is the one that was saved.",
            f"התאמת יתר (overfitting): ההפסד ב-Val עלה אחרי צעד {best_step} (מ-{best_loss:.4f} ל-{last_loss:.4f} בצעד {last_step}) בזמן שהפסד האימון ירד ({train_best:.4f} ל-{train_last:.4f}). "
            f"המודל הטוב ביותר הוא של צעד {best_step}, וזה המודל שנשמר.",
        )
    fall = (previous_loss - last_loss) / previous_loss if previous_loss > 0 else 0.0
    if last_step == best_step and fall > STILL_IMPROVING_MARGIN:
        return ConvergenceVerdict(
            "improving",
            best_step,
            f"Still improving at the end: the last Val point (step {last_step}, {last_loss:.4f}) is the lowest and fell by {fall * 100:.1f}% against the previous point ({previous_loss:.4f}). "
            "The training was probably too short: more steps should help.",
            f"עדיין משתפר בסוף: נקודת ה-Val האחרונה (צעד {last_step}, {last_loss:.4f}) היא הנמוכה ביותר וירדה ב-{fall * 100:.1f}% לעומת הקודמת ({previous_loss:.4f}). "
            "כנראה האימון היה קצר מדי: יותר צעדים כנראה יעזרו.",
        )
    return ConvergenceVerdict(
        "plateau",
        best_step,
        f"Plateau: the Val loss stopped falling (best {best_loss:.4f} at step {best_step}, last {last_loss:.4f} at step {last_step}; the change at the end is under {STILL_IMPROVING_MARGIN * 100:g}%). "
        "More steps of the same kind are unlikely to help much.",
        f"שלב יציב (plateau): ההפסד ב-Val הפסיק לרדת (הטוב ביותר {best_loss:.4f} בצעד {best_step}, האחרון {last_loss:.4f} בצעד {last_step}; השינוי בסוף קטן מ-{STILL_IMPROVING_MARGIN * 100:g}%). "
        "עוד צעדים באותה צורה כנראה לא יעזרו הרבה.",
    )


def verdict_of(history: Sequence[StepMetrics]) -> ConvergenceVerdict:
    """`verdict_from_points` for a metrics history."""
    return verdict_from_points(val_points(history), train_points(history))


def next_steps(verdict: ConvergenceVerdict, run: TrainedRunView) -> list[tuple[str, str]]:
    """The rules of the box "what to change next": (English, Hebrew) lines. One line per rule that applies."""
    lines: list[tuple[str, str]] = []
    if verdict.kind == "improving":
        lines.append(
            ("Still improving: run MORE steps (one more epoch at a time: raise train.hyperparams.max_steps by the steps of one epoch, and train.eval_every_steps with it).",
             "עדיין משתפר: להריץ יותר צעדים (epoch אחד נוסף בכל פעם: להגדיל את train.hyperparams.max_steps בצעדי epoch אחד, ואת train.eval_every_steps בהתאם).")
        )
    elif verdict.kind == "overfitting":
        lines.append(
            (f"Overfitting: use FEWER steps (stop near step {verdict.step}) or a LOWER learning rate (halve train.hyperparams.learning_rate). More data of the same kind also helps.",
             f"התאמת יתר: פחות צעדים (לעצור בסביבות צעד {verdict.step}) או קצב למידה נמוך יותר (חצו את train.hyperparams.learning_rate). גם יותר נתונים מאותו סוג עוזרים.")
        )
    elif verdict.kind == "plateau":
        lines.append(
            ("Flat: try a HIGHER learning rate (for example double train.hyperparams.learning_rate). If the WER also did not improve, check the labels and the audio before training longer.",
             "שטוח: לנסות קצב למידה גבוה יותר (למשל להכפיל את train.hyperparams.learning_rate). אם גם ה-WER לא השתפר, לבדוק את התוויות והאודיו לפני שמאריכים את האימון.")
        )
    else:
        lines.append(
            ("Too few Val-loss points: set train.eval_every_steps to about max_steps / 4 and run again to see whether the training converged.",
             "מעט מדי נקודות Val: להגדיר את train.eval_every_steps לכ-max_steps / 4 ולהריץ שוב כדי לראות אם האימון התכנס.")
        )
    val = val_points(run.history)
    reduction = relative_reduction(run.baseline.wer, run.trained.wer)
    loss_fell = len(val) >= 2 and val[0][1] > 0 and (val[0][1] - val[-1][1]) / val[0][1] >= GOOD_LOSS_FALL
    not_better = reduction is None or reduction <= 0 or (run.comparison is not None and run.comparison.verdict != "improved")
    if loss_fell and not_better:
        lines.append(
            (f"Good loss but bad WER: the Val loss fell but the WER did not clearly improve. Check for empty or repeated outputs (trained model: {run.trained.empty_output} empty, "
             f"{run.trained.hit_length_cap} hit the length cap) in predictions_trained.csv, and check that the labels say exactly what is spoken.",
             f"הפסד טוב אבל WER רע: ההפסד ב-Val ירד אך ה-WER לא השתפר בבירור. לבדוק פלטים ריקים או חוזרים (במודל המאומן: {run.trained.empty_output} ריקים, "
             f"{run.trained.hit_length_cap} הגיעו לתקרת האורך) בקובץ predictions_trained.csv, ולוודא שהתוויות אומרות בדיוק את מה שנאמר.")
        )
    return lines


def relative_reduction(baseline: float, trained: float) -> float | None:
    """(baseline - trained) / baseline; None when the baseline WER is 0."""
    return (baseline - trained) / baseline if baseline > 0 else None


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 4) -> str:
    return "-" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _rtl(text: str) -> str:
    return f'<span class="rtl" dir="rtl" lang="he">{_e(text)}</span>'


VERDICT_TEXT = {
    "improved": ("Improved: the trained model is clearly better than the baseline (paired bootstrap over lessons).", "שיפור: המודל המאומן טוב בבירור מהבסיס (bootstrap זוגי על פני שיעורים).", "proven"),
    "no_clear_difference": ("No clear difference: the change could be chance.", "אין הבדל ברור: ייתכן שהשינוי הוא מקרי.", "unproven"),
    "worse": ("Worse: the trained model is worse than the baseline.", "גרוע יותר: המודל המאומן גרוע מהבסיס.", "retained"),
    "insufficient_groups": ("Too few lessons in Val for a statement about confidence; only the plain difference is shown.", "מעט מדי שיעורים ב-Val לקביעה סטטיסטית; מוצג רק ההפרש הפשוט.", "unproven"),
    "not_comparable": ("Not comparable: the two results use different evaluation profiles.", "לא ניתן להשוואה: שתי התוצאות משתמשות בפרופילי הערכה שונים.", "unproven"),
}


def summary_html(data: ReportData) -> str:
    """The panel at the top for a `train` + `evaluate` experiment: the verdict line, the numbers and the convergence verdict."""
    run = data.trained_run
    assert run is not None
    cmp = run.comparison
    kind = cmp.verdict if cmp is not None else "not_comparable"
    english, hebrew, css = VERDICT_TEXT.get(kind, VERDICT_TEXT["not_comparable"])
    reduction = relative_reduction(run.baseline.wer, run.trained.wer)
    target = data.settings.target_relative_wer_reduction
    target_text = ""
    if target is not None and reduction is not None:
        target_text = f" Target {target * 100:g}%: {'reached' if reduction >= target else 'not reached'}" + ("" if kind == "improved" or reduction < target else " (point estimate only: the verdict above is not 'improved')") + "."
    verdict = verdict_of(run.history)
    return (
        f'<p class="vline {css}" lang="en"><b>{_e(english)}</b></p>{_he(hebrew)}'
        f'<div class="panel"><p><b>Baseline WER {_num(run.baseline.wer)}, trained WER {_num(run.trained.wer)}</b> on {run.trained.n_files} Val clips. '
        f"Relative reduction of the WER: <b>{_pct(reduction)}</b>.{_e(target_text)}</p>"
        f'<p><b>Training:</b> {_e(verdict.english)}</p>{_he(verdict.hebrew)}</div>'
    )


def trained_html(data: ReportData, warnings: list[str]) -> str:
    """The section "Did the training converge?": the chart, the verdict, baseline against trained, and the box "what to change next"."""
    run = data.trained_run
    assert run is not None
    val, train = val_points(run.history), train_points(run.history)
    cap = data.settings.max_points_per_chart
    if len(train) > cap:
        stride = math.ceil(len(train) / cap)
        train = train[::stride] + ([train[-1]] if train[-1] not in train[::stride] else [])
        warnings.append(f"The training-loss curve was thinned to about {cap} points.")
    best = min(val, key=lambda p: (p[1], p[0])) if val else None
    out = ""
    if val or train:
        out += svg.loss_chart(
            "Val loss and training loss against the step (one chart)",
            [(float(s), v) for s, v in val],
            [(float(s), v) for s, v in train],
            None if best is None else (float(best[0]), best[1]),
            "step",
            "loss",
            "" if best is None else f"best checkpoint: step {best[0]}",
        )
        out += _both(
            "Solid blue line with big points = Val loss at every evaluation. Dashed orange line = training loss. The green ring and line mark the BEST checkpoint (the lowest Val loss): this is the model that was saved.",
            "הקו הכחול המלא עם הנקודות הגדולות = ההפסד ב-Val בכל בדיקה. הקו הכתום המקווקו = הפסד האימון. העיגול והקו הירוקים מסמנים את נקודת השמירה הטובה ביותר (ההפסד הנמוך ביותר ב-Val): זה המודל שנשמר.",
        )
    else:
        out += '<p class="na">Not available in this run: no loss values were found in metrics/trial_NNN/metrics.csv.</p>'
    verdict = verdict_of(run.history)
    css = {"improving": "unproven", "plateau": "unproven", "overfitting": "retained", "too_few_points": "retained"}[verdict.kind]
    out += f'<p class="vline {css}" lang="en"><b>{_e(verdict.english)}</b></p>{_he(verdict.hebrew)}'
    # baseline against trained
    reduction = relative_reduction(run.baseline.wer, run.trained.wer)
    low, high = ABSOLUTE_WER_TARGET_BAND
    out += "<h3>Baseline against trained model</h3>"
    out += svg.wer_band_chart(
        "WER on the Val clips",
        [("baseline", run.baseline.wer, svg.GREY), ("trained", run.trained.wer, svg.PALETTE[0])],
        "WER",
        (low, high),
        f"owner's target band {low:g}-{high:g}",
    )
    out += _both(
        f"The green band is the owner's absolute target for the WER ({low:g} to {high:g}). A bar inside or below the band meets it. The band is a fixed display value, not a measured result.",
        f"הרצועה הירוקה היא יעד ה-WER המוחלט של הבעלים ({low:g} עד {high:g}). עמודה בתוך הרצועה או מתחתיה עומדת ביעד. הרצועה היא ערך תצוגה קבוע, לא תוצאה שנמדדה.",
    )
    cmp = run.comparison
    rows = [
        ["WER", _num(run.baseline.wer), _num(run.trained.wer)],
        ["CER", _num(run.baseline.cer), _num(run.trained.cer)],
        ["Relative reduction of the WER", "", _pct(reduction)],
        ["Empty outputs / hit the length cap", f"{run.baseline.empty_output} / {run.baseline.hit_length_cap}", f"{run.trained.empty_output} / {run.trained.hit_length_cap}"],
        ["Bootstrap verdict (paired, over lessons)", "", _e(cmp.verdict) if cmp is not None else "-"],
        ["Difference of the WER with its interval", "", (f"{cmp.delta_wer:+.4f} ({_num(cmp.ci_low)} to {_num(cmp.ci_high)}, {cmp.n_groups} lessons, {cmp.confidence:.0%})" if cmp is not None and cmp.delta_wer is not None else "-")],
    ]
    out += _table(["", "Baseline", "Trained"], rows)
    if cmp is not None:
        out += f'<p class="small">{_e(cmp.explanation)}</p>'
    # what to change next
    lines = next_steps(verdict, run)
    box = "".join(f'<li lang="en">{_e(en)}</li><li class="rtl" dir="rtl" lang="he">{_e(he)}</li>' for en, he in lines)
    out += f'<div class="panel"><p><b>What to change next / מה לשנות בהמשך</b></p><ul>{box}</ul></div>'
    return out


def _clip_table(rows: Sequence[ClipRow]) -> str:
    body = [
        [_e(r.sample_id), f"{r.baseline_wer:.2f} → {r.trained_wer:.2f}", _rtl(r.reference), _rtl(r.baseline_text), _rtl(r.trained_text)]
        for r in rows
    ]
    return _table(["Clip", "WER base → trained", "Reference (label)", "Baseline wrote", "Trained wrote"], body)


def clips_html(data: ReportData) -> str:
    """The tables of the 15 worst and the 15 best Val clips (by the trained model's WER): reference, baseline output, trained output."""
    run = data.trained_run
    assert run is not None
    if not run.clips:
        return '<p class="na">Not available in this run: predictions_baseline.csv and predictions_trained.csv were not found in reports/.</p>'
    ordered = sorted(run.clips, key=lambda r: (r.trained_wer, r.sample_id))
    best = ordered[:CLIPS_SHOWN]
    worst = sorted(run.clips, key=lambda r: (-r.trained_wer, r.sample_id))[:CLIPS_SHOWN]
    return (
        _both(
            f"The {CLIPS_SHOWN} worst and the {CLIPS_SHOWN} best Val clips by the trained model's WER, with the label, what the untrained model wrote and what the trained model wrote. Read them to see what really changed.",
            f"{CLIPS_SHOWN} הקליפים הגרועים ו-{CLIPS_SHOWN} הטובים ביותר ב-Val לפי ה-WER של המודל המאומן: התווית, מה שכתב המודל הלא מאומן ומה שכתב המאומן. קראו כדי לראות מה השתנה באמת.",
        )
        + f"<h3>{CLIPS_SHOWN} worst / הגרועים ביותר</h3>"
        + _clip_table(worst)
        + f"<h3>{CLIPS_SHOWN} best / הטובים ביותר</h3>"
        + _clip_table(best)
    )


def final_test_html(view: FinalTestView) -> str:
    """The section "FINAL - Test, scored once" (only when test_result.json exists)."""
    english, hebrew, css = VERDICT_TEXT.get(view.verdict, VERDICT_TEXT["not_comparable"])
    status = "CONFIRMED" if view.status == "confirmed" else "NOT CONFIRMED"
    status_he = "אושר" if view.status == "confirmed" else "לא אושר"
    rows = [
        ["WER", _num(view.baseline.wer), _num(view.trained.wer)],
        ["CER", _num(view.baseline.cer), _num(view.trained.cer)],
        ["Relative reduction of the WER", "", _pct(view.relative_reduction)],
        ["Clips / words", f"{view.baseline.n_files} / {view.baseline.n_words}", f"{view.trained.n_files} / {view.trained.n_words}"],
        ["Bootstrap verdict", "", _e(view.verdict)],
        ["Difference of the WER with its interval", "", (f"{view.delta_wer:+.4f} ({_num(view.ci_low)} to {_num(view.ci_high)}, {view.n_groups} lessons, {view.confidence:.0%})" if view.delta_wer is not None else "-")],
    ]
    return (
        f'<p class="vline {css}" lang="en"><b>Test: {status}.</b> {_e(english)}</p>{_he(f"{status_he}. {hebrew}")}'
        + _table(["", "Baseline", "Trained"], rows)
        + f'<p class="small">{_e(view.explanation)}</p>'
        + _both(
            "The Test split was scored exactly once, after the choice was made. Do not use it to choose between experiments; a second look is refused by the program.",
            "קבוצת ה-Test נבדקה פעם אחת בדיוק, אחרי שההחלטה התקבלה. אין להשתמש בה כדי לבחור בין ניסויים; הפעלה שנייה נדחית על ידי התוכנה.",
        )
    )
