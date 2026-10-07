"""ReportBuilder: the results of a run as ONE self-contained HTML page (C14).

How to run: from Python (PowerShell, project root; the working example is in docs/COMPONENTS.md, C14):
    $env:PYTHONPATH = "src"
    python -c "from stt.reportbuilder import ReportBuilder; print(ReportBuilder)"
`ReportBuilder(registry).build(data: ReportData) -> ReportOutput`. The page opens offline in a browser: no network, no script, no font or image file; the charts are inline SVG made by `svg.py`. It starts with a
plain-language summary (the verdict, the baseline and winner WER with the interval, the Test confirmation), then the sections of the brief: (a) the finalists, (b) WER bars and the difference with its interval,
(c) training curves, (d) the distribution of per-file WER, (e) WER by group, (f) the search, (g) the learning curve, (h) correlations with file attributes, (i) diagnosis hints, (j) the listening list,
(k) run information. A section whose input is missing says `Not available in this run` and names the input; it is never an error and never an empty chart. The builder presents what it is given: it never
recomputes a WER, never changes a result, never decides, never reads the clock (the generation time is passed in) and never shows per-file Test results (the Test appears only as the one-time confirmation
of the winner). Text from the data is HTML-escaped; Hebrew cells are marked right-to-left; the page has a print style. The same inputs give byte-identical output.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Sequence
from html import escape
from pathlib import Path, PurePath
from urllib.parse import quote

from stt.reportbuilder import svg
from stt.reportbuilder.improvement import improvement_line, relative_reduction
from stt.reportbuilder.correlation import correlation_row, spearman, thin
from stt.reportbuilder.hints import RuleRegistry, default_rule_registry, group_errors
from stt.reportbuilder.types import CorrelationRow, DiagnosisHint, ReportData, ReportOutput

logger = logging.getLogger("stt.reportbuilder")

CSS = """
body{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#1a1a1a;line-height:1.45}
h1{font-size:26px;margin-bottom:4px}h2{font-size:20px;border-bottom:2px solid #0072B2;padding-bottom:3px;margin-top:34px}h3{font-size:16px;margin-top:20px}
.panel{background:#f3f7fb;border:1px solid #b9cde0;border-radius:8px;padding:14px 18px;margin:14px 0}
.verdict{font-size:22px;font-weight:bold;margin:0 0 8px}.win{color:#006b4f}.stay{color:#9a5a00}
.vline{font-size:19px;padding:10px 16px;border-radius:8px;margin:14px 0}.vline.proven{background:#e3f4ec;border:2px solid #006b4f;color:#064}.vline.unproven{background:#fff3e0;border:2px solid #E69F00;color:#6b3d00}.vline.retained{background:#eef0f3;border:2px solid #666;color:#333}.noise{color:#8a3b00}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}th,td{border:1px solid #c8c8c8;padding:4px 7px;text-align:left;vertical-align:top}th{background:#eef2f6}
td.num{text-align:right;font-variant-numeric:tabular-nums}.small{font-size:12.5px;color:#444}.na{color:#8a3b00;font-style:italic}
.hint{border-left:5px solid #E69F00;padding:6px 12px;margin:10px 0;background:#fffaf0}.hint.info{border-left-color:#56B4E9;background:#f4faff}
.rtl{direction:rtl;unicode-bidi:plaintext;text-align:right}code{background:#eef2f6;padding:0 3px;border-radius:3px}
svg{max-width:100%;height:auto;margin:6px 0}.charts{display:flex;flex-wrap:wrap;gap:8px}
@media print{body{max-width:none;margin:0}h2{page-break-after:avoid}table,svg,.panel,.hint{page-break-inside:avoid}a{color:inherit;text-decoration:none}}
"""


def _e(value: object) -> str:
    return escape(str(value))


def _num(value: float | None, digits: int = 4) -> str:
    return "-" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def _signed(value: float | None) -> str:
    return "-" if value is None else f"{value:+.4f}"


def _params(params: object) -> str:
    return ", ".join(f"{k} = {v:g}" if isinstance(v, int | float) else f"{k} = {v}" for k, v in dict(params).items()) or "none searched"  # type: ignore[call-overload]


def _duration(seconds: float) -> str:
    total = int(round(seconds))
    if total < 60:
        return f"{total} s"
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60} min"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], numeric: Sequence[int] = ()) -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f'<td class="num">{c}</td>' if i in numeric else f"<td>{c}</td>" for i, c in enumerate(row)) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _hebrew(text: str) -> str:
    return f'<span class="rtl" dir="rtl" lang="he">{_e(text)}</span>'


def relative_to_folder(path: Path, folder: str | None) -> str:
    """The audio path as shown: relative to the data `folder` when the file is inside it, else the path as stored. Pure string work on the path parts (no file system, no current folder), so the
    result is the same wherever the page is built: the folder may be given relative (`data/extracted/daf_yomi`) while the path is absolute, or the other way round."""
    if not folder:
        return path.as_posix()
    wanted = [os.path.normcase(part) for part in PurePath(folder).parts]
    parts = PurePath(path).parts
    names = [os.path.normcase(part) for part in parts]
    for start in range(len(names) - len(wanted)):
        if names[start : start + len(wanted)] == wanted:
            return "/".join(parts[start + len(wanted) :])
    return path.as_posix()


def _link(path: Path, shown: str | None = None) -> str:
    """The audio file as a link: the visible text is `shown` (a path relative to the data folder) or the full path; the link always points at the REAL file."""
    text = path.as_posix()
    target = ("file:///" + quote(text, safe="/:")) if (path.is_absolute() and ":" in text[:3]) else ("file://" + quote(text, safe="/:")) if path.is_absolute() else quote(text, safe="/:")
    return f'<a href="{_e(target)}">{_e(shown if shown is not None else text)}</a>'


class ReportBuilder:
    """Builds the report. The diagnosis rules come from a registry (by name)."""

    def __init__(self, registry: RuleRegistry | None = None) -> None:
        self._rules = registry or default_rule_registry()

    # ---- the page -------------------------------------------------------------------------------------------------------------
    def build(self, data: ReportData) -> ReportOutput:
        """The report of `data`. Errors: ReportConfigError for an unknown rule name in `report.hints.enabled_rules` (the registered names are listed). Everything else is shown or named as missing."""
        enabled = data.settings.hints.enabled_rules
        rules = [self._rules.create(name) for name in (enabled if enabled is not None else self._rules.names())]  # an unknown name fails here, before anything is built
        warnings: list[str] = []
        produced: list[str] = []
        sections: list[tuple[str, str, Callable[[], tuple[str, bool]]]] = [
            ("history", "History: the Val WER of each registered experiment", lambda: self._history(data)),
            ("finalists", "Finalists compared with the baseline", lambda: self._finalists(data)),
            ("wer_bars", "WER of the baseline and the finalists", lambda: self._wer_bars(data)),
            ("training_curves", "Training curves", lambda: self._curves(data, warnings)),
            ("distribution", "Distribution of the per-file WER", lambda: self._distribution(data)),
            ("groups", "WER by group (lesson)", lambda: self._groups(data, warnings)),
            ("search", "The search: each parameter against the Val loss", lambda: self._search(data)),
            ("learning_curve", "Learning curve: Train size against Val WER", lambda: self._learning_curve(data)),
            ("correlations", "Per-file errors and file attributes", lambda: self._correlations(data, warnings)),
            ("hints", "Diagnosis hints", lambda: self._hints(data, rules, warnings)),
            ("listening", "Listening list", lambda: self._listening(data)),
            ("run_info", "Run information", lambda: self._run_info(data)),
        ]
        body = []
        for key, title, make in sections:
            html, ok = make()
            body.append(f'<h2 id="{key}">{_e(title)}</h2>{html}')
            if ok:
                produced.append(key)
        summary = self._summary(data)
        produced.insert(0, "summary")
        missing = [title for key, title, _ in sections if key not in produced]
        gaps = f'<p class="small na">Sections without data in this run: {_e(", ".join(missing))}.</p>' if missing else ""
        title = data.settings.title or data.run_info.experiment_id
        page = (
            '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{_e(title)}</title><style>{CSS}</style></head><body>"
            f'<h1>{_e(title)}</h1><p class="small">Experiment {_e(data.run_info.experiment_id)} &middot; generated {_e(data.run_info.generated_at)} &middot; self-contained page, no network needed</p>'
            f"{summary}{gaps}{''.join(body)}</body></html>\n"
        )
        return ReportOutput(page, produced, warnings)

    # ---- helpers --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _na(what: str, needs: str) -> tuple[str, bool]:
        return f'<p class="na">Not available in this run: {_e(what)}. Needs: {_e(needs)}.</p>', False

    # ---- summary --------------------------------------------------------------------------------------------------------------
    def _summary(self, data: ReportData) -> str:
        sel, base = data.selection, data.baseline_val
        if sel is None:
            return '<div class="panel"><p class="verdict">No verdict in this run</p><p class="na">Not available in this run: the selection. Needs: reports/selection.json (the select_and_register stage).</p></div>'
        level = f"{sel.confidence:.0%}" if sel.confidence is not None else "the configured level"
        phrase, why, css = self._verdict_line(data)
        banner = f'<p class="vline {css}"><b>{_e(phrase)}</b> &mdash; {_e(why)}</p>'
        lines = []
        if sel.winner_trial is not None:
            row = next((r for r in sel.table if r.trial_id == sel.winner_trial), None)
            params = _e(_params(row.params) if row else "")
            who = "the winner" if phrase == "Proven improvement" else "the best finalist"  # the word winner is used only for a proven improvement
            head = (
                f'<p class="verdict win">WINNER: trial {sel.winner_trial} ({params})</p>' if phrase == "Proven improvement"
                else f'<p class="verdict stay">Best finalist: trial {sel.winner_trial} ({params}) (not proven better than the baseline)</p>'
            )
            if row is not None and base is not None:
                lines.append(f"On the Val set the baseline has a WER of <b>{_num(base.wer)}</b> and {who} <b>{_num(row.val_wer)}</b> (a difference of <b>{_signed(row.delta_wer)}</b>; positive means {who} is better).")
                if row.ci_low is not None and row.ci_high is not None:
                    lines.append(f"Interval of the difference at the {level} level (two one-sided bounds): <b>{_signed(row.ci_low)}</b> to <b>{_signed(row.ci_high)}</b>. Verdict against the baseline: <b>{_e(row.verdict)}</b>.")
        else:
            head = '<p class="verdict stay">THE BASELINE STAYS: no finalist beat it</p>'
            if base is not None:
                lines.append(f"The baseline model has a Val WER of <b>{_num(base.wer)}</b>.")
            if sel.table:
                best = sel.table[0]
                lines.append(f"The best finalist, trial {best.trial_id} ({_e(_params(best.params))}), reached {_num(best.val_wer)} (difference {_signed(best.delta_wer)}, verdict {_e(best.verdict)}).")
        lines.append(self._test_line(data, "winner" if phrase == "Proven improvement" else "best finalist"))
        flags = "".join(f'<p class="small">Flag: {_e(w)}</p>' for w in sel.warnings)
        best_row = next((r for r in sel.table if r.trial_id == sel.winner_trial), None) if sel.winner_trial is not None else (sel.table[0] if sel.table else None)
        improvement = improvement_line(base.wer if base is not None else None, best_row.val_wer if best_row is not None else None, data.settings.target_relative_wer_reduction, phrase == "Proven improvement")
        return f'{banner}<div class="panel">{head}<p>{" ".join(lines)}</p><p class="improve"><b>{_e(improvement)}</b></p><p>{_e(sel.reason)}</p>{flags}</div>'

    @staticmethod
    def _verdict_line(data: ReportData) -> tuple[str, str, str]:
        """(phrase, one sentence, style) for the line at the top. DERIVED ONLY from the stored verdict, the Val verdict of the winner's row and the Test status:
        `Baseline retained` when the baseline stays; `Proven improvement` when the winner is `improved` against the baseline on Val AND the one-time Test confirms it;
        `No proven improvement` in every other case (the Val verdict is not `improved`, or the Test does not confirm it, or it was not evaluated)."""
        sel = data.selection
        assert sel is not None
        if sel.winner_trial is None:
            return "Baseline retained", "No finalist beat the baseline on Val, so the baseline model stays.", "retained"
        row = next((r for r in sel.table if r.trial_id == sel.winner_trial), None)
        val_verdict = row.verdict if row is not None else "unknown"
        if val_verdict == "improved" and sel.test_status == "confirmed":
            return "Proven improvement", "The winner is clearly better than the baseline on Val and the one-time Test confirms it.", "proven"
        if val_verdict != "improved":
            why = f"The best finalist is not clearly better than the baseline on Val (verdict: {val_verdict})."
        elif sel.test_status == "not_confirmed":
            why = "The winner looks better on Val, but the one-time Test does not confirm it."
        else:
            why = "The winner looks better on Val, but the Test was not evaluated, so the improvement is not confirmed."
        return "No proven improvement", why, "unproven"

    def _noise_note(self, data: ReportData, values: Sequence[float | None]) -> str:
        """The note under a Val-loss chart whose plotted values vary by less than `report.loss_noise_range` (the axis stays automatic: it would otherwise stretch a tiny range to the full chart)."""
        finite = [v for v in values if v is not None and math.isfinite(v)]
        threshold = data.settings.loss_noise_range
        if len(finite) < 2 or max(finite) - min(finite) >= threshold:
            return ""
        return (
            f'<p class="small noise"><b>Differences this small are noise.</b> The Val loss in this chart varies by only {max(finite) - min(finite):.4g} '
            f"(less than report.loss_noise_range = {threshold:g}): the axis is stretched to the data, so do not read a ranking or a trend into the picture.</p>"
        )

    @staticmethod
    def _test_line(data: ReportData, who: str = "winner") -> str:
        sel = data.selection
        assert sel is not None
        if sel.test_status == "not_evaluated":
            why = f" ({_e(sel.test_error)})" if sel.test_error else " (the Test is looked at only for a winner)" if sel.winner_trial is None else ""
            return f"<b>Test confirmation:</b> the Test set was not evaluated{why}."
        base, win = data.test_results.get("baseline"), data.test_results.get(f"trial_{sel.winner_trial}")
        numbers = f" Test WER: baseline {_num(base.wer)}, {who} {_num(win.wer)}." if base is not None and win is not None else " (the stored Test results were not passed in)."
        word = "CONFIRMED by the Test" if sel.test_status == "confirmed" else "NOT confirmed by the Test (flagged, not dropped)"
        return f"<b>Test confirmation (looked at once):</b> {word}.{numbers}"

    # ---- (a) finalists and (b) bars -------------------------------------------------------------------------------------------
    def _finalists(self, data: ReportData) -> tuple[str, bool]:
        sel = data.selection
        if sel is None or not sel.table:
            return self._na("the comparison table", "reports/selection.json")
        rows = []
        if data.baseline_val is not None:
            rows.append(["baseline", "-", "-", _num(data.baseline_val.wer), _num(data.baseline_val.cer), "-", "-", "-", "-", "-", "-"])
        for r in sel.table:
            interval = "-" if r.ci_low is None or r.ci_high is None else f"{_signed(r.ci_low)} to {_signed(r.ci_high)}"
            rows.append([_e(r.rank), f"trial {r.trial_id}", _e(_params(r.params)), _num(r.val_wer), _num(r.val_cer), _signed(r.delta_wer), interval, _e(r.verdict), _num(r.val_loss), _e(r.steps), ("yes" if r.eligible else "no") + f" - {_e(r.note)}"])
        table = _table(["rank", "model", "parameters", "Val WER", "Val CER", "difference", "interval", "verdict", "Val loss", "steps", "eligible"], rows, numeric=(3, 4, 5, 8, 9))
        return table + '<p class="small">Difference = baseline WER minus finalist WER (positive: the finalist is better). The interval is the paired group bootstrap of the Evaluator, stored as it was computed.</p>', True

    def _wer_bars(self, data: ReportData) -> tuple[str, bool]:
        sel = data.selection
        if sel is None or not sel.table or data.baseline_val is None:
            return self._na("the WER bars", "reports/selection.json and the baseline result reports/eval_result_baseline.json")
        bars = [("baseline", data.baseline_val.wer, svg.GREY)] + [(f"trial {r.trial_id}", r.val_wer, svg.PALETTE[0] if r.trial_id == sel.winner_trial else svg.PALETTE[1]) for r in sel.table]
        forest = [(f"trial {r.trial_id}" + (" (winner)" if r.trial_id == sel.winner_trial else ""), r.delta_wer, r.ci_low, r.ci_high, svg.PALETTE[0] if r.trial_id == sel.winner_trial else svg.PALETTE[1]) for r in sel.table if r.delta_wer is not None]
        out = '<div class="charts">' + svg.bar_chart("Val WER: baseline and finalists", bars, "WER (lower is better)")
        if forest:
            out += svg.forest_plot("Difference from the baseline with its interval", forest, "WER difference (right of 0: better than the baseline)")
        return out + '</div><p class="small">The interval belongs to the DIFFERENCE with the baseline (two one-sided bounds); a finalist whose interval is entirely right of zero is clearly better than the baseline at that level.</p>', True

    # ---- (c) curves -----------------------------------------------------------------------------------------------------------
    def _curves(self, data: ReportData, warnings: list[str]) -> tuple[str, bool]:
        if not data.final_histories and not data.search_histories:
            return self._na("the training curves", "the metrics files of the trials (metrics/final_trial_TTT/metrics.csv, metrics/trial_TTT/metrics.csv)")
        cap = data.settings.max_points_per_chart
        winner = data.selection.winner_trial if data.selection is not None else None
        out = ""
        if data.final_histories:
            series = []
            for i, trial in enumerate(sorted(data.final_histories)):
                colour = svg.PALETTE[0] if trial == winner else svg.PALETTE[1 + i % 6]
                hist = data.final_histories[trial]
                val, thinned = thin([(float(m.step), float(m.val_loss)) for m in hist if m.val_loss is not None], cap)
                train, thinned2 = thin([(float(m.step), float(m.train_loss)) for m in hist if m.train_loss is not None], cap)
                if thinned or thinned2:
                    warnings.append(f"The curves of trial {trial} were thinned to {cap} points.")
                tag = f"trial {trial}" + (" (winner)" if trial == winner else "")
                series += [(f"{tag} Val", val, colour, False), (f"{tag} train", train, colour, True)]
            out += "<h3>Retrained finalists: Val loss (solid) and training loss (dashed)</h3>" + svg.line_chart("Loss against step, retrained finalists", series, "step", "loss")
            if not any(m.val_loss is not None for h in data.final_histories.values() for m in h):
                out += '<p class="na">These histories have no Val loss: only the training loss is drawn.</p>'
            out += self._noise_note(data, [m.val_loss for h in data.final_histories.values() for m in h])
        if data.search_histories:
            series = []
            for i, trial in enumerate(sorted(data.search_histories)):
                points, thinned = thin([(float(m.step), float(m.val_loss)) for m in data.search_histories[trial] if m.val_loss is not None], cap)
                if thinned:
                    warnings.append(f"The curve of search trial {trial} was thinned to {cap} points.")
                series.append((f"trial {trial}", points, svg.PALETTE[i % 8], False))
            out += "<h3>Search trials: Val loss</h3>" + svg.line_chart("Val loss against step, search trials", series, "step", "Val loss")
            out += self._noise_note(data, [m.val_loss for h in data.search_histories.values() for m in h])
        return out, True

    # ---- (d) distribution and (e) groups --------------------------------------------------------------------------------------
    def _model_label(self, data: ReportData) -> str:
        if data.selection is None or data.selection.winner_trial is None:
            return "the baseline"
        return "the winner" if self._verdict_line(data)[0] == "Proven improvement" else "the best finalist"  # the word winner only for a proven improvement

    def _distribution(self, data: ReportData) -> tuple[str, bool]:
        result = data.winner_val()
        if result is None or not result.files:
            return self._na("the per-file WER distribution", "the Val result of the winner or the baseline (reports/finalists/*.val.json, reports/eval_result_baseline.json)")
        values = [f.wer for f in result.files]
        shares = "".join(f"<li>files with a WER under {t:.0%}: <b>{s:.1%}</b></li>" for t, s in result.share_files_wer_below.items())
        out = svg.histogram(f"Per-file WER of {self._model_label(data)} on {result.split_name}", values, data.settings.histogram_bins, "WER of a file")
        return out + f"<ul>{shares}</ul><p class='small'>{len(values)} files; corpus WER {_num(result.wer)}, mean per-file WER {_num(result.wer_mean_per_file)}.</p>", True

    def _groups(self, data: ReportData, warnings: list[str]) -> tuple[str, bool]:
        result = data.winner_val()
        if result is None or not result.files:
            return self._na("the WER by group", "the Val result of the winner or the baseline")
        win = group_errors(result)
        base = group_errors(data.baseline_val) if data.baseline_val is not None and result is not data.baseline_val else None
        ordered = sorted(win, key=lambda g: (-(win[g][0] / win[g][1] if win[g][1] else 0.0), g))
        rows = []
        for g in ordered[: data.settings.groups_shown]:
            e, w = win[g]
            b = "-" if base is None or g not in base or not base[g][1] else _num(base[g][0] / base[g][1])
            rows.append([_e(g), _e(sum(1 for f in result.files if f.group_id == g)), _e(w), b, _num(e / w if w else None)])
        out = ""
        if base is not None:
            pts = [(base[g][0] / base[g][1], win[g][0] / win[g][1]) for g in sorted(win) if g in base and base[g][1] and win[g][1]]
            capped, thinned = thin(pts, data.settings.max_points_per_chart)
            if thinned:
                warnings.append(f"The group scatter was thinned to {data.settings.max_points_per_chart} points.")
            if capped:
                out += svg.scatter("WER by group: baseline against the winner (below the line = better)", capped, "baseline WER of the group", "winner WER of the group", diagonal=True)
        out += f"<p>The {min(len(ordered), data.settings.groups_shown)} worst groups of {len(ordered)} for {self._model_label(data)} (the group is the lesson or the speaker unit that is never split between Train and Val):</p>"
        return out + _table(["group", "files", "words", "baseline WER", "WER"], rows, numeric=(1, 2, 3, 4)), True

    # ---- (f) search and (g) learning curve ------------------------------------------------------------------------------------
    def _search(self, data: ReportData) -> tuple[str, bool]:
        s = data.search
        if s is None:
            return self._na("the search", "reports/search_trials.json")
        rows = [[_e(t.trial_id), _e(t.status), _e(_params(t.params)), _num(t.val_loss), _e(t.steps), f"{t.seconds:.0f}", _e(t.error or "")] for t in s.trials]
        out = f"<p>Search <b>{_e(s.strategy)}</b>, seed {_e(s.seed)}, {len(s.trials)} trials, stopped because: {_e(s.stopped_reason)}. Finalists: {_e(', '.join(f'trial {t}' for t in s.finalists) or 'none')}.</p>"
        out += _table(["trial", "status", "parameters", "Val loss", "steps", "seconds", "error"], rows, numeric=(3, 4, 5))
        complete = [t for t in s.trials if t.val_loss is not None and t.status in ("complete", "pruned")]
        corr = []
        out += '<div class="charts">'
        for name, spec in s.space.items():
            pts = [(float(t.params[name]), float(t.val_loss)) for t in complete if isinstance(t.params.get(name), int | float) and t.val_loss is not None]
            if not pts:
                continue
            rho = spearman([p[0] for p in pts], [p[1] for p in pts])
            note = "" if rho is not None else ("constant" if len(pts) >= 2 else "too few points")
            if rho is not None and len(pts) < data.settings.min_points_for_correlation:
                note = f"only {len(pts)} points (fewer than {data.settings.min_points_for_correlation}): indicative at most"
            corr.append([_e(name), _e(len(pts)), _num(rho, 3), _e(note)])
            chart = svg.scatter(f"Val loss against {name}", pts, name, "Val loss", log_x=spec.get("kind") == "log_float" and all(p[0] > 0 for p in pts))
            out += f"<div>{chart}{self._noise_note(data, [p[1] for p in pts])}</div>"
        out += "</div>"
        if corr:
            out += "<h3>Rank correlation of each parameter with the Val loss</h3>" + _table(["parameter", "points", "rank correlation", "note"], corr, numeric=(1, 2))
            out += '<p class="small">A rank correlation is not a cause: with few trials it can be produced by chance, and the parameters were not varied independently.</p>'
        if s.notes:
            out += "<ul>" + "".join(f"<li>{_e(n)}</li>" for n in s.notes) + "</ul>"
        return out, True

    def _learning_curve(self, data: ReportData) -> tuple[str, bool]:
        c = data.learning_curve
        if c is None or not c.points:
            return self._na("the learning curve", "reports/learning_curve.json (run python -m stt learning-curve)")
        pts = [(p.train_minutes, p.val_wer) for p in c.points]
        out = svg.line_chart("Val WER against the Train size", [("Val WER", pts, svg.PALETTE[0], False)], "minutes of Train audio", "Val WER")
        rows = [[_e(p.preset), _e(p.train_samples), f"{p.train_minutes:.1f}", _num(p.val_wer), _num(p.val_cer), _num(p.val_loss), _e(p.steps)] for p in c.points]
        out += _table(["preset", "Train samples", "minutes", "Val WER", "Val CER", "Val loss", "steps"], rows, numeric=(1, 2, 3, 4, 5, 6))
        return out + f"<p><b>Reading:</b> {_e(c.reading)}</p><p class='small'>Same hyperparameters ({_e(c.hyperparameters_from)}) and steps for every size, the same {_e(c.val_files)} Val files.</p>", True

    # ---- history ---------------------------------------------------------------------------------------------------------------
    def _history(self, data: ReportData) -> tuple[str, bool]:
        """The Val WER of the baseline and of the best model of each registered experiment, with the target when it is set (the numbers come from the stored cards; nothing is recomputed)."""
        points = list(data.history)
        if not points:
            return self._na("the history of the registered experiments", data.history_problem or "a registry with at least one entry (run python -m stt run)")
        target = data.settings.target_relative_wer_reduction
        base = [(float(i), p.baseline_wer) for i, p in enumerate(points, 1) if p.baseline_wer is not None]
        best = [(float(i), p.best_wer) for i, p in enumerate(points, 1) if p.best_wer is not None]
        series = [("baseline Val WER", base, svg.PALETTE[1], False), ("best model Val WER", best, svg.PALETTE[0], False)]
        if target is not None:
            series.append((f"target: {100.0 * target:g}% below the baseline", [(x, y * (1.0 - target)) for x, y in base], svg.PALETTE[2], True))
        out = svg.line_chart("Val WER of each registered experiment (the best model against its baseline)", series, "registered experiment (registry order)", "Val WER")
        headers = ["#", "entry", "experiment", "kind", "baseline Val WER", "best Val WER", "relative reduction"] + (["target Val WER", "target"] if target is not None else [])
        rows = []
        for i, p in enumerate(points, 1):
            reduction = relative_reduction(p.baseline_wer, p.best_wer)
            row = [_e(i), _e(p.entry_id), _e(p.experiment_id), "baseline kept" if p.kind != "model" else "model", _num(p.baseline_wer), _num(p.best_wer), "-" if reduction is None else f"{100.0 * reduction:.1f}%"]
            if target is not None:
                goal = None if p.baseline_wer is None else p.baseline_wer * (1.0 - target)
                row += [_num(goal), "-" if reduction is None else ("reached" if reduction >= target - 1e-12 else "not reached")]
            rows.append(row)
        out += _table(headers, rows, numeric=(0, 4, 5, 6, 7))
        note = (
            f"<p class='small'>{len(points)} registered experiment(s), from the stored model cards (nothing is recomputed). For `baseline kept` the best Val WER is the best finalist's: it did not beat the baseline. "
            + ("The dashed line is the target." if target is not None else "No target is set (report.target_relative_wer_reduction).")
            + (f" Not shown: {_e(data.history_problem)}." if data.history_problem else "")
            + "</p>"
        )
        return out + note, True

    # ---- (h) correlations -----------------------------------------------------------------------------------------------------
    def _correlations(self, data: ReportData, warnings: list[str]) -> tuple[str, bool]:
        result = data.winner_val()
        if result is None or not result.files:
            return self._na("the correlations", "the Val result of the winner or the baseline with its per-file attributes")
        rows: list[CorrelationRow] = []
        charts = ""
        shown = 0
        for name in data.settings.attributes:
            is_flag = name.startswith("flag_")
            pairs = []
            for f in result.files:
                value = f.attributes.get(name, 0 if is_flag else None)
                if isinstance(value, bool | int | float):
                    pairs.append((float(value), f.wer))
            if not pairs:
                rows.append(CorrelationRow(name, 0, None, "not present in this run"))
                continue
            rows.append(correlation_row(name, [p[0] for p in pairs], [p[1] for p in pairs], data.settings.min_points_for_correlation))
            if not is_flag and len({p[0] for p in pairs}) > 1 and shown < 3:
                capped, thinned = thin(pairs, data.settings.max_points_per_chart)
                if thinned:
                    warnings.append(f"The scatter of {name} was thinned to {data.settings.max_points_per_chart} points.")
                charts += svg.scatter(f"Per-file WER against {name}", capped, name, "WER of a file")
                shown += 1
        table = _table(["attribute", "points", "rank correlation", "note"], [[_e(r.attribute), _e(r.n), _num(r.rho, 3), _e(r.note)] for r in rows], numeric=(1, 2))
        text = f"<p>Rank correlation (Spearman) of the per-file WER of {self._model_label(data)} on {_e(result.split_name)} with each attribute. <b>A correlation is not a cause.</b> A value near 0 means no monotone relation; near +1 or -1 a strong one. The number of points is always shown.</p>"
        return text + table + f'<div class="charts">{charts}</div>', True

    # ---- (i) hints ------------------------------------------------------------------------------------------------------------
    def _hints(self, data: ReportData, rules: Sequence[object], warnings: list[str]) -> tuple[str, bool]:
        if not (data.final_histories or data.search_histories or data.winner_val() or data.search):
            return self._na("the diagnosis hints", "the training histories, the Val results or the search (nothing to look at)")
        hints: list[DiagnosisHint] = []
        for rule in rules:
            try:
                hint = rule.evaluate(data)  # type: ignore[attr-defined]
            except Exception as exc:  # a failing rule becomes a warning, the report is still made
                name = getattr(rule, "name", type(rule).__name__)
                logger.error("The diagnosis rule %s failed: %s: %s", name, type(exc).__name__, exc)
                warnings.append(f"The diagnosis rule {name} failed: {type(exc).__name__}: {exc}")
                hint = DiagnosisHint(name, "warning", f"The rule {name} could not be evaluated ({type(exc).__name__}).", {}, "The error is in the log; the other rules were not affected.")
            if hint is not None:
                hints.append(hint)
        if not hints:
            return "<p>No rule fired. That does not prove the run is healthy: the rules only look for a few known patterns, and a rule whose input is missing stays silent.</p>", True
        out = ""
        for h in hints:
            evidence = ", ".join(f"{_e(k)} = {_e(v)}" for k, v in h.evidence.items())
            out += f'<div class="hint {h.severity}"><b>{_e(h.rule)}</b> ({_e(h.severity)}): {_e(h.message)}<br><span class="small">Evidence: {evidence or "none"}</span><br><span class="small">Caveat: {_e(h.caveat)}</span></div>'
        return out + '<p class="small">These are hints from simple rules with configurable thresholds (report.hints); they are advice, not findings.</p>', True

    # ---- (j) listening list ---------------------------------------------------------------------------------------------------
    def _listening(self, data: ReportData) -> tuple[str, bool]:
        result = data.winner_val()
        if result is None or not result.review.entries:
            return self._na("the listening list", "the review list inside the Val result of the winner or the baseline")
        folder = None if data.settings.absolute_audio_paths else data.run_info.data_folder
        rows = []
        shortened = 0
        for e in result.review.entries:
            shown = relative_to_folder(e.audio_path, folder)
            shortened += shown != e.audio_path.as_posix()
            rows.append([_e(e.reason), _e(e.sample_id), _num(e.wer, 2), _hebrew(e.reference), _hebrew(e.hypothesis), _hebrew(e.diff), _link(e.audio_path, shown)])
        text = f"<p>Files of {self._model_label(data)} on {_e(result.split_name)} to listen to (worst first, then random ones). Differences: <code>[a-&gt;b]</code> replaced, <code>[-a-]</code> missing, <code>[+b+]</code> extra. No audio is embedded: open the file path.</p>"
        where = (
            f"<p class='small'>Audio paths are shown relative to the data folder <code>{_e(folder)}</code> (the links point at the real files; report.absolute_audio_paths = true shows the full paths).</p>" if folder and shortened
            else "<p class='small'>Audio paths are shown in full.</p>"
        )
        return text + _table(["why", "file", "WER", "reference", "model output", "differences", "audio file"], rows, numeric=(2,)) + where, True

    # ---- (k) run information --------------------------------------------------------------------------------------------------
    def _run_info(self, data: ReportData) -> tuple[str, bool]:
        r = data.run_info
        fields = [
            ("experiment", r.experiment_id), ("generated at", r.generated_at), ("configuration", r.config_ref or "unknown"), ("base model", r.base_model or "unknown"),
            ("registry entry", r.registry_entry or "none (nothing registered)"), ("data fingerprint", r.data_fingerprint or "-"), ("split fingerprint", r.split_fingerprint or "-"),
            ("subsample fingerprint", r.subsample_fingerprint or "-"), ("Test set fingerprint", r.test_fingerprint or "-"), ("code version", r.code_version or "unknown"), ("seed", r.seed if r.seed is not None else "unknown"),
        ]
        if r.search_seconds is not None:
            fields.append(("search time", _duration(r.search_seconds)))
        out = _table(["item", "value"], [[_e(k), _e(v)] for k, v in fields])
        looked = "unknown (the registry count was not passed in)" if r.test_evaluations is None else f"{r.test_evaluations} time(s) by registered entries, this one included (a lower bound)"
        out += f"<p><b>How often this Test set has been evaluated:</b> {_e(looked)}. Comparisons between experiments are made on Val; the Test is a one-time confirmation.</p>"
        if r.library_versions:
            out += "<p class='small'>Libraries: " + _e(", ".join(f"{k} {v}" for k, v in sorted(r.library_versions.items()))) + "</p>"
        b = data.benchmark
        if b is not None:
            parts = [[_e(label), _duration(seconds)] for label, seconds in b.parts] + [["<b>total</b>", f"<b>{_duration(b.total_seconds)}</b>"]]
            out += f"<h3>Planned time (benchmark on {_e(b.model)}, {b.threads} thread(s): {b.seconds_per_step:.2f} s per step, {b.seconds_per_val_loss_example:.3f} s per Val-loss example, {b.seconds_per_decode_clip:.2f} s per decoded clip)</h3>" + _table(["part of a run", "planned time"], parts)
        else:
            out += '<p class="na">Planned time not available in this run (needs reports/benchmark.json from python -m stt benchmark).</p>'
        return out, True

