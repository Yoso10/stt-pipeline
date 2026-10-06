"""Charts as inline SVG (pure functions, standard library only; no plotting library, no script, no font file).

How to run: this module is not a script; the report builder uses it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.reportbuilder.svg import bar_chart; print(bar_chart('WER', [('baseline', 0.5, '#666666'), ('trial 1', 0.3, '#0072B2')], 'WER')[:40])"
Example output:
    <svg xmlns="http://www.w3.org/2000/svg" vie
Every function returns one `<svg>` text with a `<title>` (the accessible name), axes, ticks, labels and a colour-blind-safe palette (Okabe and Ito). The same input always gives the same text. Text that
comes from the data is escaped. The callers decide what to draw (and never call a function with nothing to draw): the builder says "not available in this run" instead of an empty chart.
Charts: `line_chart` (curves), `bar_chart` (bars), `forest_plot` (a value with an interval per row), `scatter` (points, an optional diagonal, an optional log x axis), `histogram`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from html import escape

PALETTE = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000")
GREY = "#666666"
W, H = 640, 320
LEFT, RIGHT, TOP, BOTTOM = 64, 24, 40, 52


def fmt(value: float) -> str:
    """A compact number for a tick or a label."""
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 0.001:
        return f"{value:.3g}"
    return f"{value:.4g}"


def tick_labels(ticks: Sequence[float], transform: bool = False) -> list[str]:
    """Labels for round tick values with just enough decimals to tell neighbouring ticks apart (a narrow range such as 10.8522 to 10.8528 keeps its digits)."""
    if len(ticks) < 2:
        return [fmt(v) for v in ticks]
    step = abs(ticks[1] - ticks[0])
    decimals = 0 if step >= 1 else min(8, math.ceil(-math.log10(step)))
    if max(abs(v) for v in ticks) >= 1e5 or (0 < step < 1e-8):
        return [fmt(v) for v in ticks]
    return [f"{v:.{decimals}f}" for v in ticks]


def nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    """About `count` round tick values covering [lo, hi]."""
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(1, count)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = min((m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw), default=10 * magnitude)
    first = math.floor(lo / step) * step
    ticks = []
    value = first
    while value <= hi + step * 0.5 and len(ticks) < 50:
        ticks.append(round(value, 12))
        value += step
    inside = [v for v in ticks if lo - step * 1e-9 <= v <= hi + step * 1e-9]  # a tick outside the plotted range would hang below or beside the axes
    return inside if len(inside) >= 2 else ticks


def _scale(value: float, lo: float, hi: float, a: float, b: float) -> float:
    return a if hi == lo else a + (value - lo) / (hi - lo) * (b - a)


def _open(title: str, width: int = W, height: int = H) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{escape(title, quote=True)}" font-family="sans-serif" font-size="12">',
        f"<title>{escape(title)}</title>",
        f'<text x="{width / 2:.0f}" y="18" text-anchor="middle" font-size="14" font-weight="bold">{escape(title)}</text>',
    ]


def _axes(parts: list[str], xt: Sequence[float], yt: Sequence[float], xr: tuple[float, float], yr: tuple[float, float], x_label: str, y_label: str, x_fmt: bool = True, top: int = TOP, height: int = H) -> None:
    px = lambda v: _scale(v, xr[0], xr[1], LEFT, W - RIGHT)  # noqa: E731
    py = lambda v: _scale(v, yr[0], yr[1], height - BOTTOM, top)  # noqa: E731
    for t, label in zip(yt, tick_labels(yt), strict=True):
        parts.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{py(t):.1f}" y2="{py(t):.1f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{LEFT - 6}" y="{py(t) + 4:.1f}" text-anchor="end">{label}</text>')
    if x_fmt:
        for t, label in zip(xt, tick_labels(xt), strict=True):
            parts.append(f'<line x1="{px(t):.1f}" x2="{px(t):.1f}" y1="{height - BOTTOM}" y2="{height - BOTTOM + 4}" stroke="#333333"/>')
            parts.append(f'<text x="{px(t):.1f}" y="{height - BOTTOM + 17}" text-anchor="middle">{label}</text>')
    parts.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{height - BOTTOM}" y2="{height - BOTTOM}" stroke="#333333"/><line x1="{LEFT}" x2="{LEFT}" y1="{top}" y2="{height - BOTTOM}" stroke="#333333"/>')
    parts.append(f'<text x="{(LEFT + W - RIGHT) / 2:.0f}" y="{height - 8}" text-anchor="middle">{escape(x_label)}</text>')
    parts.append(f'<text x="14" y="{(top + height - BOTTOM) / 2:.0f}" text-anchor="middle" transform="rotate(-90 14 {(top + height - BOTTOM) / 2:.0f})">{escape(y_label)}</text>')


def _range(values: Sequence[float], pad: float = 0.05, zero: bool = False) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    if zero:
        lo = min(lo, 0.0)
    if hi == lo:
        hi, lo = hi + 0.5, lo - 0.5
    span = hi - lo
    return (lo if zero and lo == 0.0 else lo - span * pad), hi + span * pad


def line_chart(title: str, series: Sequence[tuple[str, Sequence[tuple[float, float]], str, bool]], x_label: str, y_label: str) -> str:
    """Curves. `series` items are (label, points, colour, dashed). Points are (x, y); a series with no point is skipped."""
    drawn = [s for s in series if s[1]]
    xs = [x for s in drawn for x, _ in s[1]]
    ys = [y for s in drawn for _, y in s[1]]
    xr, yr = _range(xs, 0.02), _range(ys)
    rows = (len(drawn) + 2) // 3
    height = H + 14 * max(0, rows - 1)
    parts = _open(title, W, height)
    _axes(parts, nice_ticks(*xr), nice_ticks(*yr), xr, yr, x_label, y_label, top=TOP + 14 * max(0, rows - 1), height=height)
    top = TOP + 14 * max(0, rows - 1)
    for i, (label, points, colour, dashed) in enumerate(drawn):
        coords = " ".join(f"{_scale(x, xr[0], xr[1], LEFT, W - RIGHT):.1f},{_scale(y, yr[0], yr[1], height - BOTTOM, top):.1f}" for x, y in points)
        dash = ' stroke-dasharray="5,3"' if dashed else ""
        parts.append(f'<polyline fill="none" stroke="{colour}" stroke-width="2"{dash} points="{coords}"/>')
        if len(points) <= 60:
            parts += [f'<circle cx="{_scale(x, xr[0], xr[1], LEFT, W - RIGHT):.1f}" cy="{_scale(y, yr[0], yr[1], height - BOTTOM, top):.1f}" r="2.5" fill="{colour}"/>' for x, y in points]
        parts.append(f'<rect x="{LEFT + 8 + 190 * (i % 3)}" y="{30 + 14 * (i // 3)}" width="10" height="3" fill="{colour}"/><text x="{LEFT + 22 + 190 * (i % 3)}" y="{34 + 14 * (i // 3)}" font-size="11">{escape(label[:30])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def bar_chart(title: str, bars: Sequence[tuple[str, float, str]], y_label: str) -> str:
    """Vertical bars (label, value, colour) from zero, with the value written above each bar."""
    yr = _range([v for _, v, _ in bars], 0.1, zero=True)
    parts = _open(title)
    _axes(parts, [], nice_ticks(*yr), (0, 1), yr, "", y_label, x_fmt=False)
    slot = (W - LEFT - RIGHT) / len(bars)
    for i, (label, value, colour) in enumerate(bars):
        x = LEFT + slot * i + slot * 0.15
        top, base = _scale(value, yr[0], yr[1], H - BOTTOM, TOP), _scale(0, yr[0], yr[1], H - BOTTOM, TOP)
        parts.append(f'<rect x="{x:.1f}" y="{min(top, base):.1f}" width="{slot * 0.7:.1f}" height="{abs(base - top):.1f}" fill="{colour}"/>')
        parts.append(f'<text x="{x + slot * 0.35:.1f}" y="{min(top, base) - 4:.1f}" text-anchor="middle">{fmt(value)}</text>')
        parts.append(f'<text x="{x + slot * 0.35:.1f}" y="{H - BOTTOM + 17}" text-anchor="middle" font-size="11">{escape(label[:14])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def forest_plot(title: str, rows: Sequence[tuple[str, float, float | None, float | None, str]], x_label: str) -> str:
    """A value with an interval per row, horizontally: (label, estimate, low, high, colour). A row without bounds shows the dot only. The vertical line is zero."""
    values = [v for _, e, lo, hi, _ in rows for v in (e, lo, hi) if v is not None] + [0.0]
    xr = _range(values, 0.08)
    height = max(H // 2, 70 + 26 * len(rows))
    parts = _open(title, W, height)
    px = lambda v: _scale(v, xr[0], xr[1], 150, W - RIGHT)  # noqa: E731
    for t in nice_ticks(*xr):
        parts.append(f'<line x1="{px(t):.1f}" x2="{px(t):.1f}" y1="{TOP}" y2="{height - 40}" stroke="#eeeeee"/><text x="{px(t):.1f}" y="{height - 24}" text-anchor="middle">{fmt(t)}</text>')
    parts.append(f'<line x1="{px(0):.1f}" x2="{px(0):.1f}" y1="{TOP}" y2="{height - 40}" stroke="#333333" stroke-width="1.5"/>')
    for i, (label, estimate, lo, hi, colour) in enumerate(rows):
        y = TOP + 18 + 26 * i
        parts.append(f'<text x="144" y="{y + 4}" text-anchor="end">{escape(label[:22])}</text>')
        if lo is not None and hi is not None:
            parts.append(f'<line x1="{px(lo):.1f}" x2="{px(hi):.1f}" y1="{y}" y2="{y}" stroke="{colour}" stroke-width="3"/><line x1="{px(lo):.1f}" x2="{px(lo):.1f}" y1="{y - 5}" y2="{y + 5}" stroke="{colour}" stroke-width="2"/><line x1="{px(hi):.1f}" x2="{px(hi):.1f}" y1="{y - 5}" y2="{y + 5}" stroke="{colour}" stroke-width="2"/>')
        parts.append(f'<circle cx="{px(estimate):.1f}" cy="{y}" r="4.5" fill="{colour}"/>')
    parts.append(f'<text x="{(150 + W - RIGHT) / 2:.0f}" y="{height - 6}" text-anchor="middle">{escape(x_label)}</text></svg>')
    return "".join(parts)


def scatter(title: str, points: Sequence[tuple[float, float]], x_label: str, y_label: str, diagonal: bool = False, log_x: bool = False, colour: str = PALETTE[0]) -> str:
    """Points (x, y). `diagonal` draws y = x; `log_x` puts x on a log scale (all x above 0)."""
    xv = [math.log10(x) if log_x else x for x, _ in points]
    yv = [y for _, y in points]
    if diagonal:
        lo, hi = min(xv + yv), max(xv + yv)
        xr = yr = _range([lo, hi], 0.05)
    else:
        xr, yr = _range(xv, 0.08), _range(yv, 0.08)
    parts = _open(title)
    xt = nice_ticks(*xr)
    _axes(parts, xt, nice_ticks(*yr), xr, yr, x_label, y_label, x_fmt=False)
    for t in xt:
        label = fmt(10**t) if log_x else fmt(t)
        parts.append(f'<line x1="{_scale(t, xr[0], xr[1], LEFT, W - RIGHT):.1f}" x2="{_scale(t, xr[0], xr[1], LEFT, W - RIGHT):.1f}" y1="{H - BOTTOM}" y2="{H - BOTTOM + 4}" stroke="#333333"/><text x="{_scale(t, xr[0], xr[1], LEFT, W - RIGHT):.1f}" y="{H - BOTTOM + 17}" text-anchor="middle">{label}</text>')
    if diagonal:
        parts.append(f'<line x1="{LEFT}" y1="{H - BOTTOM}" x2="{W - RIGHT}" y2="{TOP}" stroke="{GREY}" stroke-dasharray="4,3"/>')
    for x, y in zip(xv, yv, strict=True):
        parts.append(f'<circle cx="{_scale(x, xr[0], xr[1], LEFT, W - RIGHT):.1f}" cy="{_scale(y, yr[0], yr[1], H - BOTTOM, TOP):.1f}" r="3" fill="{colour}" fill-opacity="0.7"/>')
    parts.append("</svg>")
    return "".join(parts)


def histogram(title: str, values: Sequence[float], bins: int, x_label: str, colour: str = PALETTE[0]) -> str:
    """A histogram of the values (counts per bin, equal-width bins over the range of the values)."""
    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + 1.0
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, int((v - lo) / (hi - lo) * bins))] += 1
    yr = (0.0, max(counts) * 1.1)
    xr = (lo, hi)
    parts = _open(title)
    _axes(parts, nice_ticks(*xr), nice_ticks(*yr), xr, yr, x_label, "files")
    width = (W - LEFT - RIGHT) / bins
    for i, count in enumerate(counts):
        top = _scale(count, yr[0], yr[1], H - BOTTOM, TOP)
        parts.append(f'<rect x="{LEFT + i * width:.1f}" y="{top:.1f}" width="{width - 1:.1f}" height="{H - BOTTOM - top:.1f}" fill="{colour}"/>')
    parts.append("</svg>")
    return "".join(parts)
