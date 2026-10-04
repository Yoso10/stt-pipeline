r"""Parse timestamp tokens into segments (before they are stripped, so no information is lost).

How to run: this module is not a script. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "import re; from stt.textnormalizer.timestamps import parse_segments; print(parse_segments('<|0.5|>שלום <|2.0|><|2.5|>' , re.compile(r'<\|(\d+(?:\.\d+)?)\|>')))"
Example output:
    (True, (TimestampSegment(start_sec=0.5, end_sec=2.0, text='שלום '),))
Format: `<|start|>text<|end|>` repeated; the next `<|start|>` follows the previous end directly. A last
token with no text after it (the start of a segment that was cut off) is normal and is ignored.
"""

from __future__ import annotations

import re

from stt.textnormalizer.types import TimestampSegment


def parse_segments(text: str, pattern: re.Pattern[str]) -> tuple[bool, tuple[TimestampSegment, ...] | None]:
    """Return `(had_tokens, segments)`.

    `segments` is None when there are no tokens (had_tokens False) or when they are malformed:
    text before the first token, text after a last START token with no end, text between an END
    and the next START, an end before its start, or a start before the previous end.
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return False, None
    if text[: matches[0].start()].strip():
        return True, None
    times = [float(m.group(1)) for m in matches]
    segments: list[TimestampSegment] = []
    previous_end = 0.0
    index = 0
    while index + 1 < len(matches):
        start, end = times[index], times[index + 1]
        if end < start or start < previous_end:
            return True, None
        segments.append(TimestampSegment(start, end, text[matches[index].end() : matches[index + 1].start()]))
        previous_end = end
        index += 2
        if index < len(matches):
            gap = text[matches[index - 1].end() : matches[index].start()]
            if gap.strip():
                return True, None
    if text[matches[-1].end() :].strip():  # a start token with text but no end, or text after the last end
        return True, None
    return True, tuple(segments)
