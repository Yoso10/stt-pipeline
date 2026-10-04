"""Read the duration of an audio file from its header (no audio array is loaded).

How to run: this module is not a script. Example (PowerShell, project root, any audio file):
    $env:PYTHONPATH = "src"
    python -c "from pathlib import Path; from stt.datasource.audio_header import read_duration_seconds; print(read_duration_seconds(Path('clip.mp3')))"
Example output:
    30.0
Raises ValueError for an unreadable file or a zero-length audio.
"""

from __future__ import annotations

from pathlib import Path

import soundfile


def read_duration_seconds(path: Path) -> float:
    """Return the duration in seconds from the header of `path` (wav, mp3, flac, ogg).

    Errors: ValueError when the file cannot be opened as audio or has no frames.
    """
    try:
        info = soundfile.info(str(path))
    except Exception as exc:  # soundfile raises LibsndfileError / RuntimeError for bad files
        raise ValueError(f"cannot read audio header: {exc}") from exc
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError("audio has zero length")
    return float(info.frames) / float(info.samplerate)
