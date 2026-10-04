"""The audio adapter: reads headers and audio with soundfile, averages to mono and resamples with scipy.

How to run: this module is not a script; the pipeline creates the reader when none is given. Example
(PowerShell, project root; any audio file):
    $env:PYTHONPATH = "src"
    python -c "from pathlib import Path; from stt.featurepipeline.audio_adapter import SoundfileAudioReader; r = SoundfileAudioReader(); print(r.header(Path('clip.mp3')))"
Example output:
    AudioHeader(sample_rate=16000, channels=1, frames=480000)
This is the ONLY module of the component that imports soundfile and scipy (and numpy for arrays). Audio is never
truncated here; the input file is only read.
"""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile

from stt.featurepipeline.errors import AudioReadError
from stt.featurepipeline.types import AudioHeader, LoadedAudio


class SoundfileAudioReader:
    """AudioReader over soundfile (wav, mp3, flac, ogg) and scipy `resample_poly`."""

    def header(self, path: Path) -> AudioHeader:
        """Read the header only. Errors: AudioReadError for an unreadable file or zero frames."""
        try:
            info = soundfile.info(str(path))
        except Exception as exc:  # soundfile raises LibsndfileError, RuntimeError and OSError for bad files
            raise AudioReadError(f"cannot read the audio header of {path}: {exc}") from exc
        if info.frames <= 0 or info.samplerate <= 0:
            raise AudioReadError(f"{path} has no audio frames")
        return AudioHeader(int(info.samplerate), int(info.channels), int(info.frames))

    def load(self, path: Path, target_rate: int) -> LoadedAudio:
        """Decode `path` to mono float32 at `target_rate`. Errors: AudioReadError."""
        try:
            data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        except Exception as exc:
            raise AudioReadError(f"cannot decode {path}: {exc}") from exc
        downmixed = data.shape[1] > 1
        mono = data.mean(axis=1) if downmixed else data[:, 0]
        resampled = int(rate) != target_rate
        if resampled:
            divisor = gcd(target_rate, int(rate))
            mono = scipy.signal.resample_poly(mono, target_rate // divisor, int(rate) // divisor)
        return LoadedAudio(np.ascontiguousarray(mono, dtype=np.float32), resampled, downmixed)
