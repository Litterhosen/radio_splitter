import numpy as np
import librosa
from dataclasses import dataclass


@dataclass
class RefineResult:
    ok: bool
    start: float
    end: float
    bpm: float
    bars: int
    score: float
    reason: str


def refine_best_1_or_2_bars(
    wav_path,
    window_start,
    window_end,
    beats_per_bar=4,
    prefer_bars=1,
    sr=22050,
):
    y, sr = librosa.load(wav_path, sr=sr)
    y = y[int(window_start * sr): int(window_end * sr)]

    if len(y) < sr:
        return RefineResult(False, 0, 0, 0, 0, 0, "too short")

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    # Handle numpy array return from newer librosa
    if hasattr(tempo, '__len__'):
        tempo = tempo[0] if len(tempo) > 0 else 120.0
    bpm = int(round(float(tempo)))

    if len(beats) < beats_per_bar:
        return RefineResult(False, 0, 0, bpm, 0, 0, "not enough beats")

    bars = prefer_bars
    beats_needed = beats_per_bar * bars

    if len(beats) <= beats_needed:
        return RefineResult(False, 0, 0, bpm, 0, 0, "not enough beats for bars")

    start_frame = beats[0]
    end_frame = beats[beats_needed]

    start = librosa.frames_to_time(start_frame, sr=sr)
    end = librosa.frames_to_time(end_frame, sr=sr)

    score = float(bpm / 200.0)

    return RefineResult(True, start, end, bpm, bars, score, "")
