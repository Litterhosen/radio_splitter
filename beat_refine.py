from dataclasses import dataclass
from typing import Optional

import numpy as np
import librosa


@dataclass
class RefineResult:
    ok: bool
    start: float
    end: float
    bpm: float
    bars: int
    score: float
    reason: str = ""


def _boundary_similarity(y, sr, t1, t2) -> float:
    # compare small slices around boundaries
    w = int(0.15 * sr)
    i1 = int(max(0, t1 * sr))
    i2 = int(max(0, t2 * sr))

    a = y[i1:i1 + w]
    b = y[max(0, i2 - w):i2]
    if len(a) < w or len(b) < w:
        return 0.0
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.mean(a * b))


def refine_best_1_or_2_bars(
    wav_path: str,
    window_start: float,
    window_end: float,
    beats_per_bar: int = 4,
    prefer_bars: int = 1,
    sr: int = 22050,
) -> RefineResult:
    try:
        y, sr = librosa.load(wav_path, sr=sr, mono=True)
        dur = len(y) / sr

        ws = max(0.0, float(window_start))
        we = min(dur, float(window_end))
        if (we - ws) < 2.0:
            return RefineResult(False, ws, we, 0.0, 0, 0.0, "window_too_short")

        # tempo + beats
        y_seg = y[int(ws * sr): int(we * sr)]
        if len(y_seg) < sr:
            return RefineResult(False, ws, we, 0.0, 0, 0.0, "segment_too_short")

        tempo, beats = librosa.beat.beat_track(y=y_seg, sr=sr)
        bpm = float(tempo or 0.0)
        if bpm <= 40 or bpm >= 220:
            # still usable, but suspicious
            pass

        beat_times = librosa.frames_to_time(beats, sr=sr)
        beat_times = beat_times + ws

        if len(beat_times) < (beats_per_bar * 2):
            return RefineResult(False, ws, we, bpm, 0, 0.0, "not_enough_beats")

        # Candidate loops: 1 bar or 2 bars (or prefer)
        candidates = []
        for bars in [1, 2]:
            beats_needed = bars * int(beats_per_bar)
            for i in range(0, len(beat_times) - beats_needed):
                a = float(beat_times[i])
                b = float(beat_times[i + beats_needed])
                if a < ws or b > we:
                    continue

                # score: boundary similarity + energy
                sim = _boundary_similarity(y, sr, a, b)
                seg = y[int(a * sr): int(b * sr)]
                e = float(np.mean(seg ** 2))
                score = (2.0 * sim) + (0.5 * np.tanh(10 * e))

                # gentle bias for preferred bars
                if bars == int(prefer_bars):
                    score += 0.15

                candidates.append((score, a, b, bars))

        if not candidates:
            return RefineResult(False, ws, we, bpm, 0, 0.0, "no_candidates")

        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0]
        return RefineResult(True, best[1], best[2], bpm, int(best[3]), float(best[0]), "")

    except Exception as e:
        return RefineResult(False, window_start, window_end, 0.0, 0, 0.0, f"error:{e}")
