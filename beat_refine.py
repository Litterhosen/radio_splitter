from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    import librosa
except Exception as e:
    raise RuntimeError(
        "Kunne ikke importere librosa i beat_refine.py. Tjek requirements og rebuild."
    ) from e


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
    w = int(0.12 * sr)
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

        y_seg = y[int(ws * sr): int(we * sr)]
        if len(y_seg) < sr:
            return RefineResult(False, ws, we, 0.0, 0, 0.0, "segment_too_short")

        tempo, beats = librosa.beat.beat_track(y=y_seg, sr=sr)
        bpm = float(tempo or 0.0)
        beat_times = librosa.frames_to_time(beats, sr=sr) + ws

        if len(beat_times) < beats_per_bar * 2:
            return RefineResult(False, ws, we, bpm, 0, 0.0, "not_enough_beats")

        candidates = []
        for bars in (1, 2):
            beats_needed = bars * int(beats_per_bar)
            for i in range(0, len(beat_times) - beats_needed):
                a = float(beat_times[i])
                b = float(beat_times[i + beats_needed])
                if a < ws or b > we:
                    continue

                sim = _boundary_similarity(y, sr, a, b)
                seg = y[int(a * sr): int(b * sr)]
                e = float(np.mean(seg ** 2))
                score = (2.2 * sim) + (0.6 * np.tanh(12 * e))
                if bars == int(prefer_bars):
                    score += 0.15
                candidates.append((score, a, b, bars))

        if not candidates:
            return RefineResult(False, ws, we, bpm, 0, 0.0, "no_candidates")

        candidates.sort(key=lambda x: x[0], reverse=True)
        score, a, b, bars = candidates[0]
        return RefineResult(True, a, b, bpm, int(bars), float(score), "")

    except Exception as e:
        return RefineResult(False, window_start, window_end, 0.0, 0, 0.0, f"error:{e}")
