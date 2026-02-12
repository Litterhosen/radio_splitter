from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# librosa kan nogle gange give import-problemer på cloud.
# Vi importerer den "rent", men giver en tydelig fejl hvis den fejler.
try:
    import librosa
except Exception as e:
    raise RuntimeError(
        "Kunne ikke importere librosa. Tjek at requirements.txt indeholder librosa==0.10.2.post1 "
        "og at Streamlit Cloud har rebuildet."
    ) from e


@dataclass
class Window:
    start: float
    end: float
    score: float
    energy: float
    loopability: float
    stability: float
    bpm: float = 0.0


def _features(y: np.ndarray, sr: int):
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop).flatten()
    flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    return chroma, rms, flux, hop


def _estimate_bpm(y: np.ndarray, sr: int) -> float:
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo or 0.0)
    except Exception:
        return 0.0


def _window_score(chroma, rms, flux, sr, hop, start_s, end_s):
    a = int(start_s * sr / hop)
    b = int(end_s * sr / hop)
    b = max(a + 2, b)

    X = chroma[:, a:b]
    if X.shape[1] < 4:
        return 0.0, 0.0, 0.0, 0.0

    Xn = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-9)
    S = Xn.T @ Xn
    rep = float(np.mean(S))

    e = float(np.mean(rms[a:b])) if b <= len(rms) else float(np.mean(rms[a:])) if a < len(rms) else 0.0

    loop = float(np.mean(Xn[:, 0] * Xn[:, -1]))

    fl = flux[a:b] if b <= len(flux) else flux[a:] if a < len(flux) else np.array([0.0])
    stab = float(1.0 / (1.0 + np.mean(fl)))

    score = (2.0 * rep) + (1.2 * loop) + (1.0 * stab) + (0.5 * np.tanh(5 * e))
    return score, e, loop, stab


def find_hooks(
    wav16k: Path,
    hook_len_range: Tuple[float, float] = (4.0, 15.0),
    prefer_len: float = 8.0,
    hop_s: float = 1.0,
    topn: int = 12,
    min_gap_s: float = 2.0,
) -> List[Window]:
    """
    Hook finder (4-15s) via repetition+loopability+stability scoring.
    """
    y, sr = librosa.load(str(wav16k), sr=16000, mono=True)
    bpm = _estimate_bpm(y, sr)

    chroma, rms, flux, hop = _features(y, sr)
    dur = len(y) / sr

    lo, hi = hook_len_range
    lo = max(2.0, float(lo))
    hi = max(lo, float(hi))
    prefer_len = float(prefer_len)

    candidates: List[Window] = []
    t = 0.0
    while t + lo <= dur:
        for L in [prefer_len, lo, hi]:
            end = min(dur, t + L)
            if (end - t) < lo:
                continue
            sc, e, loop, stab = _window_score(chroma, rms, flux, sr, hop, t, end)
            candidates.append(Window(start=t, end=end, score=sc, energy=e, loopability=loop, stability=stab, bpm=bpm))
        t += max(0.25, float(hop_s))

    candidates.sort(key=lambda w: w.score, reverse=True)

    picked: List[Window] = []
    for w in candidates:
        if len(picked) >= int(topn):
            break
        if all(abs(w.start - p.start) >= float(min_gap_s) for p in picked):
            picked.append(w)
    return picked


def find_chorus_windows(
    wav16k: Path,
    chorus_len_range: Tuple[float, float] = (30.0, 45.0),
    hop_s: float = 2.0,
    topn: int = 4,
    min_gap_s: float = 8.0,
) -> List[Window]:
    """
    Chorus-aware: find longer repetition-rich windows (30-45s)
    """
    return find_hooks(
        wav16k=wav16k,
        hook_len_range=chorus_len_range,
        prefer_len=float(sum(chorus_len_range) / 2.0),
        hop_s=hop_s,
        topn=topn,
        min_gap_s=min_gap_s,
    )


def refine_loops_within_window(
    wav16k: Path,
    chorus_window: Window,
    hook_len_range: Tuple[float, float] = (4.0, 15.0),
    prefer_len: float = 8.0,
    hop_s: float = 0.5,
    topn: int = 10,
    min_gap_s: float = 1.0,
) -> List[Window]:
    """
    Find hooks inside a chorus window (sub-window search).
    """
    y, sr = librosa.load(str(wav16k), sr=16000, mono=True)
    bpm = _estimate_bpm(y, sr)

    chroma, rms, flux, hop = _features(y, sr)
    dur = len(y) / sr

    a0 = max(0.0, float(chorus_window.start))
    b0 = min(dur, float(chorus_window.end))

    lo, hi = hook_len_range
    lo = max(2.0, float(lo))
    hi = max(lo, float(hi))

    candidates: List[Window] = []
    t = a0
    while t + lo <= b0:
        for L in [float(prefer_len), lo, hi]:
            end = min(b0, t + L)
            if (end - t) < lo:
                continue
            sc, e, loop, stab = _window_score(chroma, rms, flux, sr, hop, t, end)
            candidates.append(Window(start=t, end=end, score=sc, energy=e, loopability=loop, stability=stab, bpm=bpm))
        t += max(0.25, float(hop_s))

    candidates.sort(key=lambda w: w.score, reverse=True)

    picked: List[Window] = []
    for w in candidates:
        if len(picked) >= int(topn):
            break
        if all(abs(w.start - p.start) >= float(min_gap_s) for p in picked):
            picked.append(w)
    return picked


def beat_aligned_windows(
    wav16k: Path,
    len_range: Tuple[float, float] = (4.0, 15.0),
    topn: int = 12,
    min_gap_s: float = 2.0,
    beats_per_bar: int = 4,
) -> List[Window]:
    """
    Beat-aware finder: candidate windows start/end on beat grid (cleaner loops).
    """
    y, sr = librosa.load(str(wav16k), sr=16000, mono=True)
    dur = len(y) / sr

    bpm = _estimate_bpm(y, sr)

    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        bt = librosa.frames_to_time(beats, sr=sr)
        beat_times = [float(x) for x in bt if 0.0 <= x <= dur]
    except Exception:
        beat_times = []

    if len(beat_times) < 8:
        # fallback til score-based finder
        return find_hooks(wav16k, hook_len_range=len_range, topn=topn)

    lo, hi = float(len_range[0]), float(len_range[1])
    candidates: List[Window] = []

    # simple energy
    rms = librosa.feature.rms(y=y).flatten()
    rms_mean = float(np.mean(rms)) if len(rms) else 0.0

    for i in range(len(beat_times) - 2):
        start = beat_times[i]
        for j in range(i + 2, len(beat_times)):
            end = beat_times[j]
            d = end - start
            if d < lo:
                continue
            if d > hi:
                break

            # score: prefer strong energy + end/start similarity via chroma scoring
            # We reuse short scoring on this window
            chroma, rms2, flux, hop = _features(y, sr)
            sc, e, loop, stab = _window_score(chroma, rms2, flux, sr, hop, start, end)
            sc += 0.25 * np.tanh(10 * (e + rms_mean))

            candidates.append(Window(start=start, end=end, score=sc, energy=e, loopability=loop, stability=stab, bpm=bpm))

    candidates.sort(key=lambda w: w.score, reverse=True)

    picked: List[Window] = []
    for w in candidates:
        if len(picked) >= int(topn):
            break
        if all(abs(w.start - p.start) >= float(min_gap_s) for p in picked):
            picked.append(w)
    return picked
