from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import librosa

from utils import find_ffmpeg, run_cmd


@dataclass
class Window:
    start: float
    end: float
    score: float
    energy: float
    loopability: float
    stability: float


def ffmpeg_to_wav16k_mono(src_path: Path, out_wav: Path):
    ffmpeg = find_ffmpeg()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    run_cmd([
        ffmpeg, "-y",
        "-i", str(src_path),
        "-ac", "1", "-ar", "16000",
        str(out_wav)
    ])


def _features(y: np.ndarray, sr: int):
    hop = 512
    # chroma is good for repetition (harmonic similarity)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop).flatten()
    flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    return chroma, rms, flux, hop


def _window_score(chroma, rms, flux, sr, hop, start_s, end_s):
    a = int(start_s * sr / hop)
    b = int(end_s * sr / hop)
    b = max(a + 2, b)

    X = chroma[:, a:b]
    if X.shape[1] < 4:
        return 0.0, 0.0, 0.0, 0.0

    # repetition: self-similarity average (cosine)
    Xn = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-9)
    S = Xn.T @ Xn
    rep = float(np.mean(S))

    # energy: mean rms
    e = float(np.mean(rms[a:b])) if b <= len(rms) else float(np.mean(rms[a:])) if a < len(rms) else 0.0

    # loopability: start/end chroma similarity
    loop = float(np.mean(Xn[:, 0] * Xn[:, -1]))

    # stability: low average flux is more “loopable”
    fl = flux[a:b] if b <= len(flux) else flux[a:] if a < len(flux) else np.array([0.0])
    stab = float(1.0 / (1.0 + np.mean(fl)))  # smaller flux => higher stab

    # combined score
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
    y, sr = librosa.load(str(wav16k), sr=16000, mono=True)
    chroma, rms, flux, hop = _features(y, sr)
    dur = len(y) / sr

    lo, hi = hook_len_range
    lo = max(2.0, float(lo))
    hi = max(lo, float(hi))
    prefer_len = float(prefer_len)

    candidates: List[Window] = []
    t = 0.0
    while t + lo <= dur:
        # try a few lengths around prefer_len for better fit
        for L in [prefer_len, lo, hi]:
            end = min(dur, t + L)
            if (end - t) < lo:
                continue
            sc, e, loop, stab = _window_score(chroma, rms, flux, sr, hop, t, end)
            candidates.append(Window(start=t, end=end, score=sc, energy=e, loopability=loop, stability=stab))
        t += max(0.25, float(hop_s))

    # sort best first
    candidates.sort(key=lambda w: w.score, reverse=True)

    # greedy pick with min gap
    picked: List[Window] = []
    for w in candidates:
        if len(picked) >= int(topn):
            break
        ok = True
        for p in picked:
            if abs(w.start - p.start) < min_gap_s:
                ok = False
                break
        if ok:
            picked.append(w)

    return picked


def find_chorus_windows(
    wav16k: Path,
    chorus_len_range: Tuple[float, float] = (30.0, 45.0),
    hop_s: float = 2.0,
    topn: int = 4,
    min_gap_s: float = 8.0,
) -> List[Window]:
    # same scoring system, just longer windows
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
    hook_len_range=(4.0, 15.0),
    prefer_len=8.0,
    hop_s=0.5,
    topn=10,
    min_gap_s=1.0,
) -> List[Window]:
    # Load once, then only score inside the chorus range
    y, sr = librosa.load(str(wav16k), sr=16000, mono=True)
    chroma, rms, flux, hop = _features(y, sr)

    dur = len(y) / sr
    lo, hi = hook_len_range
    lo = max(2.0, float(lo))
    hi = max(lo, float(hi))
    prefer_len = float(prefer_len)

    a0 = max(0.0, float(chorus_window.start))
    b0 = min(dur, float(chorus_window.end))
    if (b0 - a0) < lo:
        return []

    candidates: List[Window] = []
    t = a0
    while t + lo <= b0:
        for L in [prefer_len, lo, hi]:
            end = min(b0, t + L)
            if (end - t) < lo:
                continue
            sc, e, loop, stab = _window_score(chroma, rms, flux, sr, hop, t, end)
            candidates.append(Window(start=t, end=end, score=sc, energy=e, loopability=loop, stability=stab))
        t += max(0.25, float(hop_s))

    candidates.sort(key=lambda w: w.score, reverse=True)

    picked: List[Window] = []
    for w in candidates:
        if len(picked) >= int(topn):
            break
        ok = True
        for p in picked:
            if abs(w.start - p.start) < min_gap_s:
                ok = False
                break
        if ok:
            picked.append(w)
    return picked
