import numpy as np
import librosa
from dataclasses import dataclass


@dataclass
class HookWindow:
    start: float
    end: float
    score: float
    energy: float
    loopability: float
    stability: float
    bpm: float


def ffmpeg_to_wav16k_mono(in_path, out_path):
    import subprocess
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(in_path),
        "-ac", "1",
        "-ar", "16000",
        str(out_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def estimate_bpm(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)


def find_hooks(
    wav_path,
    hook_len_range=(4.0, 15.0),
    prefer_len=8.0,
    hop_s=1.0,
    topn=12,
    min_gap_s=2.0,
):
    y, sr = librosa.load(wav_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)

    min_len, max_len = hook_len_range
    win_len = prefer_len
    hop = hop_s

    windows = []
    t = 0.0

    while t + win_len < duration:
        start = t
        end = t + win_len
        s = int(start * sr)
        e = int(end * sr)
        seg = y[s:e]

        if len(seg) < sr:
            break

        energy = float(np.mean(np.abs(seg)))
        stability = float(np.std(seg))
        loopability = 1.0 / (1.0 + stability)

        score = energy * 0.6 + loopability * 0.4
        bpm = estimate_bpm(seg, sr)

        windows.append(
            HookWindow(start, end, score, energy, loopability, stability, bpm)
        )

        t += hop

    windows.sort(key=lambda w: w.score, reverse=True)

    selected = []
    for w in windows:
        if len(selected) >= topn:
            break
        if all(abs(w.start - s.start) > min_gap_s for s in selected):
            selected.append(w)

    return selected


def _window_features(y, sr, start, end):
    s = int(start * sr)
    e = int(end * sr)
    seg = y[s:e]
    if len(seg) < sr:
        return None
    energy = float(np.mean(np.abs(seg)))
    stability = float(np.std(seg))
    loopability = 1.0 / (1.0 + stability)
    bpm = estimate_bpm(seg, sr)
    return seg, energy, loopability, stability, bpm


def find_chorus_candidates(
    wav_path,
    window_len=8.0,
    hop_s=1.0,
    topn=6,
    min_gap_s=2.0,
    sim_threshold=0.85,
):
    y, sr = librosa.load(wav_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)

    win_len = max(2.0, float(window_len))
    hop = max(0.25, float(hop_s))

    windows = []
    t = 0.0
    while t + win_len < duration:
        feats = _window_features(y, sr, t, t + win_len)
        if feats is None:
            break
        seg, energy, loopability, stability, bpm = feats
        chroma = librosa.feature.chroma_stft(y=seg, sr=sr)
        vec = np.mean(chroma, axis=1)
        norm = np.linalg.norm(vec) or 1.0
        vec = vec / norm

        windows.append({
            "start": t,
            "end": t + win_len,
            "vec": vec,
            "energy": energy,
            "loopability": loopability,
            "stability": stability,
            "bpm": bpm,
        })
        t += hop

    if len(windows) < 2:
        return []

    # Chorus-aware score: max similarity to any other window
    for i, w in enumerate(windows):
        max_sim = 0.0
        for j, o in enumerate(windows):
            if i == j:
                continue
            sim = float(np.dot(w["vec"], o["vec"]))
            if sim > max_sim:
                max_sim = sim
        w["chorus_sim"] = max_sim

    ranked = [w for w in windows if w["chorus_sim"] >= sim_threshold]
    ranked.sort(key=lambda w: w["chorus_sim"], reverse=True)

    selected = []
    for w in ranked:
        if len(selected) >= topn:
            break
        if all(abs(w["start"] - s.start) > min_gap_s for s in selected):
            selected.append(
                HookWindow(
                    w["start"],
                    w["end"],
                    float(w["chorus_sim"]),
                    w["energy"],
                    w["loopability"],
                    w["stability"],
                    w["bpm"],
                )
            )

    return selected
