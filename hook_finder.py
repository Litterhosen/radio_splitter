import numpy as np
import librosa
from dataclasses import dataclass
from utils import find_ffmpeg


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
    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-i", str(in_path),
        "-ac", "1",
        "-ar", "16000",
        str(out_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def estimate_bpm(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # Handle numpy array return from newer librosa
    if hasattr(tempo, '__len__'):
        tempo = tempo[0] if len(tempo) > 0 else 120.0
    return int(round(float(tempo)))


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
    # Clamp prefer_len to the specified range
    win_len = max(min_len, min(prefer_len, max_len))
    hop = hop_s

    windows = []
    
    # Scan with multiple window sizes within the range for better detection
    window_sizes = [win_len]
    if max_len > min_len:
        # Add min and max if they're different from prefer_len
        if abs(min_len - win_len) > 1.0:
            window_sizes.append(min_len)
        if abs(max_len - win_len) > 1.0:
            window_sizes.append(max_len)
    
    for current_win_len in window_sizes:
        t = 0.0
        while t + current_win_len < duration:
            start = t
            end = t + current_win_len
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
