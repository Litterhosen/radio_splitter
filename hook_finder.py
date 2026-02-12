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


def find_hooks(
    wav_path,
    hook_len_range=(4.0, 15.0),
    prefer_len=8.0,
    hop_s=1.0,
    topn=10,
    min_gap_s=2.0,
):
    y, sr = librosa.load(wav_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)

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
        score = energy * 0.7 + loopability * 0.3

        tempo, _ = librosa.beat.beat_track(y=seg, sr=sr)

        windows.append(
            HookWindow(start, end, score, energy, loopability, stability, float(tempo))
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
