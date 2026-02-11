import librosa
import numpy as np


def beat_aligned_hooks(path, min_len=4.0, max_len=15.0, sr=22050):
    y, sr = librosa.load(path, sr=sr, mono=True)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)

    hooks = []

    for i in range(len(beat_times) - 1):
        start = beat_times[i]

        # find næste beat der giver korrekt længde
        for j in range(i + 1, len(beat_times)):
            end = beat_times[j]
            dur = end - start

            if min_len <= dur <= max_len:
                hooks.append((start, end))
                break
            if dur > max_len:
                break

    return hooks
