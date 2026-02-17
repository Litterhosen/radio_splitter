# loops.py
import subprocess
from pathlib import Path

import numpy as np
import librosa

def export_bar_loops(
    wav_music: Path,
    out_dir: Path,
    bars: int,
    snap_to_beats: bool,
    hop_beats: int,
    fade_ms: int,
    export_format: str = "wav",
    top_n: int = 20,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(str(wav_music), sr=None, mono=False)
    y_mono = librosa.to_mono(y) if getattr(y, "ndim", 1) > 1 else y

    tempo, beat_frames = librosa.beat.beat_track(y=y_mono, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_times) < 8:
        tempo = float(tempo) if tempo else 120.0
        beat_period = 60.0 / float(tempo)
        duration = librosa.get_duration(y=y_mono, sr=sr)
        beat_times = np.arange(0.0, duration, beat_period)

    beats_per_bar = 4
    beats_in_loop = bars * beats_per_bar

    starts = []
    if snap_to_beats:
        for i in range(0, len(beat_times) - beats_in_loop, hop_beats):
            starts.append(i)
    else:
        starts = list(range(0, len(beat_times) - beats_in_loop, hop_beats))

    starts = starts[:80]

    candidates = []
    for i in starts:
        start_t = float(beat_times[i])
        end_t = float(beat_times[i + beats_in_loop])
        dur = end_t - start_t
        if dur <= 0.5:
            continue
        s = max(0, int(start_t * sr))
        e = min(len(y_mono), int(end_t * sr))
        seg = y_mono[s:e]
        if seg.size < int(0.5 * sr):
            continue
        energy = float(np.mean(np.abs(seg)))
        variation = float(np.std(np.abs(seg)))
        candidates.append((i, start_t, end_t, dur, energy, variation))

    scores = []
    if candidates:
        energies = np.array([c[4] for c in candidates], dtype=np.float64)
        variations = np.array([c[5] for c in candidates], dtype=np.float64)
        e_min, e_max = float(energies.min()), float(energies.max())
        v_min, v_max = float(variations.min()), float(variations.max())
        e_den = (e_max - e_min) if (e_max - e_min) > 1e-12 else 1.0
        v_den = (v_max - v_min) if (v_max - v_min) > 1e-12 else 1.0
        for c in candidates:
            e_n = (c[4] - e_min) / e_den
            v_n = (c[5] - v_min) / v_den
            score = 0.6 * e_n + 0.4 * v_n
            scores.append((*c, float(score)))

    scores.sort(key=lambda x: x[-1], reverse=True)
    selected = scores[: max(1, int(top_n))]

    for n, it in enumerate(selected, start=1):
        _, start_t, end_t, dur, _, _, score = it
        out_path = out_dir / f"loop_{bars:02d}bars_{n:03d}_score_{score:.3f}.{export_format}"
        fade_s = fade_ms / 1000.0
        afade = []
        if fade_ms > 0:
            afade = [
                "-af",
                f"afade=t=in:st=0:d={fade_s},afade=t=out:st={max(0.0, dur - fade_s)}:d={fade_s}",
            ]
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_t),
            "-to",
            str(end_t),
            "-i",
            str(wav_music),
            "-vn",
            *afade,
            str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    details = "".join(
        f"{idx:03d} {float(item[1]):.3f}-{float(item[2]):.3f} score={float(item[-1]):.3f}\n"
        for idx, item in enumerate(selected, start=1)
    )
    (out_dir / "loops_report.txt").write_text(
        f"Detected tempo: {float(tempo):.2f} BPM\n"
        f"Beats found: {len(beat_times)}\n"
        f"Bars per loop: {bars}\n"
        f"Candidates: {len(candidates)}\n"
        f"Exported top: {len(selected)}\n"
        f"{details}",
        encoding="utf-8",
    )
