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
):
    """
    Finds tempo/beats and exports loops of N bars (4/4).
    Strategy:
      - detect beats
      - choose loop windows starting at beats (or fixed grid)
      - export via ffmpeg with fades
    """
    y, sr = librosa.load(str(wav_music), sr=None, mono=False)
    if y.ndim > 1:
        y_mono = librosa.to_mono(y)
    else:
        y_mono = y

    tempo, beat_frames = librosa.beat.beat_track(y=y_mono, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Fallback hvis beat-detektion fejler
    if len(beat_times) < 8:
        # assume 120 bpm as fallback grid
        tempo = float(tempo) if tempo else 120.0
        beat_period = 60.0 / float(tempo)
        duration = librosa.get_duration(y=y_mono, sr=sr)
        beat_times = np.arange(0.0, duration, beat_period)

    beats_per_bar = 4
    beats_in_loop = bars * beats_per_bar

    # pick start indices
    starts = []
    if snap_to_beats:
        # start every hop_beats
        for i in range(0, len(beat_times) - beats_in_loop, hop_beats):
            starts.append(i)
    else:
        starts = list(range(0, len(beat_times) - beats_in_loop, hop_beats))

    # limit output count (avoid 500 loops)
    starts = starts[:80]

    for n, i in enumerate(starts, start=1):
        start_t = float(beat_times[i])
        end_t = float(beat_times[i + beats_in_loop])
        dur = end_t - start_t
        if dur <= 0.5:
            continue

        out_path = out_dir / f"loop_{bars:02d}bars_{n:03d}.{export_format}"

        fade_s = fade_ms / 1000.0
        afade = []
        if fade_ms > 0:
            afade = [
                "-af",
                f"afade=t=in:st=0:d={fade_s},afade=t=out:st={max(0.0, dur - fade_s)}:d={fade_s}",
            ]

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_t),
            "-to", str(end_t),
            "-i", str(wav_music),
            "-vn",
            *afade,
            str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    # Save a quick report
    (out_dir / "loops_report.txt").write_text(
        f"Detected tempo: {float(tempo):.2f} BPM\n"
        f"Beats found: {len(beat_times)}\n"
        f"Bars per loop: {bars}\n",
        encoding="utf-8",
    )
