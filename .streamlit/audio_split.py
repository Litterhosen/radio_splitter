import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from utils import find_ffmpeg, find_ffprobe, run_cmd


@dataclass
class SilenceEvent:
    start: float
    end: float


def get_duration_seconds(audio_path: Path) -> float:
    ffprobe = find_ffprobe()
    r = run_cmd([
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nk=1:nw=1",
        str(audio_path)
    ])
    try:
        return float(r.out.strip())
    except Exception:
        return 0.0


def fixed_length_intervals(audio_path: Path, segment_len: float) -> List[Tuple[float, float]]:
    dur = get_duration_seconds(audio_path)
    if dur <= 0:
        return []
    seg = max(0.5, float(segment_len))
    out = []
    t = 0.0
    while t < dur:
        out.append((t, min(dur, t + seg)))
        t += seg
    return out


def _parse_silencedetect(stderr: str) -> List[SilenceEvent]:
    """
    Parses ffmpeg silencedetect output lines:
    silence_start: 12.34
    silence_end: 15.67 | silence_duration: 3.33
    """
    starts = []
    events = []

    for line in (stderr or "").splitlines():
        m1 = re.search(r"silence_start:\s*([0-9.]+)", line)
        if m1:
            starts.append(float(m1.group(1)))
            continue

        m2 = re.search(r"silence_end:\s*([0-9.]+)", line)
        if m2:
            end = float(m2.group(1))
            start = starts.pop(0) if starts else max(0.0, end - 0.1)
            events.append(SilenceEvent(start=start, end=end))

    return events


def detect_non_silent_intervals(
    audio_path: Path,
    noise_db: float = -35.0,
    min_silence_s: float = 0.7,
    pad_s: float = 0.15,
    min_segment_s: float = 1.2,
) -> List[Tuple[float, float]]:
    """
    Returns non-silent intervals by using ffmpeg silencedetect and then inverting silence regions.
    """
    ffmpeg = find_ffmpeg()
    dur = get_duration_seconds(audio_path)
    if dur <= 0:
        return []

    thr = float(noise_db)
    min_sil = float(min_silence_s)
    pad = max(0.0, float(pad_s))
    min_seg = max(0.1, float(min_segment_s))

    r = run_cmd([
        ffmpeg, "-hide_banner", "-nostats",
        "-i", str(audio_path),
        "-af", f"silencedetect=noise={thr}dB:d={min_sil}",
        "-f", "null", "-"
    ], check=False)

    silences = _parse_silencedetect(r.err)

    # Invert silence → non-silent
    intervals = []
    cur = 0.0
    for s in silences:
        a = cur
        b = max(cur, s.start)
        if (b - a) >= min_seg:
            intervals.append((max(0.0, a - pad), min(dur, b + pad)))
        cur = max(cur, s.end)

    # tail
    if (dur - cur) >= min_seg:
        intervals.append((max(0.0, cur - pad), dur))

    # Merge overlaps caused by padding
    intervals.sort()
    merged = []
    for a, b in intervals:
        if not merged:
            merged.append([a, b])
        else:
            pa, pb = merged[-1]
            if a <= pb:
                merged[-1][1] = max(pb, b)
            else:
                merged.append([a, b])

    return [(float(a), float(b)) for a, b in merged]


def cut_segment_to_wav(src_path: Path, out_path: Path, start_s: float, end_s: float):
    ffmpeg = find_ffmpeg()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    a = max(0.0, float(start_s))
    b = max(a, float(end_s))
    run_cmd([
        ffmpeg, "-y",
        "-ss", f"{a:.3f}", "-to", f"{b:.3f}",
        "-i", str(src_path),
        "-ac", "1", "-ar", "16000",
        str(out_path)
    ])


def cut_segment_to_mp3(src_path: Path, out_path: Path, start_s: float, end_s: float, bitrate: str = "192k"):
    ffmpeg = find_ffmpeg()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    a = max(0.0, float(start_s))
    b = max(a, float(end_s))
    run_cmd([
        ffmpeg, "-y",
        "-ss", f"{a:.3f}", "-to", f"{b:.3f}",
        "-i", str(src_path),
        "-vn",
        "-codec:a", "libmp3lame",
        "-b:a", bitrate,
        str(out_path)
    ])
