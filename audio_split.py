import re
import numpy as np
import librosa
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

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


def find_zero_crossing(audio_data: np.ndarray, sample_rate: int, target_time: float, 
                      search_window_ms: float = 10.0) -> Optional[float]:
    """
    Find the nearest zero-crossing point near target_time.
    
    Args:
        audio_data: Audio samples (mono)
        sample_rate: Sample rate of audio
        target_time: Target time in seconds
        search_window_ms: Search window in milliseconds (±)
    
    Returns:
        Adjusted time in seconds, or None if not found
    """
    target_sample = int(target_time * sample_rate)
    window_samples = int((search_window_ms / 1000.0) * sample_rate)
    
    start_sample = max(0, target_sample - window_samples)
    end_sample = min(len(audio_data), target_sample + window_samples)
    
    if start_sample >= end_sample:
        return None
    
    window_data = audio_data[start_sample:end_sample]
    
    # Find zero crossings (sign changes)
    signs = np.sign(window_data)
    zero_crossings = np.where(np.diff(signs) != 0)[0]
    
    if len(zero_crossings) == 0:
        return None
    
    # Find closest zero crossing to target
    target_in_window = target_sample - start_sample
    closest_idx = zero_crossings[np.argmin(np.abs(zero_crossings - target_in_window))]
    
    # Convert back to time
    adjusted_sample = start_sample + closest_idx
    return adjusted_sample / float(sample_rate)


def cut_segment_with_fades(
    src_path: Path, 
    out_path: Path, 
    core_start: float, 
    core_end: float,
    pre_roll_ms: float = 25.0,
    fade_in_ms: float = 15.0,
    fade_out_ms: float = 15.0,
    tail_sec: float = 0.0,
    apply_zero_crossing: bool = True,
    bitrate: str = "192k",
    is_wav: bool = False
) -> dict:
    """
    Export audio clip with pre-roll, fades, and optional zero-crossing alignment.
    
    Args:
        src_path: Source audio file
        out_path: Output file path
        core_start: Core clip start time in seconds
        core_end: Core clip end time in seconds
        pre_roll_ms: Pre-roll duration in milliseconds (default 25ms)
        fade_in_ms: Fade-in duration in milliseconds (default 15ms)
        fade_out_ms: Fade-out duration in milliseconds (default 15ms)
        tail_sec: Tail duration in seconds (default 0.0)
        apply_zero_crossing: Whether to apply zero-crossing alignment (default True)
        bitrate: Bitrate for MP3 encoding (default "192k")
        is_wav: Export as WAV instead of MP3
    
    Returns:
        Dictionary with export metadata:
        - core_start_sec, core_end_sec, core_dur_sec
        - export_start_sec, export_end_sec, export_dur_sec
        - pre_roll_ms, fade_in_ms, fade_out_ms, tail_sec
        - zero_cross_applied (boolean)
    """
    # Convert milliseconds to seconds
    pre_roll_sec = pre_roll_ms / 1000.0
    fade_in_sec = fade_in_ms / 1000.0
    fade_out_sec = fade_out_ms / 1000.0
    
    zero_cross_applied = False
    adjusted_start = core_start
    adjusted_end = core_end
    
    # Apply zero-crossing alignment if requested
    if apply_zero_crossing:
        try:
            # Load audio for zero-crossing detection
            y, sr = librosa.load(src_path, sr=22050, mono=True)
            
            # Find zero crossings for start and end
            zc_start = find_zero_crossing(y, sr, core_start, search_window_ms=10.0)
            zc_end = find_zero_crossing(y, sr, core_end, search_window_ms=10.0)
            
            if zc_start is not None:
                adjusted_start = zc_start
                zero_cross_applied = True
            
            if zc_end is not None:
                adjusted_end = zc_end
                zero_cross_applied = True
        except Exception:
            # If zero-crossing fails, use original times
            pass
    
    # Calculate export times
    export_start = max(0.0, adjusted_start - pre_roll_sec)
    export_end = adjusted_end + tail_sec
    
    # Build ffmpeg filter for fades
    filters = []
    
    # Fade in at the beginning of the clip (after pre-roll)
    fade_in_start = pre_roll_sec if pre_roll_sec > 0 else 0
    filters.append(f"afade=t=in:st={fade_in_start:.3f}:d={fade_in_sec:.3f}")
    
    # Fade out at the end of core loop (before tail)
    core_duration = adjusted_end - adjusted_start
    fade_out_start = pre_roll_sec + core_duration - fade_out_sec
    filters.append(f"afade=t=out:st={fade_out_start:.3f}:d={fade_out_sec:.3f}")
    
    filter_complex = ",".join(filters)
    
    # Export with ffmpeg
    ffmpeg = find_ffmpeg()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        ffmpeg, "-y",
        "-ss", f"{export_start:.3f}",
        "-to", f"{export_end:.3f}",
        "-i", str(src_path),
        "-af", filter_complex,
    ]
    
    if is_wav:
        cmd.extend(["-ac", "1", "-ar", "16000"])
    else:
        cmd.extend(["-vn", "-codec:a", "libmp3lame", "-b:a", bitrate])
    
    cmd.append(str(out_path))
    
    run_cmd(cmd)
    
    # Return metadata
    return {
        "core_start_sec": float(adjusted_start),
        "core_end_sec": float(adjusted_end),
        "core_dur_sec": float(adjusted_end - adjusted_start),
        "export_start_sec": float(export_start),
        "export_end_sec": float(export_end),
        "export_dur_sec": float(export_end - export_start),
        "pre_roll_ms": float(pre_roll_ms),
        "fade_in_ms": float(fade_in_ms),
        "fade_out_ms": float(fade_out_ms),
        "tail_sec": float(tail_sec),
        "zero_cross_applied": bool(zero_cross_applied),
    }
