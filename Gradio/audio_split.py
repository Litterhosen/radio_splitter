import re
import math
import numpy as np
import librosa
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

from rs_utils import find_ffmpeg, find_ffprobe, run_cmd

MIN_SEGMENT_SECONDS = 4.0
AUTO_THRESHOLD_FALLBACK_DB = -25.0
AUTO_THRESHOLD_CLAMP_MIN_DB = -55.0
AUTO_THRESHOLD_CLAMP_MAX_DB = -18.0
HIGH_NOISE_FLOOR_WARNING_DB = -20.0

_LAST_NON_SILENT_DEBUG: Dict[str, Any] = {}


@dataclass
class SilenceEvent:
    start: float
    end: float


def get_last_non_silent_debug() -> Dict[str, Any]:
    """Return diagnostics from the most recent detect_non_silent_intervals call."""
    return dict(_LAST_NON_SILENT_DEBUG)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(v)))


def _select_sample_offsets(duration: float, sample_seconds: float, windows: int) -> List[Tuple[float, float]]:
    if duration <= 0.0:
        return []
    total = max(1.0, min(float(sample_seconds), float(duration)))
    n = max(1, int(windows))
    window_dur = max(2.0, total / float(n))
    if duration <= window_dur:
        return [(0.0, float(duration))]
    max_start = max(0.0, float(duration) - window_dur)
    starts = np.linspace(0.0, max_start, num=n)
    return [(float(s), float(min(duration, s + window_dur))) for s in starts]


def estimate_noise_floor_db(
    audio_path: Path,
    sample_seconds: float = 60.0,
    method: str = "rms_percentile",
    percentile: float = 20.0,
    frame_ms: float = 50.0,
    sample_windows: int = 3,
) -> float:
    """
    Estimate noise floor in dBFS using robust RMS percentile frames.
    """
    if method != "rms_percentile":
        raise ValueError(f"Unsupported method: {method}")

    duration = get_duration_seconds(audio_path)
    if duration <= 0:
        return float(AUTO_THRESHOLD_FALLBACK_DB)

    offsets = _select_sample_offsets(duration, sample_seconds=sample_seconds, windows=sample_windows)
    all_frame_db: List[np.ndarray] = []
    sr = 16000
    frame_len = max(1, int(sr * (float(frame_ms) / 1000.0)))
    p = float(_clamp(percentile, 1.0, 99.0))

    for start_s, end_s in offsets:
        seg_dur = max(0.0, float(end_s) - float(start_s))
        if seg_dur <= 0.0:
            continue
        try:
            y, _ = librosa.load(str(audio_path), sr=sr, mono=True, offset=float(start_s), duration=float(seg_dur))
        except Exception:
            continue
        if y is None or y.size < frame_len:
            continue
        n_frames = int(y.size // frame_len)
        if n_frames <= 0:
            continue
        y_trim = y[: n_frames * frame_len]
        frames = y_trim.reshape(n_frames, frame_len)
        rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
        db = 20.0 * np.log10(np.clip(rms, 1e-12, None))
        all_frame_db.append(db.astype(np.float32))

    if not all_frame_db:
        return float(AUTO_THRESHOLD_FALLBACK_DB)

    frame_db = np.concatenate(all_frame_db, axis=0)
    noise_floor_db = float(np.percentile(frame_db, p))
    if not math.isfinite(noise_floor_db):
        return float(AUTO_THRESHOLD_FALLBACK_DB)
    return float(noise_floor_db)


def _detect_non_silent_intervals_with_threshold(
    audio_path: Path,
    threshold_db: float,
    min_silence_s: float,
    pad_s: float,
    min_segment_s: float,
    *,
    duration_limit_s: Optional[float] = None,
) -> List[Tuple[float, float]]:
    ffmpeg = find_ffmpeg()
    full_dur = get_duration_seconds(audio_path)
    if full_dur <= 0:
        return []

    analysis_dur = float(full_dur)
    if duration_limit_s is not None and duration_limit_s > 0.0:
        analysis_dur = min(analysis_dur, float(duration_limit_s))
    if analysis_dur <= 0.0:
        return []

    thr = float(threshold_db)
    min_sil = max(0.05, float(min_silence_s))
    pad = max(0.0, float(pad_s))
    min_seg = max(MIN_SEGMENT_SECONDS, float(min_segment_s))

    cmd = [ffmpeg, "-hide_banner", "-nostats"]
    if analysis_dur < full_dur:
        cmd.extend(["-t", f"{analysis_dur:.3f}"])
    cmd.extend(
        [
            "-i",
            str(audio_path),
            "-af",
            f"silencedetect=noise={thr}dB:d={min_sil}",
            "-f",
            "null",
            "-",
        ]
    )
    r = run_cmd(cmd, check=False)
    silences = _parse_silencedetect(r.err)

    intervals: List[Tuple[float, float]] = []
    cur = 0.0
    for s in silences:
        a = cur
        b = max(cur, float(s.start))
        if (b - a) >= min_seg:
            intervals.append((max(0.0, a - pad), min(analysis_dur, b + pad)))
        cur = max(cur, float(s.end))

    if (analysis_dur - cur) >= min_seg:
        intervals.append((max(0.0, cur - pad), analysis_dur))

    intervals.sort()
    merged: List[List[float]] = []
    for a, b in intervals:
        if not merged:
            merged.append([a, b])
        else:
            pa, pb = merged[-1]
            if a <= pb:
                merged[-1][1] = max(pb, b)
            else:
                merged.append([a, b])
    return [(float(a), float(b)) for a, b in merged if (float(b) - float(a)) >= min_seg]


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
    if dur <= 0 or dur < MIN_SEGMENT_SECONDS:
        return []
    seg = max(MIN_SEGMENT_SECONDS, float(segment_len))
    out = []
    t = 0.0
    while t < dur:
        out.append((t, min(dur, t + seg)))
        t += seg
    if len(out) >= 2 and (out[-1][1] - out[-1][0]) < MIN_SEGMENT_SECONDS:
        out[-2] = (out[-2][0], out[-1][1])
        out.pop()
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
    threshold_mode: str = "auto",
    margin_db: float = 10.0,
    threshold_clamp_min_db: float = AUTO_THRESHOLD_CLAMP_MIN_DB,
    threshold_clamp_max_db: float = AUTO_THRESHOLD_CLAMP_MAX_DB,
    noise_sample_seconds: float = 60.0,
    noise_percentile: float = 20.0,
    noise_frame_ms: float = 50.0,
    noise_sample_windows: int = 3,
    quick_test_mode: bool = True,
    quick_test_seconds: float = 120.0,
    quick_test_retries: int = 3,
    quick_test_margin_step_db: float = 2.0,
    quick_test_min_silence_factor: float = 0.85,
    return_debug: bool = False,
) -> Union[List[Tuple[float, float]], Tuple[List[Tuple[float, float]], Dict[str, Any]]]:
    """
    Returns non-silent intervals by using ffmpeg silencedetect and then inverting silence regions.
    Supports automatic thresholding based on estimated noise floor and quick-test retries.
    """
    dur = get_duration_seconds(audio_path)
    global _LAST_NON_SILENT_DEBUG
    debug: Dict[str, Any] = {
        "threshold_mode": str(threshold_mode),
        "noise_floor_db": None,
        "silence_thresh_db": None,
        "margin_db": None,
        "threshold_clamp_min_db": float(threshold_clamp_min_db),
        "threshold_clamp_max_db": float(threshold_clamp_max_db),
        "threshold_fallback_used": False,
        "warning": "",
        "quick_test": {"enabled": bool(quick_test_mode), "attempts": []},
        "segments_found": 0,
    }
    if dur <= 0:
        _LAST_NON_SILENT_DEBUG = dict(debug)
        if return_debug:
            return [], dict(debug)
        return []

    mode = str(threshold_mode or "auto").strip().lower()
    auto_mode = mode.startswith("auto")
    min_sil = max(0.05, float(min_silence_s))
    min_seg = max(MIN_SEGMENT_SECONDS, float(min_segment_s))
    threshold_db = float(noise_db)
    margin = float(margin_db)
    noise_floor_db: Optional[float] = None

    if auto_mode:
        noise_floor_db = estimate_noise_floor_db(
            audio_path,
            sample_seconds=float(noise_sample_seconds),
            method="rms_percentile",
            percentile=float(noise_percentile),
            frame_ms=float(noise_frame_ms),
            sample_windows=int(noise_sample_windows),
        )
        debug["noise_floor_db"] = round(float(noise_floor_db), 3)
        if float(noise_floor_db) > HIGH_NOISE_FLOOR_WARNING_DB:
            threshold_db = float(AUTO_THRESHOLD_FALLBACK_DB)
            debug["threshold_fallback_used"] = True
            debug["warning"] = (
                f"High noise floor detected ({noise_floor_db:.2f} dBFS). "
                f"Using fallback threshold {AUTO_THRESHOLD_FALLBACK_DB:.1f} dB."
            )
        else:
            threshold_db = _clamp(
                float(noise_floor_db) + margin,
                float(threshold_clamp_min_db),
                float(threshold_clamp_max_db),
            )
        debug["margin_db"] = round(margin, 3)

    test_attempts: List[Dict[str, Any]] = []
    if bool(quick_test_mode):
        test_limit = min(float(dur), max(5.0, float(quick_test_seconds)))
        retry_count = max(1, int(quick_test_retries))
        trial_margin = float(margin)
        trial_min_sil = float(min_sil)
        trial_threshold = float(threshold_db)

        for attempt in range(1, retry_count + 1):
            if auto_mode and not bool(debug["threshold_fallback_used"]) and noise_floor_db is not None:
                trial_threshold = _clamp(
                    float(noise_floor_db) + float(trial_margin),
                    float(threshold_clamp_min_db),
                    float(threshold_clamp_max_db),
                )
            trial_intervals = _detect_non_silent_intervals_with_threshold(
                audio_path,
                threshold_db=trial_threshold,
                min_silence_s=trial_min_sil,
                pad_s=float(pad_s),
                min_segment_s=min_seg,
                duration_limit_s=test_limit,
            )
            attempt_meta = {
                "attempt": int(attempt),
                "duration_limit_s": round(test_limit, 3),
                "silence_thresh_db": round(float(trial_threshold), 3),
                "min_silence_s": round(float(trial_min_sil), 3),
                "margin_db": round(float(trial_margin), 3) if auto_mode else None,
                "segments_found": int(len(trial_intervals)),
            }
            test_attempts.append(attempt_meta)
            if trial_intervals:
                threshold_db = float(trial_threshold)
                min_sil = float(trial_min_sil)
                margin = float(trial_margin)
                break

            if auto_mode and not bool(debug["threshold_fallback_used"]):
                trial_margin = max(2.0, float(trial_margin) - abs(float(quick_test_margin_step_db)))
            trial_min_sil = max(0.15, float(trial_min_sil) * float(quick_test_min_silence_factor))

        if test_attempts and test_attempts[-1]["segments_found"] == 0:
            # Keep the final retry values even if test still found 0, then run full-file pass.
            threshold_db = float(test_attempts[-1]["silence_thresh_db"])
            min_sil = float(test_attempts[-1]["min_silence_s"])
            if auto_mode and test_attempts[-1]["margin_db"] is not None:
                margin = float(test_attempts[-1]["margin_db"])

    intervals = _detect_non_silent_intervals_with_threshold(
        audio_path,
        threshold_db=threshold_db,
        min_silence_s=min_sil,
        pad_s=float(pad_s),
        min_segment_s=min_seg,
        duration_limit_s=None,
    )

    debug["silence_thresh_db"] = round(float(threshold_db), 3)
    if auto_mode:
        debug["margin_db"] = round(float(margin), 3)
    debug["quick_test"] = {
        "enabled": bool(quick_test_mode),
        "attempts": test_attempts,
    }
    debug["segments_found"] = int(len(intervals))
    _LAST_NON_SILENT_DEBUG = dict(debug)

    if return_debug:
        return intervals, dict(debug)
    return intervals


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
        except Exception as e:
            # If zero-crossing fails, use original times
            # Note: This is expected for very short clips or clips with no zero crossings
            import logging
            logging.debug(f"Zero-crossing alignment skipped: {e}")
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
