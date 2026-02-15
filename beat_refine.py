import numpy as np
import librosa
from dataclasses import dataclass
from utils import estimate_bars_from_duration


@dataclass
class RefineResult:
    ok: bool
    start: float
    end: float
    bpm: float
    bars: int
    score: float
    reason: str
    bars_estimated: int = 0
    bpm_confidence: float = 0.0


def _normalize_window_to_audio_timebase(window_start: float, window_end: float, audio_duration: float):
    """
    Normalize requested window into the loaded audio's timebase.

    Handles both:
    1) full-track timebase (absolute seconds in full file), and
    2) segment-relative files where callers may mistakenly pass absolute times
       from the source track (window_start >> segment duration).
    """
    ws = max(0.0, float(window_start))
    we = max(ws, float(window_end))

    if audio_duration <= 0.0:
        return 0.0, 0.0

    # Normal case: values fit within this file's duration.
    if ws < audio_duration and we <= audio_duration + 1e-6:
        return ws, we

    # Timebase mismatch heuristic:
    # segment file duration ~= requested interval length, while absolute start
    # lies beyond this file's own duration.
    requested_len = max(0.0, we - ws)
    if ws >= audio_duration and requested_len > 0.0:
        if abs(requested_len - audio_duration) <= 0.35:
            return 0.0, audio_duration

    # Safe clamp fallback.
    ws = min(ws, audio_duration)
    we = min(max(we, ws), audio_duration)
    return ws, we


def refine_to_n_bars(
    wav_path,
    window_start,
    window_end,
    beats_per_bar=4,
    prefer_bars=2,
    sr=22050,
):
    """
    Refine audio segment to exact N bars with beat grid alignment.

    Args:
        wav_path: Path to audio file
        window_start: Start time in seconds
        window_end: End time in seconds
        beats_per_bar: Number of beats per bar (default 4)
        prefer_bars: Preferred number of bars (1, 2, 4, 8, 16)
        sr: Sample rate

    Returns:
        RefineResult with ok, start, end, bpm, bars, score, reason, bars_estimated, bpm_confidence
    """
    y, sr = librosa.load(wav_path, sr=sr)
    audio_duration = len(y) / float(sr)
    window_start, window_end = _normalize_window_to_audio_timebase(window_start, window_end, audio_duration)

    y_segment = y[int(window_start * sr): int(window_end * sr)]
    duration = len(y_segment) / float(sr)

    # Guard: only classify as too_short under product threshold (2.0s)
    if duration < 2.0:
        return RefineResult(
            False, 0, 0, 0, 0, 0, "too_short",
            bars_estimated=0, bpm_confidence=0.0
        )

    tempo, beats = librosa.beat.beat_track(y=y_segment, sr=sr)

    # Handle numpy array return from newer librosa
    if hasattr(tempo, '__len__'):
        tempo = tempo[0] if len(tempo) > 0 else 120.0
    bpm = int(round(float(tempo)))

    # Calculate BPM confidence from beat interval consistency
    if len(beats) > 2:
        beat_times = librosa.frames_to_time(beats, sr=sr)
        intervals = np.diff(beat_times)
        if len(intervals) > 0:
            cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
            bpm_confidence = max(0.0, min(1.0, 1.0 - cv))
        else:
            bpm_confidence = 0.0
    else:
        bpm_confidence = 0.0

    # Estimate bars from duration
    bars_estimated = estimate_bars_from_duration(duration, bpm, beats_per_bar)

    if len(beats) == 0:
        return RefineResult(
            False, 0, 0, bpm, 0, 0, "no_onsets",
            bars_estimated=bars_estimated, bpm_confidence=bpm_confidence
        )

    if len(beats) < beats_per_bar:
        return RefineResult(
            False, 0, 0, bpm, 0, 0, "not_enough_beats",
            bars_estimated=bars_estimated, bpm_confidence=bpm_confidence
        )

    # Respect explicit user choice from Song Hunter: never silently snap 16→8/4/2/1.
    bars = max(1, int(prefer_bars))
    beats_needed = beats_per_bar * bars

    # Need beats[beats_needed] to exist
    if len(beats) >= beats_needed + 1:
        start_frame = beats[0]
        end_frame = beats[beats_needed]

        start = librosa.frames_to_time(start_frame, sr=sr)
        end = librosa.frames_to_time(end_frame, sr=sr)

        score = float(bpm / 200.0)

        return RefineResult(
            True, start, end, bpm, bars, score, "",
            bars_estimated=bars_estimated, bpm_confidence=bpm_confidence
        )

    # Not enough beats for requested bars
    return RefineResult(
        False, 0, 0, bpm, 0, 0, "not_enough_beats_for_requested_bars",
        bars_estimated=bars_estimated, bpm_confidence=bpm_confidence
    )


def refine_best_1_or_2_bars(
    wav_path,
    window_start,
    window_end,
    beats_per_bar=4,
    prefer_bars=1,
    sr=22050,
):
    """
    Backward-compatible alias for refine_to_n_bars.
    """
    return refine_to_n_bars(
        wav_path, window_start, window_end,
        beats_per_bar, prefer_bars, sr
    )
