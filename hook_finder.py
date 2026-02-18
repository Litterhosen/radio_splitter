import numpy as np
import librosa
from dataclasses import dataclass
from rs_utils import find_ffmpeg


@dataclass
class HookWindow:
    start: float
    end: float
    score: float
    energy: float
    loopability: float
    stability: float
    bpm: float
    bpm_confidence: float = 0.0
    bpm_source: str = "segment_estimate"
    bpm_clip_confidence: float = 0.0  # Per-segment confidence


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


def estimate_bpm_with_confidence(y, sr):
    """
    Estimate BPM from audio segment with confidence metric.
    
    Args:
        y: Audio time series (numpy array)
        sr: Sample rate of audio
    
    Returns:
        (bpm, confidence) where confidence is based on beat interval consistency (0.0-1.0)
    """
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    # Handle numpy array return from newer librosa
    if hasattr(tempo, '__len__'):
        tempo = tempo[0] if len(tempo) > 0 else 120.0
    bpm = int(round(float(tempo)))
    
    # Calculate confidence from beat interval consistency
    if len(beats) > 2:
        beat_times = librosa.frames_to_time(beats, sr=sr)
        intervals = np.diff(beat_times)
        if len(intervals) > 0:
            cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
            confidence = max(0.0, min(1.0, 1.0 - cv))
        else:
            confidence = 0.0
    else:
        confidence = 0.0
    
    return bpm, confidence


def estimate_global_bpm(y, sr):
    """
    Estimate global BPM for entire track with confidence metric.
    
    Args:
        y: Audio time series (numpy array)
        sr: Sample rate of audio
    
    Returns:
        (bpm, confidence) where confidence is based on beat interval consistency (0.0-1.0)
    """
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    # Handle numpy array return from newer librosa
    if hasattr(tempo, '__len__'):
        tempo = tempo[0] if len(tempo) > 0 else 120.0
    bpm = int(round(float(tempo)))
    
    # Calculate confidence from beat interval consistency
    if len(beats) > 2:
        beat_times = librosa.frames_to_time(beats, sr=sr)
        intervals = np.diff(beat_times)
        if len(intervals) > 0:
            cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
            confidence = max(0.0, min(1.0, 1.0 - cv))
        else:
            confidence = 0.0
    else:
        confidence = 0.0
    
    return bpm, confidence


def normalize_bpm_family(bpm: float, confidence: float = 0.0) -> int:
    """
    Map BPM to a musically plausible family when estimate likely lands on subdivision.
    Conservative rule: only adjust low/high or low-confidence estimates.
    """
    bpm = float(bpm or 0.0)
    if bpm <= 0:
        return 0

    if 96 <= bpm <= 176 and confidence >= 0.35:
        return int(round(bpm))

    factors = [1.0, 2.0, 0.5, 1.5, (2.0 / 3.0)]
    candidates = []
    for f in factors:
        c = bpm * f
        if 60.0 <= c <= 220.0:
            candidates.append(c)
    if not candidates:
        return int(round(bpm))

    anchor = 128.0

    def _score(c):
        range_penalty = 0.0 if 96.0 <= c <= 176.0 else 18.0
        anchor_penalty = abs(c - anchor) * 0.15
        transform_penalty = abs(np.log2(max(c, 1e-6) / max(bpm, 1e-6))) * 10.0
        return range_penalty + anchor_penalty + transform_penalty

    best = min(candidates, key=_score)
    return int(round(best))


def normalize_bpm_to_prior(segment_bpm, global_bpm, tolerance=0.15):
    """
    Normalize segment BPM to global BPM by detecting half/double tempo errors.
    
    Args:
        segment_bpm: BPM estimated from segment
        global_bpm: Global BPM from full track
        tolerance: Relative tolerance for matching (default 0.15 = 15%)
    
    Returns:
        (normalized_bpm, bpm_source) where bpm_source is 'track_global' if normalized,
        'segment_estimate' otherwise
    """
    if global_bpm <= 0:
        return segment_bpm, "segment_estimate"
    
    # Check common tempo aliases (x2, /2, x1.5, x2/3) against global BPM.
    candidates = [
        (segment_bpm, "segment_estimate"),
        (segment_bpm * 2, "track_global"),
        (segment_bpm / 2, "track_global"),
        (segment_bpm * 1.5, "track_global"),
        (segment_bpm * (2.0 / 3.0), "track_global"),
    ]
    
    best_bpm = segment_bpm
    best_source = "segment_estimate"
    min_error = float('inf')
    
    for candidate_bpm, source in candidates:
        error = abs(candidate_bpm - global_bpm) / global_bpm
        if error < tolerance and error < min_error:
            min_error = error
            best_bpm = int(round(candidate_bpm))
            best_source = source
    
    return best_bpm, best_source


def _select_diverse_hooks(windows, duration, topn, min_gap_s):
    """
    Select hooks with diversity guardrails:
    - keep temporal spacing
    - avoid over-concentration in one section
    - cap clip count on short tracks for higher precision
    """
    if not windows or topn <= 0:
        return []

    duration = float(duration or 0.0)
    min_gap_s = float(min_gap_s or 0.0)

    # Precision cap for short tracks (e.g., 3-4 min pop songs)
    if duration > 0:
        dynamic_cap = max(6, int(duration / 20.0))
        target_topn = min(int(topn), dynamic_cap)
    else:
        target_topn = int(topn)

    if target_topn <= 0:
        return []

    # Bucket size for section diversity (roughly 6-10 sections across a track)
    if duration > 0:
        bucket_size = max(14.0, min(36.0, duration / 8.0))
    else:
        bucket_size = 24.0

    # A stronger initial spacing to reduce near-duplicates.
    typical_len = float(np.median([max(0.0, w.end - w.start) for w in windows[: min(len(windows), 24)]]) or 0.0)
    strong_gap = max(min_gap_s, typical_len * 0.70)

    selected = []
    per_bucket = {}
    selected_keys = set()
    max_per_bucket = 2

    def _key(w):
        return (round(float(w.start), 3), round(float(w.end), 3))

    # Pass 1: strict diversity.
    for w in windows:
        bkey = int(float(w.start) // bucket_size)
        if per_bucket.get(bkey, 0) >= max_per_bucket:
            continue
        if any(abs(float(w.start) - float(s.start)) < strong_gap for s in selected):
            continue
        selected.append(w)
        selected_keys.add(_key(w))
        per_bucket[bkey] = per_bucket.get(bkey, 0) + 1
        if len(selected) >= target_topn:
            break

    # Pass 2: fill remainder with normal gap only.
    if len(selected) < target_topn:
        for w in windows:
            wk = _key(w)
            if wk in selected_keys:
                continue
            if any(abs(float(w.start) - float(s.start)) < min_gap_s for s in selected):
                continue
            selected.append(w)
            selected_keys.add(wk)
            if len(selected) >= target_topn:
                break

    return selected


def find_hooks(
    wav_path,
    hook_len_range=(4.0, 15.0),
    prefer_len=8.0,
    hop_s=1.0,
    topn=12,
    min_gap_s=2.0,
    prefer_bars=None,
    beats_per_bar=4,
):
    """
    Find hook windows with optional bar-based length calculation.
    
    Args:
        wav_path: Path to audio file
        hook_len_range: (min, max) hook length in seconds
        prefer_len: Preferred length in seconds (fallback if prefer_bars not provided)
        hop_s: Hop size for scanning in seconds
        topn: Number of top hooks to return
        min_gap_s: Minimum gap between hooks in seconds
        prefer_bars: Preferred number of bars (1, 2, 4, 8, 16) - overrides prefer_len
        beats_per_bar: Number of beats per bar (default 4)
    
    Returns:
        Tuple of (hooks, global_bpm, global_confidence)
    """
    y, sr = librosa.load(wav_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Estimate global BPM once for the entire track
    global_bpm, global_confidence = estimate_global_bpm(y, sr)
    global_bpm = normalize_bpm_family(global_bpm, global_confidence)

    min_len, max_len = hook_len_range
    
    use_bar_length = prefer_bars is not None and prefer_bars > 0 and global_bpm > 0

    # Calculate target window length based on prefer_bars if provided
    if use_bar_length:
        # Calculate bar duration: (60 / BPM) * beats_per_bar
        bar_duration = (60.0 / global_bpm) * beats_per_bar
        target_len = prefer_bars * bar_duration
        # In bar mode, user preference owns loop duration.
        # Keep only a safety floor, and cap to track duration.
        win_len = max(min_len, min(target_len, duration))
    else:
        # Fallback to prefer_len
        win_len = max(min_len, min(prefer_len, max_len))
    
    hop = hop_s

    windows = []
    
    # Scan with multiple window sizes within the range for better detection
    MIN_WINDOW_SIZE_DIFFERENCE = 1.0  # seconds
    window_sizes = [win_len]
    if (not use_bar_length) and max_len > min_len:
        # Add min and max if they're different from prefer_len
        if abs(min_len - win_len) > MIN_WINDOW_SIZE_DIFFERENCE:
            window_sizes.append(min_len)
        if abs(max_len - win_len) > MIN_WINDOW_SIZE_DIFFERENCE:
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
            segment_bpm, segment_confidence = estimate_bpm_with_confidence(seg, sr)
            segment_bpm = normalize_bpm_family(segment_bpm, segment_confidence)
            
            # Normalize segment BPM to global BPM
            normalized_bpm, bpm_source = normalize_bpm_to_prior(
                segment_bpm, global_bpm, tolerance=0.15
            )
            
            # Determine final confidence based on source
            if bpm_source == "track_global":
                # Using global BPM, so use global confidence
                final_confidence = global_confidence
            else:
                # Using segment estimate, so use segment confidence
                final_confidence = segment_confidence

            windows.append(
                HookWindow(
                    start, end, score, energy, loopability, stability, 
                    normalized_bpm, final_confidence, bpm_source, segment_confidence
                )
            )

            t += hop

    windows.sort(key=lambda w: w.score, reverse=True)
    selected = _select_diverse_hooks(
        windows=windows,
        duration=duration,
        topn=topn,
        min_gap_s=min_gap_s,
    )

    return selected, global_bpm, global_confidence
