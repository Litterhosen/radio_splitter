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
    
    # Check segment_bpm, segment_bpm*2, segment_bpm/2 against global_bpm
    candidates = [
        (segment_bpm, "segment_estimate"),
        (segment_bpm * 2, "track_global"),
        (segment_bpm / 2, "track_global"),
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
    
    # Estimate global BPM once for the entire track
    global_bpm, global_confidence = estimate_global_bpm(y, sr)

    min_len, max_len = hook_len_range
    # Clamp prefer_len to the specified range
    win_len = max(min_len, min(prefer_len, max_len))
    hop = hop_s

    windows = []
    
    # Scan with multiple window sizes within the range for better detection
    MIN_WINDOW_SIZE_DIFFERENCE = 1.0  # seconds
    window_sizes = [win_len]
    if max_len > min_len:
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

    selected = []
    for w in windows:
        if len(selected) >= topn:
            break
        if all(abs(w.start - s.start) > min_gap_s for s in selected):
            selected.append(w)

    return selected, global_bpm, global_confidence
