from __future__ import annotations

import contextlib
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple, Dict, Any

import numpy as np

from Gradio.audio_split import detect_non_silent_intervals, get_duration_seconds
from rs_utils import find_ffmpeg, run_cmd

MIN_SEGMENT_FLOOR_SEC = 4.0
NATURAL_EDGE_PAD_SEC = 0.12
_LAST_BROADCAST_SEGMENTATION_DEBUG: Dict[str, Any] = {}


def get_last_broadcast_segmentation_debug() -> Dict[str, Any]:
    return dict(_LAST_BROADCAST_SEGMENTATION_DEBUG)


def _extract_chunk_wav16k(src: Path, out_wav: Path, start_s: float, dur_s: float) -> None:
    ffmpeg = find_ffmpeg()
    run_cmd(
        [
            ffmpeg,
            "-y",
            "-ss",
            f"{start_s:.3f}",
            "-t",
            f"{dur_s:.3f}",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out_wav),
        ]
    )


def _sanitize_intervals(intervals: List[Tuple[float, float]], duration: float) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for a, b in intervals:
        aa = max(0.0, min(float(duration), float(a)))
        bb = max(aa, min(float(duration), float(b)))
        if (bb - aa) > 1e-6:
            out.append((aa, bb))
    out.sort()
    return out


def _merge_nearby(intervals: List[Tuple[float, float]], merge_gap: float) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    merged: List[List[float]] = [[intervals[0][0], intervals[0][1]]]
    for a, b in intervals[1:]:
        if a - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(float(a), float(b)) for a, b in merged]


def _enforce_min_duration(
    intervals: List[Tuple[float, float]],
    min_segment_sec: float,
    merge_gap_sec: float,
) -> List[Tuple[float, float]]:
    items: List[List[float]] = [[float(a), float(b)] for a, b in intervals]
    if not items:
        return []

    strong_join_gap = max(float(merge_gap_sec) * 2.0, 0.35)
    weak_join_gap = max(strong_join_gap * 1.8, 0.8)

    changed = True
    while changed and len(items) > 1:
        changed = False
        for idx, (a, b) in enumerate(list(items)):
            if (b - a) >= min_segment_sec:
                continue

            left_gap = float("inf")
            right_gap = float("inf")
            if idx > 0:
                left_gap = a - items[idx - 1][1]
            if idx + 1 < len(items):
                right_gap = items[idx + 1][0] - b

            left_ok = idx > 0 and left_gap <= strong_join_gap
            right_ok = idx + 1 < len(items) and right_gap <= strong_join_gap
            if not left_ok and not right_ok:
                left_ok = idx > 0 and left_gap <= weak_join_gap
                right_ok = idx + 1 < len(items) and right_gap <= weak_join_gap

            if left_ok or right_ok:
                if left_ok and (not right_ok or left_gap <= right_gap):
                    items[idx - 1][1] = max(items[idx - 1][1], b)
                    del items[idx]
                else:
                    items[idx + 1][0] = min(items[idx + 1][0], a)
                    del items[idx]
                changed = True
                break

    return [(float(a), float(b)) for a, b in items if (float(b) - float(a)) >= min_segment_sec]


def _split_bounded(
    intervals: List[Tuple[float, float]],
    min_segment_sec: float,
    max_segment_sec: float,
) -> List[Tuple[float, float]]:
    bounded: List[Tuple[float, float]] = []
    for a, b in intervals:
        cur = float(a)
        end = float(b)
        while (end - cur) > max_segment_sec:
            nxt = cur + max_segment_sec
            tail = end - nxt
            if tail < min_segment_sec:
                nxt = max(cur + min_segment_sec, end - min_segment_sec)
            if (nxt - cur) >= min_segment_sec:
                bounded.append((cur, nxt))
            cur = nxt
        if (end - cur) >= min_segment_sec:
            bounded.append((cur, end))
    return bounded


def _naturalize_intervals(
    intervals: List[Tuple[float, float]],
    duration: float,
    min_segment_sec: float,
    max_segment_sec: float,
    merge_gap_sec: float,
) -> List[Tuple[float, float]]:
    if not intervals:
        return []

    cleaned = _sanitize_intervals(intervals, duration)
    if not cleaned:
        return []

    # Small edge padding softens "hard cuts" at detected boundaries.
    padded = [
        (max(0.0, a - NATURAL_EDGE_PAD_SEC), min(duration, b + NATURAL_EDGE_PAD_SEC))
        for a, b in cleaned
    ]
    merged = _merge_nearby(padded, merge_gap_sec)
    merged = _enforce_min_duration(merged, min_segment_sec=min_segment_sec, merge_gap_sec=merge_gap_sec)
    merged = _split_bounded(merged, min_segment_sec=min_segment_sec, max_segment_sec=max_segment_sec)
    merged = _sanitize_intervals(merged, duration)
    return [(a, b) for a, b in merged if (b - a) >= min_segment_sec]


def _frames_to_intervals(flags: List[bool], frame_dur: float, min_segment: float, max_segment: float, merge_gap: float) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    i = 0
    n = len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < n and flags[j]:
            j += 1
        a = i * frame_dur
        b = j * frame_dur
        intervals.append((a, b))
        i = j

    if not intervals:
        return []

    # Merge nearby speech islands
    merged: List[List[float]] = [[intervals[0][0], intervals[0][1]]]
    for a, b in intervals[1:]:
        if a - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    # Keep short islands for a later global "naturalization" pass,
    # but still respect max_segment here to avoid oversized chunks.
    provisional_min = max(0.30, min(float(min_segment), MIN_SEGMENT_FLOOR_SEC) * 0.35)
    bounded: List[Tuple[float, float]] = []
    for a, b in merged:
        cur = float(a)
        end = float(b)
        while cur < end:
            nxt = min(cur + float(max_segment), end)
            seg_len = nxt - cur
            if seg_len > 1e-6 and (seg_len >= provisional_min or not bounded):
                bounded.append((cur, nxt))
            cur = nxt
    return bounded


def _webrtc_vad_intervals(chunk_wav: Path, min_segment: float, max_segment: float, merge_gap: float) -> List[Tuple[float, float]]:
    import webrtcvad

    vad = webrtcvad.Vad(2)
    frame_ms = 30
    frame_dur = frame_ms / 1000.0

    with contextlib.closing(wave.open(str(chunk_wav), "rb")) as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        if sr != 16000 or channels != 1 or sampwidth != 2:
            return []
        pcm = wf.readframes(wf.getnframes())

    bytes_per_frame = int(sr * frame_dur) * 2
    flags: List[bool] = []
    for i in range(0, len(pcm) - bytes_per_frame + 1, bytes_per_frame):
        frame = pcm[i : i + bytes_per_frame]
        try:
            flags.append(vad.is_speech(frame, sr))
        except Exception:
            flags.append(False)

    return _frames_to_intervals(flags, frame_dur, min_segment, max_segment, merge_gap)


def _energy_intervals(chunk_wav: Path, min_segment: float, max_segment: float, merge_gap: float) -> List[Tuple[float, float]]:
    frame_ms = 30
    frame_dur = frame_ms / 1000.0

    with contextlib.closing(wave.open(str(chunk_wav), "rb")) as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        if sr != 16000 or channels != 1 or sampwidth != 2:
            return []
        pcm = wf.readframes(wf.getnframes())

    if not pcm:
        return []

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    frame_samples = int(sr * frame_dur)
    n_frames = len(samples) // frame_samples
    if n_frames <= 0:
        return []

    trimmed = samples[: n_frames * frame_samples]
    frames = trimmed.reshape(n_frames, frame_samples)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-9)

    threshold = max(float(np.percentile(rms, 60)), 0.008)
    flags = [float(v) >= threshold for v in rms]
    return _frames_to_intervals(flags, frame_dur, min_segment, max_segment, merge_gap)


def detect_broadcast_segments(
    audio_path: Path,
    min_segment_sec: float = MIN_SEGMENT_FLOOR_SEC,
    max_segment_sec: float = 45.0,
    merge_gap_sec: float = 0.35,
    chunk_sec: float = 600.0,
    silence_noise_db: float = -30.0,
    silence_min_s: float = 0.4,
    silence_pad_s: float = 0.15,
    silence_threshold_mode: str = "auto",
    silence_margin_db: float = 10.0,
    silence_quick_test_mode: bool = True,
    silence_quick_test_seconds: float = 120.0,
    silence_quick_test_retries: int = 3,
    prefer_method: str = "vad",
) -> Tuple[List[Tuple[float, float]], str, bool]:
    global _LAST_BROADCAST_SEGMENTATION_DEBUG
    duration = get_duration_seconds(audio_path)
    if duration <= 0:
        _LAST_BROADCAST_SEGMENTATION_DEBUG = {
            "method": "none",
            "chunking_enabled": False,
            "segments_found": 0,
            "silence_debug": {},
        }
        return [], "vad", False
    min_segment_sec = max(float(min_segment_sec), MIN_SEGMENT_FLOOR_SEC)
    max_segment_sec = max(float(max_segment_sec), min_segment_sec + 1.0)
    merge_gap_sec = max(0.05, float(merge_gap_sec))

    chunking_enabled = duration > chunk_sec
    intervals: List[Tuple[float, float]] = []
    used_vad = False
    used_energy = False

    with TemporaryDirectory(prefix="broadcast_chunks_") as td:
        tmp_dir = Path(td)
        offset = 0.0
        while offset < duration:
            current_dur = min(chunk_sec, duration - offset)
            chunk_wav = tmp_dir / f"chunk_{int(offset):09d}.wav"
            _extract_chunk_wav16k(audio_path, chunk_wav, offset, current_dur)

            local_intervals: List[Tuple[float, float]] = []
            method_used = "energy"
            if prefer_method == "vad":
                try:
                    local_intervals = _webrtc_vad_intervals(
                        chunk_wav,
                        min_segment=min_segment_sec,
                        max_segment=max_segment_sec,
                        merge_gap=merge_gap_sec,
                    )
                except Exception:
                    local_intervals = []
                if local_intervals:
                    method_used = "vad"
                    used_vad = True
                else:
                    local_intervals = _energy_intervals(
                        chunk_wav,
                        min_segment=min_segment_sec,
                        max_segment=max_segment_sec,
                        merge_gap=merge_gap_sec,
                    )
                    if local_intervals:
                        used_energy = True
            else:
                local_intervals = _energy_intervals(
                    chunk_wav,
                    min_segment=min_segment_sec,
                    max_segment=max_segment_sec,
                    merge_gap=merge_gap_sec,
                )
                if local_intervals:
                    used_energy = True

            if local_intervals:
                intervals.extend([(offset + a, offset + b) for a, b in local_intervals])

            offset += current_dur

    if intervals:
        intervals.sort()
        bounded = _naturalize_intervals(
            intervals=intervals,
            duration=float(duration),
            min_segment_sec=min_segment_sec,
            max_segment_sec=max_segment_sec,
            merge_gap_sec=merge_gap_sec,
        )
        split_method = "vad" if used_vad else "energy"
        _LAST_BROADCAST_SEGMENTATION_DEBUG = {
            "method": split_method,
            "chunking_enabled": bool(chunking_enabled),
            "segments_found": int(len(bounded)),
            "silence_debug": {},
        }
        return bounded, split_method, chunking_enabled

    # Last resort: silence detection
    silence_intervals, silence_debug = detect_non_silent_intervals(
        audio_path,
        noise_db=silence_noise_db,
        min_silence_s=silence_min_s,
        pad_s=silence_pad_s,
        min_segment_s=min_segment_sec,
        threshold_mode=silence_threshold_mode,
        margin_db=silence_margin_db,
        quick_test_mode=silence_quick_test_mode,
        quick_test_seconds=silence_quick_test_seconds,
        quick_test_retries=silence_quick_test_retries,
        return_debug=True,
    )

    bounded_silence = _naturalize_intervals(
        intervals=silence_intervals,
        duration=float(duration),
        min_segment_sec=min_segment_sec,
        max_segment_sec=max_segment_sec,
        merge_gap_sec=merge_gap_sec,
    )
    _LAST_BROADCAST_SEGMENTATION_DEBUG = {
        "method": "silence",
        "chunking_enabled": bool(chunking_enabled),
        "segments_found": int(len(bounded_silence)),
        "silence_debug": silence_debug if isinstance(silence_debug, dict) else {},
    }

    return bounded_silence, "silence", chunking_enabled
