from __future__ import annotations

import contextlib
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple

import numpy as np

from Gradio.audio_split import detect_non_silent_intervals, get_duration_seconds
from rs_utils import find_ffmpeg, run_cmd


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

    # Enforce min and max duration constraints
    constrained: List[Tuple[float, float]] = []
    for a, b in merged:
        seg_len = b - a
        if seg_len < min_segment:
            continue
        if seg_len <= max_segment:
            constrained.append((a, b))
            continue
        cur = a
        while cur < b:
            nxt = min(cur + max_segment, b)
            if (nxt - cur) >= min_segment:
                constrained.append((cur, nxt))
            cur = nxt

    return constrained


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
    min_segment_sec: float = 1.5,
    max_segment_sec: float = 45.0,
    merge_gap_sec: float = 0.35,
    chunk_sec: float = 600.0,
    silence_noise_db: float = -30.0,
    silence_min_s: float = 0.4,
    silence_pad_s: float = 0.15,
    prefer_method: str = "vad",
) -> Tuple[List[Tuple[float, float]], str, bool]:
    duration = get_duration_seconds(audio_path)
    if duration <= 0:
        return [], "vad", False

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
        merged: List[List[float]] = [[intervals[0][0], intervals[0][1]]]
        for a, b in intervals[1:]:
            if a - merged[-1][1] <= merge_gap_sec:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        intervals = [(float(a), float(b)) for a, b in merged]

        # final max split guard
        bounded: List[Tuple[float, float]] = []
        for a, b in intervals:
            cur = a
            while cur < b:
                nxt = min(cur + max_segment_sec, b)
                if (nxt - cur) >= min_segment_sec:
                    bounded.append((cur, nxt))
                cur = nxt
        split_method = "vad" if used_vad else "energy"
        return bounded, split_method, chunking_enabled

    # Last resort: silence detection
    silence_intervals = detect_non_silent_intervals(
        audio_path,
        noise_db=silence_noise_db,
        min_silence_s=silence_min_s,
        pad_s=silence_pad_s,
        min_segment_s=min_segment_sec,
    )

    bounded_silence: List[Tuple[float, float]] = []
    for a, b in silence_intervals:
        cur = a
        while cur < b:
            nxt = min(cur + max_segment_sec, b)
            if (nxt - cur) >= min_segment_sec:
                bounded_silence.append((cur, nxt))
            cur = nxt

    return bounded_silence, "silence", chunking_enabled
