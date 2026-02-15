from pathlib import Path
from unittest.mock import patch

from broadcast_splitter import _frames_to_intervals, detect_broadcast_segments


def test_frames_to_intervals_merges_and_splits_by_max():
    flags = [False] * 5 + [True] * 120 + [False] * 5  # 3.6s speech at 30ms frame
    intervals = _frames_to_intervals(flags, frame_dur=0.03, min_segment=1.5, max_segment=2.0, merge_gap=0.35)
    assert intervals
    assert all((b - a) <= 2.0 + 1e-6 for a, b in intervals)
    assert sum((b - a) for a, b in intervals) >= 3.0


def test_detect_broadcast_segments_prefers_energy_when_vad_empty(tmp_path):
    dummy = tmp_path / "dummy.wav"
    dummy.write_bytes(b"x")

    with patch("broadcast_splitter.get_duration_seconds", return_value=120.0), \
         patch("broadcast_splitter._extract_chunk_wav16k", return_value=None), \
         patch("broadcast_splitter._webrtc_vad_intervals", return_value=[]), \
         patch("broadcast_splitter._energy_intervals", return_value=[(0.0, 10.0)]):
        intervals, method, chunking = detect_broadcast_segments(Path(dummy), chunk_sec=60.0)

    assert method == "energy"
    assert chunking is True
    assert intervals
