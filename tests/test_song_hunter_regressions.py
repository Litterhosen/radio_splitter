import numpy as np
from unittest.mock import patch

from hook_finder import find_hooks
from beat_refine import refine_to_n_bars
from downloaders import classify_error, ErrorClassification


def test_prefer_bars_controls_hook_duration_not_fixed_four_seconds(tmp_path):
    sr = 22050
    # 70s dummy audio long enough for a 32s window.
    y = np.random.randn(sr * 70).astype(np.float32) * 0.01

    with patch("hook_finder.librosa.load", return_value=(y, sr)), \
         patch("hook_finder.estimate_global_bpm", return_value=(120, 0.9)), \
         patch("hook_finder.estimate_bpm_with_confidence", return_value=(120, 0.8)):
        hooks, _, _ = find_hooks(
            "dummy.wav",
            hook_len_range=(2.0, 8.0),
            prefer_len=4.0,
            prefer_bars=16,
            beats_per_bar=4,
            topn=1,
            hop_s=1.0,
            min_gap_s=0.5,
        )

    assert hooks, "Expected at least one hook"
    dur = hooks[0].end - hooks[0].start
    assert abs(dur - 32.0) <= 0.2
    assert dur > 4.5


def test_refine_too_short_only_for_under_two_seconds():
    sr = 22050
    # Exactly 2.0s segment
    y = np.zeros(sr * 2, dtype=np.float32)

    with patch("beat_refine.librosa.load", return_value=(y, sr)), \
         patch("beat_refine.librosa.beat.beat_track", return_value=(120.0, np.array([]))):
        rr = refine_to_n_bars("dummy.wav", 0.0, 2.0, prefer_bars=2, sr=sr)

    assert rr.reason != "too_short"


def test_classify_js_runtime_missing_from_runtime_message():
    code, _, _ = classify_error("No supported JavaScript runtime was found", has_js_runtime=False)
    assert code == ErrorClassification.ERR_JS_RUNTIME_MISSING


def test_refine_handles_absolute_times_against_segment_audio_timebase():
    sr = 22050
    # Simulate a segment wav of 4.0s, but caller passes absolute full-track times (100-104s).
    y = np.zeros(sr * 4, dtype=np.float32)

    with patch("beat_refine.librosa.load", return_value=(y, sr)), \
         patch("beat_refine.librosa.beat.beat_track", return_value=(120.0, np.array([0, 10, 20, 30, 40, 50, 60, 70, 80]))), \
         patch("beat_refine.librosa.frames_to_time", side_effect=lambda frames, sr: np.asarray(frames) * 0.05):
        rr = refine_to_n_bars("dummy_segment.wav", 100.0, 104.0, prefer_bars=2, beats_per_bar=4, sr=sr)

    # Without timebase normalization this would be sliced empty and often return too_short.
    assert rr.reason != "too_short"


def test_refine_does_not_snap_down_when_requested_bars_unavailable():
    sr = 22050
    y = np.zeros(sr * 16, dtype=np.float32)
    # Enough beats for 8 bars but intentionally not enough for requested 16 bars
    beats = np.arange(0, 33)  # 33 beats => max 8 bars (32 beats needed + start)

    with patch("beat_refine.librosa.load", return_value=(y, sr)), \
         patch("beat_refine.librosa.beat.beat_track", return_value=(120.0, beats)), \
         patch("beat_refine.librosa.frames_to_time", side_effect=lambda frames, sr: np.asarray(frames) * 0.1):
        rr = refine_to_n_bars("dummy.wav", 0.0, 16.0, prefer_bars=16, beats_per_bar=4, sr=sr)

    assert rr.ok is False
    assert rr.reason == "not_enough_beats_for_requested_bars"


def test_classify_youtube_unavailable_family_maps_to_unified_code():
    code1, _, _ = classify_error("Video unavailable", has_js_runtime=True)
    code2, _, _ = classify_error("Sign in to confirm your age", has_js_runtime=True)
    code3, _, _ = classify_error("This video is not available in your country", has_js_runtime=True)

    assert code1 == ErrorClassification.ERR_YT_UNAVAILABLE
    assert code2 == ErrorClassification.ERR_YT_UNAVAILABLE
    assert code3 == ErrorClassification.ERR_YT_UNAVAILABLE
