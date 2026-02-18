from audio_detection import _classify_audio_type, _merge_window_votes


def test_long_music_window_not_labeled_as_jingle():
    features = {
        "beat_strength": 1.4,
        "tempo": 138.0,
        "zcr_std": 0.020,
        "zcr_mean": 0.060,
        "rms_variation": 0.30,
        "spectral_centroid_mean": 3200.0,
        "spectral_flatness_mean": 0.010,
        "onset_pulse": 0.62,
    }
    audio_type, confidence, _ = _classify_audio_type(
        features,
        clip_duration=24.0,
        full_duration=240.0,
    )
    assert audio_type == "music"
    assert confidence >= 0.5


def test_short_mixed_window_can_be_jingle():
    features = {
        "beat_strength": 1.1,
        "tempo": 124.0,
        "zcr_std": 0.052,
        "zcr_mean": 0.115,
        "rms_variation": 0.62,
        "spectral_centroid_mean": 2150.0,
        "spectral_flatness_mean": 0.032,
        "onset_pulse": 0.46,
    }
    audio_type, _, scores = _classify_audio_type(
        features,
        clip_duration=20.0,
        full_duration=50.0,
    )
    assert audio_type in {"jingle_ad", "mixed", "speech"}
    assert scores["speech"] > 0.0


def test_long_file_vote_downgrades_jingle_to_music_or_mixed():
    votes = [
        {"audio_type_guess": "jingle_ad", "audio_type_confidence": 0.70},
        {"audio_type_guess": "jingle_ad", "audio_type_confidence": 0.68},
        {"audio_type_guess": "music", "audio_type_confidence": 0.66},
    ]
    merged = _merge_window_votes(votes, total_duration=260.0)
    assert merged["audio_type_guess"] in {"music", "mixed"}
