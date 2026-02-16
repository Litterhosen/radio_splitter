from audio_detection import merge_language_votes, resolve_clip_language


def test_merge_language_votes_majority_da():
    votes = [
        {"language_guess": "da", "language_confidence": 0.8},
        {"language_guess": "en", "language_confidence": 0.7},
        {"language_guess": "da", "language_confidence": 0.6},
    ]
    merged = merge_language_votes(votes)
    assert merged["language_guess"] == "da"
    assert merged["language_confidence"] > 0.0


def test_resolve_clip_language_uses_file_for_short_low_conf_clip():
    clip_info = {"language_guess": "en", "language_confidence": 0.6}
    resolved = resolve_clip_language(
        file_language_guess="da",
        file_language_confidence=0.88,
        clip_language_info=clip_info,
        clip_text="Oh",
        min_chars=20,
        high_confidence=0.85,
    )
    assert resolved["language_guess"] == "da"
    assert resolved["language_source"] == "file"


def test_resolve_clip_language_uses_clip_when_text_long_enough():
    clip_info = {"language_guess": "en", "language_confidence": 0.62}
    resolved = resolve_clip_language(
        file_language_guess="da",
        file_language_confidence=0.88,
        clip_language_info=clip_info,
        clip_text="This is a long enough english phrase for robust clip language.",
        min_chars=20,
        high_confidence=0.85,
    )
    assert resolved["language_guess"] == "en"
    assert resolved["language_source"] == "clip"
