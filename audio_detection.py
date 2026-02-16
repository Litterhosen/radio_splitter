"""
Audio type detection and classification for radio_splitter.
Uses librosa and rule-based heuristics for lightweight detection.
"""
from typing import Dict, Any, Tuple
from pathlib import Path
import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def detect_audio_type(audio_path: Path, sr: int = 16000, duration: float = 30.0) -> Dict[str, Any]:
    """
    Detect audio type (music/speech/mixed/jingle_ad/unknown) using lightweight heuristics.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate for analysis
        duration: Maximum duration to analyze (seconds) for efficiency
    
    Returns:
        Dict with:
            - audio_type_guess: str (music | speech | mixed | jingle_ad | unknown)
            - audio_type_confidence: float (0..1)
            - recommended_mode: str (Song Hunter | Broadcast Hunter)
    """
    if not LIBROSA_AVAILABLE:
        return {
            "audio_type_guess": "unknown",
            "audio_type_confidence": 0.0,
            "recommended_mode": "Song Hunter | Broadcast Hunter"
        }
    
    try:
        # Load audio snippet for analysis
        y, sr_loaded = librosa.load(str(audio_path), sr=sr, duration=duration, mono=True)
        
        # Extract features
        features = _extract_audio_features(y, sr_loaded)
        
        # Classify based on features
        audio_type, confidence = _classify_audio_type(features)
        
        # Recommend mode
        recommended_mode = _recommend_mode(audio_type, confidence)
        
        return {
            "audio_type_guess": audio_type,
            "audio_type_confidence": round(confidence, 3),
            "recommended_mode": recommended_mode
        }
    
    except Exception as e:
        # Log error for debugging
        import sys
        print(f"Warning: Audio detection failed: {e}", file=sys.stderr)
        # Fallback on error
        return {
            "audio_type_guess": "unknown",
            "audio_type_confidence": 0.0,
            "recommended_mode": "Song Hunter | Broadcast Hunter"
        }


def _extract_audio_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Extract relevant audio features for classification."""
    features = {}
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
    features['spectral_centroid_std'] = float(np.std(spectral_centroids))
    
    # Zero crossing rate (speech tends to have higher ZCR)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std'] = float(np.std(zcr))
    
    # Tempo/beat strength (music tends to have stronger beat)
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo) if tempo else 0.0
        features['beat_strength'] = float(len(beats) / (len(y) / sr)) if len(beats) > 0 else 0.0
    except Exception as e:
        features['tempo'] = 0.0
        features['beat_strength'] = 0.0
    
    # RMS energy variation
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))
    features['rms_variation'] = features['rms_std'] / (features['rms_mean'] + 1e-8)
    
    # Spectral rolloff (frequency distribution)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean'] = float(np.mean(rolloff))
    
    return features


def _classify_audio_type(features: Dict[str, float]) -> Tuple[str, float]:
    """
    Classify audio type based on extracted features.
    Uses simple rule-based heuristics.
    
    Returns:
        (audio_type, confidence)
    """
    # Initialize scores
    music_score = 0.0
    speech_score = 0.0
    jingle_score = 0.0
    
    # Rule 1: Strong beat indicates music
    if features['beat_strength'] > 1.5:
        music_score += 0.4
    elif features['beat_strength'] > 1.0:
        music_score += 0.2
    
    # Rule 2: Tempo in typical music range (80-180 BPM)
    if 80 <= features['tempo'] <= 180:
        music_score += 0.2
    
    # Rule 3: High ZCR variation suggests speech
    if features['zcr_std'] > 0.05:
        speech_score += 0.3
    
    # Rule 4: Lower spectral centroid suggests speech (voice is lower frequency)
    if features['spectral_centroid_mean'] < 2000:
        speech_score += 0.2
    elif features['spectral_centroid_mean'] > 3000:
        music_score += 0.2
    
    # Rule 5: High RMS variation suggests speech or mixed content
    if features['rms_variation'] > 0.5:
        speech_score += 0.2
    elif features['rms_variation'] < 0.3:
        music_score += 0.2
    
    # Rule 6: Very short duration with high energy could be jingle/ad
    # (This would need duration info, so we skip for now)
    
    # Determine type based on scores
    max_score = max(music_score, speech_score, jingle_score)
    
    if max_score < 0.3:
        # No clear winner, likely mixed or unknown
        if music_score > 0.1 and speech_score > 0.1:
            return "mixed", 0.5
        return "unknown", 0.3
    
    if music_score == max_score:
        # Check if it could be a jingle (short musical segment)
        if music_score > 0.5 and speech_score > 0.2:
            return "jingle_ad", 0.6
        return "music", min(music_score + 0.3, 0.95)
    
    if speech_score == max_score:
        # Check if mixed (has some musical elements)
        if music_score > 0.3:
            return "mixed", 0.7
        return "speech", min(speech_score + 0.3, 0.95)
    
    # Fallback
    return "mixed", 0.5


def _recommend_mode(audio_type: str, confidence: float) -> str:
    """Recommend processing mode based on audio type."""
    if audio_type == "music":
        return "Song Hunter | Broadcast Hunter"
    elif audio_type == "speech":
        return "Broadcast Hunter"
    elif audio_type == "mixed":
        return "Broadcast Hunter | Song Hunter"
    elif audio_type == "jingle_ad":
        return "Broadcast Hunter"
    else:
        return "Song Hunter | Broadcast Hunter"


def normalize_text_for_signature(text: str, max_words: int = 10) -> str:
    """
    Create normalized text signature for grouping/deduplication.
    
    Args:
        text: Input text
        max_words: Maximum number of words to include
    
    Returns:
        Normalized signature string
    """
    if not text:
        return ""
    
    # Normalize: lowercase, remove extra spaces, take first N words
    words = text.lower().strip().split()[:max_words]
    return " ".join(words)




def merge_language_votes(votes: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Majority vote for file-level language from multiple language observations."""
    normalized = []
    for vote in votes:
        lang = (vote or {}).get("language_guess", "unknown")
        conf = float((vote or {}).get("language_confidence", 0.0) or 0.0)
        if lang in {"da", "en"}:
            normalized.append((lang, conf))

    if not normalized:
        return {"language_guess": "unknown", "language_confidence": 0.0}

    counts: Dict[str, int] = {}
    conf_totals: Dict[str, float] = {}
    for lang, conf in normalized:
        counts[lang] = counts.get(lang, 0) + 1
        conf_totals[lang] = conf_totals.get(lang, 0.0) + conf

    best_lang = sorted(counts.items(), key=lambda kv: (-kv[1], -conf_totals.get(kv[0], 0.0), kv[0]))[0][0]
    best_count = counts[best_lang]
    avg_conf = conf_totals[best_lang] / max(best_count, 1)
    majority_factor = best_count / len(normalized)

    return {
        "language_guess": best_lang,
        "language_confidence": round(min(0.99, avg_conf * majority_factor), 3),
    }


def resolve_clip_language(
    file_language_guess: str,
    file_language_confidence: float,
    clip_language_info: Dict[str, Any],
    clip_text: str,
    min_chars: int = 20,
    high_confidence: float = 0.85,
) -> Dict[str, Any]:
    """Choose clip language, defaulting to file language for short/low-confidence clips."""
    clip_guess = (clip_language_info or {}).get("language_guess", "unknown")
    clip_conf = float((clip_language_info or {}).get("language_confidence", 0.0) or 0.0)
    has_enough_text = len((clip_text or "").strip()) >= int(min_chars)
    use_clip_language = clip_guess in {"da", "en"} and (has_enough_text or clip_conf >= float(high_confidence))

    if use_clip_language:
        return {"language_guess": clip_guess, "language_confidence": round(clip_conf, 3), "language_source": "clip"}

    file_guess = file_language_guess if file_language_guess in {"da", "en"} else "unknown"
    file_conf = float(file_language_confidence or 0.0)
    return {"language_guess": file_guess, "language_confidence": round(file_conf, 3), "language_source": "file"}
def extract_language_info(transcribe_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract language detection info from Whisper transcription result.
    
    Args:
        transcribe_result: Result dict from transcribe_wav
    
    Returns:
        Dict with language_guess and language_confidence
    """
    language = transcribe_result.get("language", None)
    
    # Whisper doesn't provide confidence directly, so we use heuristics
    # Based on text length and language detection
    confidence = 0.5  # Default moderate confidence
    
    if language in ["da", "en"]:
        # Common languages we expect - higher confidence
        text = transcribe_result.get("text", "")
        if len(text) > 50:
            confidence = 0.8
        elif len(text) > 20:
            confidence = 0.7
        else:
            confidence = 0.6
    elif language:
        # Other detected language
        confidence = 0.6
    else:
        # No language detected
        language = "unknown"
        confidence = 0.0
    
    return {
        "language_guess": language or "unknown",
        "language_confidence": round(confidence, 3)
    }
