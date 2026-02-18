"""
Audio type detection and classification for radio_splitter.
Uses librosa and rule-based heuristics for lightweight detection.
"""
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def _safe_tempo_scalar(tempo: Any) -> float:
    """librosa may return tempo as scalar or ndarray depending on version."""
    if hasattr(tempo, "__len__"):
        return float(tempo[0]) if len(tempo) > 0 else 0.0
    try:
        return float(tempo)
    except Exception:
        return 0.0


def _window_starts(total_duration: float, window_duration: float, max_windows: int = 3) -> List[float]:
    """Choose start/middle/end windows for robust file-level voting."""
    if total_duration <= 0:
        return [0.0]
    wd = max(4.0, min(window_duration, total_duration))
    if total_duration <= wd + 0.25:
        return [0.0]
    if max_windows <= 1:
        return [0.0]

    starts = [0.0, max(0.0, total_duration / 2.0 - wd / 2.0), max(0.0, total_duration - wd)]
    deduped = []
    for s in starts:
        if all(abs(s - prev) > 1.0 for prev in deduped):
            deduped.append(float(s))
    return deduped[:max_windows]


def detect_audio_type(
    audio_path: Path,
    sr: int = 16000,
    duration: float = 30.0,
    known_duration_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Detect audio type (music/speech/mixed/jingle_ad/unknown) with multi-window voting.
    """
    if not LIBROSA_AVAILABLE:
        return {
            "audio_type_guess": "unknown",
            "audio_type_confidence": 0.0,
            "recommended_mode": "Song Hunter | Broadcast Hunter",
            "analysis_windows": 0,
        }

    try:
        if known_duration_sec is not None and float(known_duration_sec) > 0:
            total_duration = float(known_duration_sec)
        else:
            try:
                total_duration = float(librosa.get_duration(path=str(audio_path)))
            except TypeError:
                # librosa<0.10 uses "filename" instead of "path"
                total_duration = float(librosa.get_duration(filename=str(audio_path)))
            except Exception:
                total_duration = 0.0

        if total_duration <= 0.0:
            # Last-resort fallback to avoid long-track misclassification.
            try:
                y_full, sr_full = librosa.load(str(audio_path), sr=sr, mono=True)
                total_duration = len(y_full) / float(sr_full) if sr_full else 0.0
            except Exception:
                total_duration = 0.0

        window_duration = float(duration if duration > 0 else 20.0)
        if total_duration > 0:
            window_duration = min(window_duration, total_duration)
        window_duration = max(8.0, window_duration)

        starts = _window_starts(total_duration, window_duration, max_windows=3)
        votes = []
        for start in starts:
            y, sr_loaded = librosa.load(
                str(audio_path),
                sr=sr,
                offset=max(0.0, float(start)),
                duration=window_duration,
                mono=True,
            )
            clip_duration = len(y) / float(sr_loaded) if sr_loaded else 0.0
            if clip_duration < 2.0:
                continue

            features = _extract_audio_features(y, sr_loaded)
            audio_type, confidence, scores = _classify_audio_type(
                features,
                clip_duration=clip_duration,
                full_duration=(total_duration if total_duration > 0 else clip_duration),
            )
            votes.append(
                {
                    "audio_type_guess": audio_type,
                    "audio_type_confidence": float(confidence),
                    "scores": scores,
                }
            )

        if not votes:
            return {
                "audio_type_guess": "unknown",
                "audio_type_confidence": 0.0,
                "recommended_mode": "Song Hunter | Broadcast Hunter",
                "analysis_windows": 0,
            }

        merged = _merge_window_votes(votes, total_duration=total_duration)
        audio_type = merged["audio_type_guess"]
        confidence = merged["audio_type_confidence"]
        return {
            "audio_type_guess": audio_type,
            "audio_type_confidence": round(confidence, 3),
            "recommended_mode": _recommend_mode(audio_type, confidence),
            "analysis_windows": len(votes),
            "duration_sec_used": round(float(total_duration), 2) if total_duration > 0 else 0.0,
        }

    except Exception as e:
        import sys

        print(f"Warning: Audio detection failed: {e}", file=sys.stderr)
        return {
            "audio_type_guess": "unknown",
            "audio_type_confidence": 0.0,
            "recommended_mode": "Song Hunter | Broadcast Hunter",
            "analysis_windows": 0,
        }


def _merge_window_votes(votes: List[Dict[str, Any]], total_duration: float) -> Dict[str, Any]:
    weighted: Dict[str, float] = {}
    for vote in votes:
        typ = str(vote.get("audio_type_guess", "unknown"))
        conf = max(0.05, float(vote.get("audio_type_confidence", 0.0) or 0.0))
        weighted[typ] = weighted.get(typ, 0.0) + conf

    if not weighted:
        return {"audio_type_guess": "unknown", "audio_type_confidence": 0.0}

    total_weight = sum(weighted.values()) or 1.0
    best_type = max(weighted.items(), key=lambda kv: kv[1])[0]
    best_conf = weighted[best_type] / total_weight

    # Guardrail: long tracks should almost never be classified as jingle/ad.
    if best_type == "jingle_ad" and total_duration >= 120.0:
        music_w = weighted.get("music", 0.0)
        mixed_w = weighted.get("mixed", 0.0)
        speech_w = weighted.get("speech", 0.0)
        if speech_w >= max(music_w, mixed_w):
            best_type = "speech"
            best_conf = max(speech_w / total_weight if total_weight > 0 else 0.0, 0.55)
        elif music_w >= mixed_w * 1.15:
            best_type = "music"
            base_conf = music_w / total_weight if total_weight > 0 else 0.0
            best_conf = max(base_conf, 0.55)
        else:
            best_type = "mixed"
            best_conf = max(mixed_w / total_weight if total_weight > 0 else 0.0, 0.55)

    return {
        "audio_type_guess": best_type,
        "audio_type_confidence": float(max(0.0, min(0.99, best_conf))),
    }


def _extract_audio_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Extract relevant audio features for classification."""
    features: Dict[str, float] = {}

    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
    features["spectral_centroid_std"] = float(np.std(spectral_centroids))

    # Zero crossing rate (speech tends to have higher ZCR)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # Tempo/beat strength (music tends to have stronger beat)
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = _safe_tempo_scalar(tempo)
        clip_dur = max(len(y) / float(sr), 1e-6)
        features["tempo"] = tempo_val
        features["beat_strength"] = float(len(beats) / clip_dur) if len(beats) > 0 else 0.0
    except Exception:
        features["tempo"] = 0.0
        features["beat_strength"] = 0.0

    # RMS energy variation
    rms = librosa.feature.rms(y=y)[0]
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))
    features["rms_variation"] = features["rms_std"] / (features["rms_mean"] + 1e-8)

    # Spectral rolloff (frequency distribution)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))

    flatness = librosa.feature.spectral_flatness(y=y)[0]
    features["spectral_flatness_mean"] = float(np.mean(flatness))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    features["onset_pulse"] = float(np.std(onset_env) / (np.mean(onset_env) + 1e-8)) if len(onset_env) > 0 else 0.0

    return features


def _classify_audio_type(features: Dict[str, float], clip_duration: float, full_duration: float) -> Tuple[str, float, Dict[str, float]]:
    """
    Classify audio type based on extracted features.
    Uses conservative, long-track-safe heuristics.
    
    Returns:
        (audio_type, confidence, score_map)
    """
    music_score = 0.0
    speech_score = 0.0
    beat_strength = float(features.get("beat_strength", 0.0))
    tempo = float(features.get("tempo", 0.0))
    zcr_std = float(features.get("zcr_std", 0.0))
    zcr_mean = float(features.get("zcr_mean", 0.0))
    rms_var = float(features.get("rms_variation", 0.0))
    centroid = float(features.get("spectral_centroid_mean", 0.0))
    flatness = float(features.get("spectral_flatness_mean", 0.0))
    onset_pulse = float(features.get("onset_pulse", 0.0))

    # Music cues
    if beat_strength >= 1.6:
        music_score += 0.32
    elif beat_strength >= 1.1:
        music_score += 0.24
    elif beat_strength >= 0.7:
        music_score += 0.12
    else:
        speech_score += 0.08

    if 70 <= tempo <= 180:
        music_score += 0.20
    elif tempo > 0:
        speech_score += 0.05

    if onset_pulse >= 0.55:
        music_score += 0.14
    elif onset_pulse <= 0.25:
        speech_score += 0.08

    if zcr_std <= 0.025:
        music_score += 0.10
    elif zcr_std >= 0.045:
        speech_score += 0.22

    if zcr_mean >= 0.11:
        speech_score += 0.10

    if rms_var <= 0.35:
        music_score += 0.12
    elif rms_var >= 0.55:
        speech_score += 0.18

    if centroid >= 3000:
        music_score += 0.12
    elif centroid <= 1900:
        speech_score += 0.12

    if flatness <= 0.015:
        music_score += 0.06
    elif flatness >= 0.030:
        speech_score += 0.08

    music_score = min(1.0, max(0.0, music_score))
    speech_score = min(1.0, max(0.0, speech_score))

    mixed_score = min(music_score, speech_score) * 0.9
    if abs(music_score - speech_score) <= 0.12 and max(music_score, speech_score) >= 0.38:
        mixed_score = min(1.0, mixed_score + 0.12)

    jingle_score = 0.0
    if (
        clip_duration <= 35.0
        and full_duration <= 120.0
        and music_score >= 0.50
        and speech_score >= 0.34
        and abs(music_score - speech_score) <= 0.24
    ):
        jingle_score = min(1.0, 0.45 * music_score + 0.55 * speech_score + 0.05)

    scores = {
        "music": music_score,
        "speech": speech_score,
        "mixed": mixed_score,
        "jingle_ad": jingle_score,
        "unknown": 0.2,
    }
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_type, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if best_score < 0.34:
        return "unknown", 0.35, scores

    margin = max(0.0, best_score - second_score)
    confidence = 0.42 + 0.45 * best_score + 0.35 * margin
    if best_type == "mixed":
        confidence = min(confidence, 0.85)
    confidence = max(0.40, min(0.98, confidence))

    return best_type, confidence, scores


def _recommend_mode(audio_type: str, confidence: float) -> str:
    """Recommend processing mode based on audio type."""
    if float(confidence or 0.0) < 0.65:
        return "Song Hunter | Broadcast Hunter (low confidence)"
    if audio_type == "music":
        return "Song Hunter (recommended) | Broadcast Hunter"
    elif audio_type == "speech":
        return "Broadcast Hunter (recommended)"
    elif audio_type == "mixed":
        return "Broadcast Hunter (recommended) | Song Hunter"
    elif audio_type == "jingle_ad":
        return "Broadcast Hunter (recommended)"
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
