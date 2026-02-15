from dataclasses import dataclass
from typing import Optional, Dict, Any

import streamlit as st
from faster_whisper import WhisperModel

# Supported languages for explicit transcription
SUPPORTED_LANGUAGES = ("da", "en")


@dataclass
class WhisperState:
    model: WhisperModel
    model_size: str
    device: str
    compute_type: str


@st.cache_resource
def load_model(model_size: str = "small", device: str = "cpu", compute_type: str = "int8") -> WhisperModel:
    """Load Whisper model with caching to avoid reloading on every rerun"""
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_wav(model: WhisperModel, wav_path, language: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns dict with:
      - text
      - segments: [{start, end, text}, ...]
    
    Args:
        language: None for auto-detect, "da" for Danish, "en" for English
    """
    lang_arg = language if language in SUPPORTED_LANGUAGES else None
    segments, info = model.transcribe(
        str(wav_path),
        language=lang_arg,
        vad_filter=False,
        beam_size=5,
    )

    out_segments = []
    full = []
    for s in segments:
        txt = (s.text or "").strip()
        if txt:
            full.append(txt)
        out_segments.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": txt,
        })

    return {
        "text": " ".join(full).strip(),
        "language": getattr(info, "language", None),
        "segments": out_segments,
    }
