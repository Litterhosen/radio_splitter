import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

from faster_whisper import WhisperModel

# Supported languages for explicit transcription
SUPPORTED_LANGUAGES = ("da", "en")


@dataclass
class WhisperState:
    model: WhisperModel
    model_size: str
    device: str
    compute_type: str


def load_model(model_size: str = "small", device: str = "cpu", compute_type: str = "int8") -> WhisperModel:
    """
    Load a Whisper model from HuggingFace Hub.
    
    If HF_TOKEN is set (via environment variable or Streamlit secrets), it will be used
    for authenticated requests to enable higher rate limits and faster downloads.
    
    Args:
        model_size: Size of the Whisper model (tiny, base, small, medium, large)
        device: Device to run on (cpu, cuda, auto)
        compute_type: Compute type for inference (int8, float16, float32)
    
    Returns:
        WhisperModel instance
    """
    # Try to get HF_TOKEN from environment or Streamlit secrets
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    # If not in environment, try Streamlit secrets (if available)
    if not hf_token:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
                hf_token = st.secrets["HF_TOKEN"]
        except (ImportError, FileNotFoundError, KeyError):
            # Streamlit not available or secrets not configured - continue without token
            pass
    
    # Set the token in environment if found (faster-whisper will use it)
    # Only set if token is non-empty string
    if hf_token and hf_token.strip():
        os.environ["HF_TOKEN"] = hf_token
    
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
