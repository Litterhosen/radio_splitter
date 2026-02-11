from dataclasses import dataclass
from typing import Optional, Dict, Any

from faster_whisper import WhisperModel


@dataclass
class WhisperState:
    model: WhisperModel
    model_size: str
    device: str
    compute_type: str


def load_model(model_size: str = "small", device: str = "cpu", compute_type: str = "int8") -> WhisperModel:
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_wav(model: WhisperModel, wav_path, language: Optional[str] = "da") -> Dict[str, Any]:
    """
    Returns dict with:
      - text
      - segments: [{start, end, text}, ...]
    """
    segments, info = model.transcribe(
        str(wav_path),
        language=language if language in ("da", "en") else None,
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
