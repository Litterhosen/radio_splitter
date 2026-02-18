# transcribe.py
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from faster_whisper import WhisperModel

def transcribe_audio(
    wav_path: Path,
    out_dir: Path,
    backend: str,
    model_size: str,
    language: Optional[str],
    beam_size: int,
) -> Dict[str, Any]:
    """
    Creates:
      - transcript.txt
      - transcript.json (segments with start/end)
      - transcript.srt
    Optional (WhisperX):
      - transcript_words.json (word-level timings)
    Returns dict with at least: language, language_probability, segments, text
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if backend == "whisperx":
        try:
            return _transcribe_whisperx(
                wav_path=wav_path,
                out_dir=out_dir,
                model_size=model_size,
                language=language,
            )
        except Exception as e:
            # Fallback to faster-whisper
            return _transcribe_faster_whisper(
                wav_path=wav_path,
                out_dir=out_dir,
                model_size=model_size,
                language=language,
                beam_size=beam_size,
                note=f"WhisperX failed or missing; fell back to faster-whisper. ({type(e).__name__})",
            )

    return _transcribe_faster_whisper(
        wav_path=wav_path,
        out_dir=out_dir,
        model_size=model_size,
        language=language,
        beam_size=beam_size,
        note=None,
    )

def _transcribe_faster_whisper(
    wav_path: Path,
    out_dir: Path,
    model_size: str,
    language: Optional[str],
    beam_size: int,
    note: Optional[str],
) -> Dict[str, Any]:
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(
        str(wav_path),
        language=language,
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 250},
    )

    seg_list: List[Dict[str, Any]] = []
    text_lines: List[str] = []

    for s in segments:
        t = s.text.strip()
        seg_list.append({"start": float(s.start), "end": float(s.end), "text": t})
        if t:
            text_lines.append(t)

    transcript = {
        "engine": "faster-whisper",
        "note": note,
        "language": info.language,
        "language_probability": float(info.language_probability),
        "segments": seg_list,
        "text": "\n".join(text_lines).strip(),
    }

    (out_dir / "transcript.txt").write_text(transcript["text"], encoding="utf-8")
    (out_dir / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "transcript.srt").write_text(_to_srt(seg_list), encoding="utf-8")
    return transcript

def _transcribe_whisperx(
    wav_path: Path,
    out_dir: Path,
    model_size: str,
    language: Optional[str],
) -> Dict[str, Any]:
    """
    WhisperX path: word-level alignment.
    Requires: pip install whisperx
    """
    import whisperx  # type: ignore

    device = "cpu"
    compute_type = "int8"

    # 1) ASR
    model = whisperx.load_model(model_size, device=device, compute_type=compute_type, language=language)
    audio = whisperx.load_audio(str(wav_path))
    result = model.transcribe(audio, batch_size=8)

    # 2) Align to get word timestamps
    lang = result.get("language", language or "unknown")
    align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
    aligned = whisperx.align(result["segments"], align_model, metadata, audio, device)

    # Convert to our standard segments
    seg_list: List[Dict[str, Any]] = []
    text_lines: List[str] = []
    for s in aligned["segments"]:
        t = (s.get("text") or "").strip()
        seg_list.append({"start": float(s["start"]), "end": float(s["end"]), "text": t})
        if t:
            text_lines.append(t)

    transcript = {
        "engine": "whisperx",
        "language": lang,
        "language_probability": None,
        "segments": seg_list,
        "text": "\n".join(text_lines).strip(),
    }

    # Save transcript outputs
    (out_dir / "transcript.txt").write_text(transcript["text"], encoding="utf-8")
    (out_dir / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "transcript.srt").write_text(_to_srt(seg_list), encoding="utf-8")

    # Save word-level timings if present
    words = []
    for s in aligned["segments"]:
        for w in s.get("words", []) or []:
            words.append({
                "start": w.get("start"),
                "end": w.get("end"),
                "word": w.get("word"),
                "score": w.get("score"),
            })
    (out_dir / "transcript_words.json").write_text(json.dumps({"language": lang, "words": words}, ensure_ascii=False, indent=2), encoding="utf-8")

    return transcript

def _fmt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h = ms // 3600000
    ms -= h * 3600000
    m = ms // 60000
    ms -= m * 60000
    s = ms // 1000
    ms -= s * 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def _to_srt(segments: List[Dict[str, Any]]) -> str:
    out = []
    for i, seg in enumerate(segments, start=1):
        out.append(str(i))
        out.append(f"{_fmt_time(seg['start'])} --> {_fmt_time(seg['end'])}")
        out.append(seg.get("text", ""))
        out.append("")
    return "\n".join(out)
