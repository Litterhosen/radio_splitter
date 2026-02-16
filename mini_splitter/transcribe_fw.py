import json
from pathlib import Path

from faster_whisper import WhisperModel


def transcribe_faster_whisper(
    wav_path: Path,
    model_size: str,
    language: str | None,
    beam_size: int,
    out_dir: Path,
) -> dict:
    """
    Outputs:
      - transcript.txt
      - transcript.json (segments w/ start/end)
      - transcript.srt
    """
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(
        str(wav_path),
        language=language,
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 250},
    )

    segment_list = []
    text_lines = []

    for seg in segments:
        segment_list.append(
            {"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()}
        )
        text_lines.append(seg.text.strip())

    transcript = {
        "language": info.language,
        "language_probability": float(info.language_probability),
        "segments": segment_list,
        "text": "\n".join(text_lines).strip(),
    }

    (out_dir / "transcript.txt").write_text(transcript["text"], encoding="utf-8")
    (out_dir / "transcript.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "transcript.srt").write_text(_to_srt(segment_list), encoding="utf-8")

    return transcript


def _fmt_time(timestamp_seconds: float) -> str:
    # SRT: HH:MM:SS,mmm
    ms = int(round(timestamp_seconds * 1000))
    h = ms // 3_600_000
    ms -= h * 3_600_000
    m = ms // 60_000
    ms -= m * 60_000
    s = ms // 1000
    ms -= s * 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def _to_srt(segments: list[dict]) -> str:
    out = []
    for i, seg in enumerate(segments, start=1):
        out.append(str(i))
        out.append(f"{_fmt_time(seg['start'])} --> {_fmt_time(seg['end'])}")
        out.append(seg["text"])
        out.append("")
    return "\n".join(out)
