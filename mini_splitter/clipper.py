# clipper.py
import subprocess
from pathlib import Path
from typing import List, Dict, Any

def export_clips_from_segments(
    wav_path: Path,
    segments: List[Dict[str, Any]],
    out_dir: Path,
    min_clip_sec: float,
    pad_sec: float,
    fade_ms: int,
    export_format: str = "wav",
):
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, seg in enumerate(segments, start=1):
        start = max(0.0, float(seg["start"]) - pad_sec)
        end = float(seg["end"]) + pad_sec
        dur = end - start
        if dur < min_clip_sec:
            continue

        text = _slug(seg.get("text", ""), max_len=48)
        out_path = out_dir / (f"clip_{idx:04d}_{text}.{export_format}" if text else f"clip_{idx:04d}.{export_format}")

        fade_s = fade_ms / 1000.0
        afade = []
        if fade_ms > 0:
            afade = [
                "-af",
                f"afade=t=in:st=0:d={fade_s},afade=t=out:st={max(0.0, dur - fade_s)}:d={fade_s}",
            ]

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", str(wav_path),
            "-vn",
            *afade,
            str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

def _slug(s: str, max_len: int = 48) -> str:
    s = (s or "").strip().lower()
    keep = []
    for ch in s:
        if ch.isalnum():
            keep.append(ch)
        elif ch in [" ", "_", "-"]:
            keep.append("_")
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out[:max_len].strip("_")
