# clipper.py
import subprocess
from pathlib import Path
from typing import List, Dict, Any

HARD_MIN_CLIP_SEC = 4.0


def _merge_text(a: str, b: str) -> str:
    a_clean = (a or "").strip()
    b_clean = (b or "").strip()
    if not a_clean:
        return b_clean
    if not b_clean:
        return a_clean
    return f"{a_clean} {b_clean}"


def _coalesce_short_segments(
    segments: List[Dict[str, Any]],
    min_clip_sec: float,
    max_gap_sec: float,
) -> List[Dict[str, Any]]:
    norm: List[Dict[str, Any]] = []
    for seg in segments:
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", 0.0) or 0.0)
        if end <= start:
            continue
        norm.append({"start": start, "end": end, "text": seg.get("text", "")})
    norm.sort(key=lambda s: (float(s["start"]), float(s["end"])))
    if not norm:
        return []

    merged: List[Dict[str, Any]] = [dict(norm[0])]
    for seg in norm[1:]:
        prev = merged[-1]
        gap = float(seg["start"]) - float(prev["end"])
        prev_dur = float(prev["end"]) - float(prev["start"])
        cur_dur = float(seg["end"]) - float(seg["start"])
        if gap <= max_gap_sec and (prev_dur < min_clip_sec or cur_dur < min_clip_sec):
            prev["end"] = max(float(prev["end"]), float(seg["end"]))
            prev["text"] = _merge_text(str(prev.get("text", "")), str(seg.get("text", "")))
        else:
            merged.append(dict(seg))

    # Second pass: absorb any remaining short segments into nearest neighbor when close enough.
    i = 0
    soft_gap = max(max_gap_sec * 2.0, 0.8)
    while i < len(merged):
        dur = float(merged[i]["end"]) - float(merged[i]["start"])
        if dur >= min_clip_sec or len(merged) == 1:
            i += 1
            continue

        left_gap = float("inf")
        right_gap = float("inf")
        if i > 0:
            left_gap = float(merged[i]["start"]) - float(merged[i - 1]["end"])
        if i + 1 < len(merged):
            right_gap = float(merged[i + 1]["start"]) - float(merged[i]["end"])

        if i > 0 and left_gap <= soft_gap and (i + 1 >= len(merged) or left_gap <= right_gap):
            merged[i - 1]["end"] = max(float(merged[i - 1]["end"]), float(merged[i]["end"]))
            merged[i - 1]["text"] = _merge_text(str(merged[i - 1].get("text", "")), str(merged[i].get("text", "")))
            del merged[i]
            continue

        if i + 1 < len(merged) and right_gap <= soft_gap:
            merged[i + 1]["start"] = min(float(merged[i + 1]["start"]), float(merged[i]["start"]))
            merged[i + 1]["text"] = _merge_text(str(merged[i].get("text", "")), str(merged[i + 1].get("text", "")))
            del merged[i]
            continue

        i += 1

    return merged


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
    effective_min_clip_sec = max(HARD_MIN_CLIP_SEC, float(min_clip_sec))
    merged_segments = _coalesce_short_segments(
        segments=segments,
        min_clip_sec=effective_min_clip_sec,
        max_gap_sec=max(0.35, float(pad_sec) * 2.0),
    )

    for idx, seg in enumerate(merged_segments, start=1):
        start = max(0.0, float(seg["start"]) - pad_sec)
        end = float(seg["end"]) + pad_sec
        dur = end - start
        if dur < effective_min_clip_sec:
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
