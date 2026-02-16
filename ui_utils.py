"""
UI helper functions for the Sample Machine.
"""
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import streamlit as st

from config import THEMES, OVERLAP_THRESHOLD, MIN_DURATION_SECONDS, OUTPUT_ROOT, DECAY_TAIL_DURATION
from utils import ensure_dir, safe_dirname
from audio_split import cut_segment_to_wav, cut_segment_to_mp3, cut_segment_with_fades


def bars_ui_to_int(bars_ui: str) -> int:
    """Convert UI bar string to integer value"""
    bars_mapping = {
        "1 bar": 1, 
        "2 bars": 2, 
        "4 bars": 4, 
        "8 bars": 8, 
        "16 bars": 16
    }
    return bars_mapping.get(bars_ui, 2)  # Default to 2 if unknown


def get_whisper_language() -> Optional[str]:
    """Convert UI language to whisper language code"""
    mapping = {"Auto": None, "Dansk": "da", "English": "en"}
    language_ui = st.session_state.get("whisper_language_ui", "Auto")
    return mapping.get(language_ui, None)


def detect_themes(text: str) -> List[str]:
    """Scan transcript for bilingual theme keywords"""
    text_lower = (text or "").lower()
    found = []
    for theme, keywords in THEMES.items():
        if any(kw in text_lower for kw in keywords):
            found.append(theme)
    return found


def overlap_ratio(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Calculate overlap ratio between two intervals"""
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    if right <= left:
        return 0.0
    intersection = right - left
    shortest = min(a[1] - a[0], b[1] - b[0])
    if shortest <= 0:
        return 0.0
    return intersection / shortest


def anti_overlap_keep_best(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove overlapping candidates, keeping the highest scoring ones"""
    kept = []
    for item in sorted(candidates, key=lambda x: float(x["score"]), reverse=True):
        rng = (float(item["start"]), float(item["end"]))
        if any(overlap_ratio(rng, (float(k["start"]), float(k["end"]))) > OVERLAP_THRESHOLD for k in kept):
            continue
        kept.append(item)
    return kept


def filter_by_duration(candidates: List[Dict[str, Any]], min_duration: float = MIN_DURATION_SECONDS) -> List[Dict[str, Any]]:
    """Filter out candidates shorter than min_duration seconds"""
    return [c for c in candidates if (float(c["end"]) - float(c["start"])) >= min_duration]


def save_input_to_session_dir(
    src_type: str, 
    name_or_path: str, 
    maybe_bytes: Optional[bytes], 
    youtube_info: Optional[Dict[str, Any]] = None, 
    run_id: str = "run"
) -> Tuple[Path, Path, str, Dict[str, Any]]:
    """Save uploaded or downloaded file to a run-scoped session directory"""
    if src_type == "upload":
        in_name = name_or_path
        # Use safe_dirname for session directory
        safe_name = safe_dirname(Path(in_name).stem)
        session_dir = ensure_dir(OUTPUT_ROOT / f"{safe_name}__{run_id}")
        in_path = session_dir / in_name
        in_path.write_bytes(maybe_bytes)
        return in_path, session_dir, in_name, youtube_info or {}

    p = Path(name_or_path)
    in_name = p.name
    # Use safe_dirname for session directory
    safe_name = safe_dirname(p.stem)
    session_dir = ensure_dir(OUTPUT_ROOT / f"{safe_name}__{run_id}")
    local_in = session_dir / in_name
    if not local_in.exists():
        local_in.write_bytes(p.read_bytes())
    return local_in, session_dir, in_name, youtube_info or {}


def export_clip_with_tail(
    in_path: Path, 
    session_dir: Path, 
    stem: str, 
    a: float, 
    b: float, 
    want_format: str,
    add_tail: bool = True,
    add_fades: bool = True
) -> Tuple[Path, Dict[str, Any]]:
    """
    Export clip with optional decay tail and fades.
    
    Returns:
        Tuple of (clip_path, export_metadata_dict)
    """
    tail_duration = DECAY_TAIL_DURATION if add_tail else 0.0
    
    # Determine file extension
    is_wav = want_format.startswith("wav")
    ext = "wav" if is_wav else "mp3"
    suffix = "_tail" if add_tail else ""
    outp = session_dir / f"{stem}{suffix}.{ext}"
    
    # Use new export function with fades if enabled for Song Hunter
    # Standard pre-roll, fade-in, and fade-out durations are specified in milliseconds
    if add_fades:
        export_meta = cut_segment_with_fades(
            in_path, 
            outp, 
            core_start=a, 
            core_end=b,
            pre_roll_ms=25.0,  # Standard pre-roll duration in milliseconds
            fade_in_ms=15.0,   # Standard fade-in duration in milliseconds
            fade_out_ms=15.0,  # Standard fade-out duration in milliseconds
            tail_sec=tail_duration,
            apply_zero_crossing=True,
            bitrate="192k",
            is_wav=is_wav
        )
        return outp, export_meta
    else:
        # Legacy export without fades for broadcast mode
        b_with_tail = b + tail_duration
        
        if is_wav:
            cut_segment_to_wav(in_path, outp, a, b_with_tail)
        else:
            cut_segment_to_mp3(in_path, outp, a, b_with_tail, bitrate="192k")
        
        # Return simple metadata for backward compatibility
        export_meta = {
            "core_start_sec": float(a),
            "core_end_sec": float(b),
            "core_dur_sec": float(b - a),
            "export_start_sec": float(a),
            "export_end_sec": float(b_with_tail),
            "export_dur_sec": float(b_with_tail - a),
            "pre_roll_ms": 0.0,
            "fade_in_ms": 0.0,
            "fade_out_ms": 0.0,
            "tail_sec": float(tail_duration),
            "zero_cross_applied": False,
        }
        return outp, export_meta


def build_groups(selected_rows: List[Dict[str, Any]], group_mode: str) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """Build grouped rows for preview rendering."""
    if group_mode == "none":
        return None

    groups: Dict[str, List[Dict[str, Any]]] = {}

    for row in selected_rows:
        if group_mode == "phrase":
            key = row.get("clip_text_signature", "") or "[No text]"
        elif group_mode == "tag_theme":
            tags = str(row.get("tags", "")).split(", ")
            themes = str(row.get("themes", "")).split(", ")
            combined = [t.strip() for t in (tags + themes) if t.strip()]
            key = ", ".join(combined[:3]) if combined else "[No tags]"
        elif group_mode == "language":
            key = row.get("language_guess_clip", row.get("language_detected", "unknown"))
        else:
            return None

        groups.setdefault(key, []).append(row)

    return groups
