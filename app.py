# CRITICAL: st.set_page_config MUST be the VERY FIRST Streamlit call
import streamlit as st

# Version number
VERSION = "1.1.6"

st.set_page_config(page_title=f"The Sample Machine v{VERSION}", layout="wide")

import io
import json
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd

# Local imports
from audio_split import (
    detect_non_silent_intervals,
    fixed_length_intervals,
    cut_segment_to_wav,
    cut_segment_to_mp3,
    cut_segment_with_fades,
)
from transcribe import load_model, transcribe_wav
from downloaders import download_audio, DownloadError
from tagging import auto_tags
from jingle_finder import jingle_score
from hook_finder import ffmpeg_to_wav16k_mono, find_hooks
from beat_refine import refine_best_1_or_2_bars
from utils import ensure_dir, hhmmss_ms, mmss, safe_slug, safe_dirname, clip_uid, estimate_bars_from_duration, snap_bars_to_valid, extract_track_metadata

# ----------------------------
# Constants and Config
# ----------------------------
OUTPUT_ROOT = Path("output")
ensure_dir(OUTPUT_ROOT)

# Anti-overlap and filter thresholds
OVERLAP_THRESHOLD = 0.30  # 30% overlap threshold for duplicate detection
MIN_DURATION_SECONDS = 4.0  # Minimum clip duration
DECAY_TAIL_DURATION = 0.75  # Extra audio tail for loops (seconds)
MAX_SLUG_LENGTH = 24  # Maximum characters for slug in filename

MODE_OPTIONS = [
    "🎵 Song Hunter (Loops)",
    "📻 Broadcast Hunter (Mix)",
]

# Bilingual theme keywords (DA + EN)
THEMES = {
    "THEME:TIME": ["tid", "evighed", "nu", "time", "eternity", "now"],
    "THEME:MEMORY": ["huske", "glemme", "remember", "forget", "back"],
    "THEME:DREAM": ["drøm", "natten", "dream", "night", "sleep"],
    "THEME:EXISTENTIAL": ["livet", "verden", "cirkel", "life", "world", "circle"],
    "THEME:META": ["radio", "musik", "stemme", "lyd", "music", "voice", "sound"],
}

DEFAULTS = {
    "mode": MODE_OPTIONS[0],
    "model_size": "small",
    "whisper_language_ui": "Auto",
    "device": "cpu",
    "compute_type": "int8",
    "noise_db": -28.0,
    "min_silence_s": 0.4,
    "pad_s": 0.15,
    "min_segment_s": 1.2,
    "fixed_len": 8.0,
    "hook_len_range_min": 4.0,
    "hook_len_range_max": 15.0,
    "prefer_len": 8.0,
    "hook_hop": 1.0,
    "hook_topn": 12,
    "hook_gap": 2.0,
    "chorus_len_range_min": 30.0,
    "chorus_len_range_max": 45.0,
    "chorus_hop": 2.0,
    "chorus_topn": 4,
    "chorus_gap": 8.0,
    "loops_per_chorus": 10,
    "beat_refine": True,
    "beats_per_bar": 4,
    "prefer_bars": 2,
    "prefer_bars_ui": "2 bars",
    "try_both_bars": True,
    "use_slug": True,
    "slug_words": 6,
    "export_format": "mp3 (192k)",
    "downloaded_files": [],
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ----------------------------
# Helper Functions
# ----------------------------
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


def get_whisper_language():
    """Convert UI language to whisper language code"""
    mapping = {"Auto": None, "Dansk": "da", "English": "en"}
    return mapping.get(st.session_state["whisper_language_ui"], None)


def detect_themes(text: str) -> List[str]:
    """Scan transcript for bilingual theme keywords"""
    text_lower = (text or "").lower()
    found = []
    for theme, keywords in THEMES.items():
        if any(kw in text_lower for kw in keywords):
            found.append(theme)
    return found


def overlap_ratio(a: tuple, b: tuple) -> float:
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


def anti_overlap_keep_best(candidates: List[Dict]) -> List[Dict]:
    """Remove overlapping candidates, keeping the highest scoring ones"""
    kept = []
    for item in sorted(candidates, key=lambda x: float(x["score"]), reverse=True):
        rng = (float(item["start"]), float(item["end"]))
        if any(overlap_ratio(rng, (float(k["start"]), float(k["end"]))) > OVERLAP_THRESHOLD for k in kept):
            continue
        kept.append(item)
    return kept


def filter_by_duration(candidates: List[Dict], min_duration: float = MIN_DURATION_SECONDS) -> List[Dict]:
    """Filter out candidates shorter than min_duration seconds"""
    return [c for c in candidates if (float(c["end"]) - float(c["start"])) >= min_duration]


def save_input_to_session_dir(src_type: str, name_or_path: str, maybe_bytes, youtube_info=None):
    """Save uploaded or downloaded file to session directory"""
    if src_type == "upload":
        in_name = name_or_path
        # Use safe_dirname for session directory
        safe_name = safe_dirname(Path(in_name).stem)
        session_dir = ensure_dir(OUTPUT_ROOT / safe_name)
        in_path = session_dir / in_name
        in_path.write_bytes(maybe_bytes)
        return in_path, session_dir, in_name, youtube_info or {}

    p = Path(name_or_path)
    in_name = p.name
    # Use safe_dirname for session directory
    safe_name = safe_dirname(p.stem)
    session_dir = ensure_dir(OUTPUT_ROOT / safe_name)
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
) -> Tuple[Path, dict]:
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
    if add_fades:
        export_meta = cut_segment_with_fades(
            in_path, 
            outp, 
            core_start=a, 
            core_end=b,
            pre_roll_ms=25.0,
            fade_in_ms=15.0,
            fade_out_ms=15.0,
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


def maybe_refine_barloop(wav_for_analysis: Path, a: float, b: float):
    """Refine loop to beat grid if enabled"""
    if not st.session_state["beat_refine"]:
        dur = b - a
        # Estimate bars even when disabled
        bpm_estimate = 120  # Default fallback
        bars_est = estimate_bars_from_duration(dur, bpm_estimate, 4)
        return a, b, 0, None, 0.0, False, "disabled", bars_est, 0.0

    beats_per_bar = int(st.session_state["beats_per_bar"])
    prefer_bars = int(st.session_state["prefer_bars"])

    rr = refine_best_1_or_2_bars(
        str(wav_for_analysis),
        window_start=float(a),
        window_end=float(b),
        beats_per_bar=beats_per_bar,
        prefer_bars=prefer_bars,
        sr=22050,
    )

    if rr.ok and (rr.end - rr.start) > 0.5:
        # Fix double-offset bug: use original window_start as base
        return (
            float(a) + rr.start,
            float(a) + rr.end,
            rr.bpm,
            rr.bars,
            rr.score,
            True,
            "",
            rr.bars_estimated,
            rr.bpm_confidence
        )

    return a, b, rr.bpm, None, rr.score, False, rr.reason, rr.bars_estimated, rr.bpm_confidence


# ----------------------------
# UI - Title and Description
# ----------------------------
st.title(f"🎛️ The Sample Machine v{VERSION}")
st.caption("Bilingual audio splitter with whisper transcription, hook detection, and theme tagging")

# ----------------------------
# UI - Sidebar Settings
# ----------------------------
st.sidebar.header("⚙️ Settings")

# Mode selection
mode_idx = MODE_OPTIONS.index(st.session_state["mode"]) if st.session_state["mode"] in MODE_OPTIONS else 0
st.session_state["mode"] = st.sidebar.selectbox("Mode", MODE_OPTIONS, index=mode_idx)

# Whisper settings
st.sidebar.subheader("🎙️ Whisper")
st.sidebar.selectbox("Model size", ["tiny", "base", "small", "medium"], key="model_size")
st.sidebar.selectbox("Language", ["Auto", "Dansk", "English"], key="whisper_language_ui")
st.sidebar.selectbox("Device", ["cpu"], key="device")
st.sidebar.selectbox("Compute type", ["int8", "float32"], key="compute_type")

# Export settings
st.sidebar.subheader("📤 Export")
st.sidebar.selectbox("Format", ["wav (16000 mono)", "mp3 (192k)"], key="export_format")
st.sidebar.checkbox("Add slug from transcript", key="use_slug")
st.sidebar.slider("Slug words", 2, 12, key="slug_words")

# Mode-specific settings
if st.session_state["mode"] == "📻 Broadcast Hunter (Mix)":
    st.sidebar.subheader("📻 Broadcast Split")
    st.sidebar.slider("Silence threshold (dB)", -60.0, -10.0, step=1.0, key="noise_db")
    st.sidebar.slider("Min silence (sec)", 0.2, 2.0, step=0.1, key="min_silence_s")
    st.sidebar.slider("Padding (sec)", 0.0, 1.0, step=0.05, key="pad_s")
    st.sidebar.slider("Min segment (sec)", 0.5, 10.0, step=0.1, key="min_segment_s")
else:
    st.sidebar.subheader("🎵 Hook Detection")
    st.sidebar.slider("Min hook length (sec)", 2.0, 30.0, step=0.5, key="hook_len_range_min")
    st.sidebar.slider("Max hook length (sec)", 2.0, 30.0, step=0.5, key="hook_len_range_max")
    
    # Validate that min <= max
    if st.session_state["hook_len_range_min"] > st.session_state["hook_len_range_max"]:
        st.sidebar.error("⚠️ Min hook length cannot exceed max hook length")
    
    st.sidebar.slider("Preferred length (sec)", 2.0, 20.0, step=0.5, key="prefer_len")
    st.sidebar.slider("Scan hop (sec)", 0.25, 5.0, step=0.25, key="hook_hop")
    st.sidebar.slider("Top N hooks", 3, 30, step=1, key="hook_topn")
    st.sidebar.slider("Min gap between hooks (sec)", 0.0, 10.0, step=0.5, key="hook_gap")
    
    st.sidebar.subheader("🎼 Beat Refinement")
    st.sidebar.checkbox("Refine to beat grid", key="beat_refine")
    st.sidebar.number_input("Beats per bar", min_value=3, max_value=7, step=1, key="beats_per_bar")
    bars_options = ["1 bar", "2 bars", "4 bars", "8 bars", "16 bars"]
    
    # Get current value from session state or use default
    current_bars_ui = st.session_state.get("prefer_bars_ui", "2 bars")
    try:
        default_idx = bars_options.index(current_bars_ui)
    except ValueError:
        default_idx = 1  # Default to "2 bars"
    
    # Use radio without default parameter to avoid conflict
    selected_bars_ui = st.sidebar.radio("Preferred loop length", bars_options, index=default_idx, key="prefer_bars_selector")
    
    # Update session state from widget
    st.session_state["prefer_bars_ui"] = selected_bars_ui
    st.session_state["prefer_bars"] = bars_ui_to_int(selected_bars_ui)
    st.sidebar.checkbox("Try both (choose best)", key="try_both_bars")

# ----------------------------
# UI - Input Tabs
# ----------------------------
st.subheader("📥 Input")
tab_upload, tab_link = st.tabs(["📂 Upload Filer", "🔗 Hent fra Link"])

with tab_upload:
    files = st.file_uploader(
        "Upload MP3/WAV files",
        type=["mp3", "wav"],
        accept_multiple_files=True,
        help="Select one or more audio files"
    )

with tab_link:
    url = st.text_input("URL", placeholder="https://... (YouTube, Archive, etc.)")
    dl_col1, dl_col2 = st.columns([1, 3])
    with dl_col1:
        dl_btn = st.button("Download", type="primary", disabled=not url.strip())
    with dl_col2:
        st.caption("Downloads to: output/Downloads/")
    
    if dl_btn:
        try:
            with st.spinner("Downloading..."):
                p, youtube_info = download_audio(url.strip(), OUTPUT_ROOT / "Downloads")
            st.success(f"✅ Downloaded: {p.name}")
            # Store both path and YouTube info
            download_entry = {"path": str(p), "youtube_info": youtube_info}
            st.session_state["downloaded_files"] = list(st.session_state.get("downloaded_files", [])) + [download_entry]
        except DownloadError as e:
            st.error(f"❌ Download failed: {e}")
            if e.log_file:
                st.error(f"📄 Log file saved: `{e.log_file}`")
                try:
                    with open(e.log_file, 'r') as f:
                        log_content = f.read()
                    with st.expander("📋 View full log"):
                        st.code(log_content, language="text")
                except Exception as read_error:
                    st.warning(f"Could not read log file: {read_error}")
            if e.hint:
                st.warning(f"💡 Hint: {e.hint}")
            if e.last_error:
                with st.expander("🔍 Technical details"):
                    st.code(e.last_error, language="text")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

# Prepare input paths
input_paths = []
if files:
    for uf in files:
        input_paths.append(("upload", uf.name, uf.getvalue(), None))
for entry in st.session_state.get("downloaded_files", []):
    # Handle both old format (str) and new format (dict)
    if isinstance(entry, dict):
        input_paths.append(("path", entry["path"], None, entry.get("youtube_info", {})))
    else:
        input_paths.append(("path", entry, None, None))

# ----------------------------
# UI - Actions
# ----------------------------
st.divider()
if "model" not in st.session_state:
    st.session_state.model = None

col1, col2 = st.columns([1, 1])
with col1:
    load_btn = st.button("🔧 Load Whisper Model", type="secondary", use_container_width=True)
with col2:
    run_btn = st.button("▶️ Process", type="primary", disabled=not input_paths, use_container_width=True)

if load_btn:
    try:
        with st.spinner("Loading model... (first time downloads it)"):
            st.session_state.model = load_model(
                model_size=st.session_state["model_size"],
                device=st.session_state["device"],
                compute_type=st.session_state["compute_type"],
            )
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")

# ----------------------------
# Processing Logic
# ----------------------------
if run_btn:
    mode_now = st.session_state["mode"]
    
    # Check if model is needed and loaded
    if st.session_state.model is None:
        st.warning("⚠️ Please click 'Load Whisper Model' first!")
        st.stop()
    
    with st.status("🚀 Starting processing...", expanded=True) as status:
        results = []
        lang = get_whisper_language()
        total_files = len(input_paths)

        for idx_file, (src_type, name_or_path, maybe_bytes, youtube_info) in enumerate(input_paths):
            status.update(
                label=f"📦 Processing file {idx_file+1} of {total_files}",
                state="running"
            )
            
            st.write(f"### File {idx_file+1}/{total_files}: {name_or_path}")
            
            # Step 1: Save and convert to analysis format
            in_path, session_dir, in_name, yt_info = save_input_to_session_dir(src_type, name_or_path, maybe_bytes, youtube_info)
            
            # Extract track metadata (artist, title, track_id)
            track_artist, track_title, track_id = extract_track_metadata(in_path, yt_info)
            st.write(f"🎵 Track: **{track_artist} - {track_title}**")
            
            st.write("⏳ Converting audio...")
            wav16 = session_dir / "_analysis_16k_mono.wav"
            ffmpeg_to_wav16k_mono(in_path, wav16)
            st.write("✅ Conversion complete.")
            
            # ----------------------------
            # Mode: Song Hunter (Loops)
            # ----------------------------
            if mode_now == "🎵 Song Hunter (Loops)":
                st.write("🎵 Finding hooks...")
                
                # Get prefer_bars from session state
                prefer_bars = int(st.session_state.get("prefer_bars", 2))
                beats_per_bar = int(st.session_state.get("beats_per_bar", 4))
                
                hooks, global_bpm, global_confidence = find_hooks(
                    wav16,
                    hook_len_range=(
                        st.session_state["hook_len_range_min"],
                        st.session_state["hook_len_range_max"]
                    ),
                    prefer_len=st.session_state["prefer_len"],
                    hop_s=st.session_state["hook_hop"],
                    topn=st.session_state["hook_topn"],
                    min_gap_s=st.session_state["hook_gap"],
                    prefer_bars=prefer_bars,
                    beats_per_bar=beats_per_bar,
                )
                st.write(f"🎉 Found {len(hooks)} potential hooks!")
                st.write(f"🎼 Global BPM: {global_bpm} (confidence: {global_confidence:.2f})")
                
                # Convert to candidate format
                candidates = []
                for h in hooks:
                    candidates.append({
                        "start": float(h.start),
                        "end": float(h.end),
                        "score": float(h.score),
                        "hook_score": float(h.score),
                        "energy": float(h.energy),
                        "loopability": float(h.loopability),
                        "bpm": float(h.bpm),
                        "bpm_confidence": float(h.bpm_confidence),
                        "bpm_source": h.bpm_source,
                        "bpm_clip_confidence": float(h.bpm_clip_confidence),
                    })
                
                # Apply minimum duration filter
                min_duration = st.session_state["hook_len_range_min"]
                candidates = filter_by_duration(candidates, min_duration=min_duration)
                st.write(f"✂️ After {min_duration}s filter: {len(candidates)} hooks")
                
                # Apply anti-overlap
                candidates = anti_overlap_keep_best(candidates)
                st.write(f"🔄 After anti-overlap: {len(candidates)} hooks")
                
                # Process each hook
                progress_bar = st.progress(0)
                total_hooks = len(candidates)
                
                for idx, cand in enumerate(candidates, start=1):
                    progress_bar.progress(idx / total_hooks)
                    
                    a, b = float(cand["start"]), float(cand["end"])
                    
                    # Beat refinement
                    a2, b2, bpm_refined, bars, rscore, refined_ok, rreason, bars_est, bpm_conf = maybe_refine_barloop(wav16, a, b)
                    aa, bb = (a2, b2) if refined_ok else (a, b)
                    dur = max(0.0, bb - aa)
                    
                    # Fallback to original if refined clip is too short
                    min_duration = st.session_state["hook_len_range_min"]
                    if refined_ok and dur < min_duration:
                        aa, bb = a, b
                        dur = max(0.0, bb - aa)
                        refined_ok = False
                        rreason = "too_short"
                    
                    # Get preference for bar snapping
                    prefer_bars = int(st.session_state.get("prefer_bars", 2))
                    
                    # Determine BPM and bars with proper tracking
                    bpm_clip = int(cand.get("bpm", global_bpm))
                    bpm_clip_confidence = cand.get("bpm_clip_confidence", 0.0)
                    
                    if refined_ok:
                        # Refined: use beat-grid aligned values
                        bpm_used = bpm_refined
                        bpm_used_source = "refined_grid"
                        raw_bars_estimate = bars_est  # Raw estimate before grid alignment
                        bars_used = bars  # Grid-aligned bars
                        final_bpm_conf = bpm_conf
                        refined_reason = ""
                    else:
                        # Not refined: use estimates with snapping
                        # Calculate raw bars estimate
                        if bars_est > 0:
                            raw_bars_estimate = bars_est
                        else:
                            raw_bars_estimate = estimate_bars_from_duration(dur, bpm_clip, 4)
                        
                        # Snap bars to valid values
                        bars_used = snap_bars_to_valid(raw_bars_estimate, prefer_bars, tolerance=0.25)
                        
                        # Determine BPM to use: prefer global if high confidence
                        if cand.get("bpm_source") == "track_global":
                            bpm_used = bpm_clip  # Already normalized to global
                            bpm_used_source = "global_snapped"
                            final_bpm_conf = cand.get("bpm_confidence", global_confidence)
                        else:
                            bpm_used = bpm_clip
                            bpm_used_source = "segment_estimate"
                            final_bpm_conf = bpm_clip_confidence
                        
                        # Use the reason from refinement attempt (disabled, too_short, no_onsets, etc.)
                        refined_reason = rreason
                    
                    # Generate UID
                    uid = clip_uid(in_name, aa, bb, idx)
                    
                    # Transcribe
                    temp_wav = session_dir / f"temp_{idx:04d}__whisper.wav"
                    cut_segment_to_wav(in_path, temp_wav, aa, bb)
                    tjson = transcribe_wav(st.session_state.model, temp_wav, language=lang)
                    text = (tjson.get("text") or "").strip()
                    
                    # Generate slug (max 24 chars, use __noslug if empty)
                    slug = ""
                    if st.session_state["use_slug"] and text:
                        slug = safe_slug(" ".join(text.split()[:int(st.session_state["slug_words"])]), max_len=MAX_SLUG_LENGTH)
                    
                    # New filename template: {artist} - {title}__{idx:04d}__{bpm_used}bpm__{bars_used}bar__{start_mmss}-{end_mmss}__{slug}__{uid6}_tail.mp3
                    start_mmss = mmss(aa)
                    end_mmss = mmss(bb)
                    
                    # Build BPM/bars parts
                    if refined_ok:
                        bpm_part = f"{bpm_used}bpm"
                        bars_part = f"{bars_used}bar"
                    else:
                        if final_bpm_conf < 0.4:
                            bpm_part = f"{bpm_used}bpm_low"
                        else:
                            bpm_part = f"{bpm_used}bpm_est"
                        bars_part = f"{bars_used}bar_est"
                    
                    # slug_part
                    slug_part = slug if slug else "noslug"
                    
                    # Build complete filename stem
                    stem = f"{track_artist} - {track_title}__{idx:04d}__{bpm_part}__{bars_part}__{start_mmss}-{end_mmss}__{slug_part}__{uid}"
                    
                    # Export with tail
                    clip_path, export_meta = export_clip_with_tail(
                        in_path, session_dir, stem, aa, bb,
                        st.session_state["export_format"],
                        add_tail=True,
                        add_fades=True
                    )
                    
                    # Save metadata
                    txt_path = session_dir / f"{stem}.txt"
                    json_path = session_dir / f"{stem}.json"
                    txt_path.write_text((text or "") + "\n", encoding="utf-8")
                    json_path.write_text(
                        json.dumps(tjson, ensure_ascii=False, indent=2),
                        encoding="utf-8"
                    )
                    
                    # Tags and themes
                    tags = ["musik", "hook"]
                    if bars_used is not None:
                        tags.append(f"{bars_used}bar")
                    if not refined_ok:
                        tags.append("unrefined")
                    if text:
                        tags = list(set(tags + auto_tags(text)))
                    themes = detect_themes(text)
                    
                    results.append({
                        "source": in_name,
                        "filename": clip_path.name,
                        "filename_stem": stem,
                        "filename_template_version": "v2",
                        "track_artist": track_artist,
                        "track_title": track_title,
                        "track_id": track_id,
                        "session_dir": str(session_dir.name),
                        "pick": True,
                        "clip": idx,
                        "start_sec": aa,
                        "end_sec": bb,
                        "dur_sec": dur,
                        "core_start_sec": export_meta["core_start_sec"],
                        "core_end_sec": export_meta["core_end_sec"],
                        "core_dur_sec": export_meta["core_dur_sec"],
                        "export_start_sec": export_meta["export_start_sec"],
                        "export_end_sec": export_meta["export_end_sec"],
                        "export_dur_sec": export_meta["export_dur_sec"],
                        "pre_roll_ms": export_meta["pre_roll_ms"],
                        "fade_in_ms": export_meta["fade_in_ms"],
                        "fade_out_ms": export_meta["fade_out_ms"],
                        "tail_sec": export_meta["tail_sec"],
                        "zero_cross_applied": export_meta["zero_cross_applied"],
                        "bpm_global": global_bpm,
                        "bpm_global_confidence": round(global_confidence, 2),
                        "bpm_clip": bpm_clip,
                        "bpm_clip_confidence": round(bpm_clip_confidence, 2),
                        "bpm_used": bpm_used,
                        "bpm_used_source": bpm_used_source,
                        "bars_estimated": raw_bars_estimate,
                        "bars_used": bars_used,
                        "bars_used_source": "refined_grid" if refined_ok else "estimated",
                        "refined": bool(refined_ok),
                        "refined_reason": refined_reason,
                        "tags": ", ".join(sorted(tags)),
                        "themes": ", ".join(themes),
                        "hook_score": cand.get("hook_score", 0.0),
                        "energy": cand.get("energy", 0.0),
                        "loopability": cand.get("loopability", 0.0),
                        "clip_path": str(clip_path),
                        "txt": str(txt_path),
                        "json": str(json_path),
                        "text": text[:240] if text else "",
                    })
                
                progress_bar.empty()
            
            # ----------------------------
            # Mode: Broadcast Hunter (Mix)
            # ----------------------------
            else:
                st.write("📻 Analyzing silence...")
                intervals = detect_non_silent_intervals(
                    in_path,
                    noise_db=st.session_state["noise_db"],
                    min_silence_s=st.session_state["min_silence_s"],
                    pad_s=st.session_state["pad_s"],
                    min_segment_s=st.session_state["min_segment_s"],
                )
                
                # Apply minimum segment filter
                min_duration = st.session_state["min_segment_s"]
                intervals = [(a, b) for a, b in intervals if (b - a) >= min_duration]
                st.write(f"✂️ Found {len(intervals)} segments (after {min_duration}s filter)")
                
                progress_bar = st.progress(0)
                total_int = len(intervals)
                
                for idx, (a, b) in enumerate(intervals, start=1):
                    progress_bar.progress(idx / total_int)
                    
                    a, b = float(a), float(b)
                    dur = max(0.0, b - a)
                    
                    # Transcribe
                    temp_wav = session_dir / f"temp_{idx:04d}__whisper.wav"
                    cut_segment_to_wav(in_path, temp_wav, a, b)
                    t = transcribe_wav(st.session_state.model, temp_wav, language=lang)
                    text = (t.get("text") or "").strip()
                    
                    # Generate slug (max 24 chars, use __noslug if empty)
                    slug = ""
                    if st.session_state["use_slug"] and text:
                        slug = safe_slug(" ".join(text.split()[:int(st.session_state["slug_words"])]), max_len=MAX_SLUG_LENGTH)
                    
                    # Generate UID
                    uid = clip_uid(in_name, a, b, idx)
                    
                    # New filename template for broadcast (no BPM/bars)
                    start_mmss = mmss(a)
                    end_mmss = mmss(b)
                    slug_part = slug if slug else "noslug"
                    
                    stem = f"{track_artist} - {track_title}__{idx:04d}__{start_mmss}-{end_mmss}__{slug_part}__{uid}"
                    
                    # Export without tail for broadcast (and without fades)
                    clip_path, export_meta = export_clip_with_tail(
                        in_path, session_dir, stem, a, b,
                        st.session_state["export_format"],
                        add_tail=False,
                        add_fades=False
                    )
                    
                    # Save metadata
                    txt_path = session_dir / f"{stem}.txt"
                    json_path = session_dir / f"{stem}.json"
                    txt_path.write_text(text + "\n", encoding="utf-8")
                    json_path.write_text(
                        json.dumps(t, ensure_ascii=False, indent=2),
                        encoding="utf-8"
                    )
                    
                    tags = auto_tags(text)
                    score = float(jingle_score(text, dur))
                    themes = detect_themes(text)
                    
                    results.append({
                        "source": in_name,
                        "filename": clip_path.name,
                        "filename_stem": stem,
                        "filename_template_version": "v2",
                        "track_artist": track_artist,
                        "track_title": track_title,
                        "track_id": track_id,
                        "session_dir": str(session_dir.name),
                        "pick": True,
                        "clip": idx,
                        "start_sec": a,
                        "end_sec": b,
                        "dur_sec": dur,
                        "core_start_sec": export_meta["core_start_sec"],
                        "core_end_sec": export_meta["core_end_sec"],
                        "core_dur_sec": export_meta["core_dur_sec"],
                        "export_start_sec": export_meta["export_start_sec"],
                        "export_end_sec": export_meta["export_end_sec"],
                        "export_dur_sec": export_meta["export_dur_sec"],
                        "pre_roll_ms": export_meta["pre_roll_ms"],
                        "fade_in_ms": export_meta["fade_in_ms"],
                        "fade_out_ms": export_meta["fade_out_ms"],
                        "tail_sec": export_meta["tail_sec"],
                        "zero_cross_applied": export_meta["zero_cross_applied"],
                        "bpm_global": 0,
                        "bpm_global_confidence": 0.0,
                        "bpm_clip": 0,
                        "bpm_clip_confidence": 0.0,
                        "bpm_used": 0,
                        "bpm_used_source": "unknown",
                        "bars_estimated": 0,
                        "bars_used": None,
                        "bars_used_source": "unknown",
                        "refined": False,
                        "refined_reason": "",
                        "tags": ", ".join(tags),
                        "themes": ", ".join(themes),
                        "jingle_score": score,
                        "clip_path": str(clip_path),
                        "txt": str(txt_path),
                        "json": str(json_path),
                        "text": text[:240] if text else "",
                    })
                
                progress_bar.empty()
        
        st.session_state.results = results
        
        if not results:
            status.update(label="⚠️ No clips found", state="error")
            st.error("No clips found with current settings. Try adjusting parameters.")
        else:
            status.update(label=f"✅ Success! Found {len(results)} clips", state="complete")
            st.balloons()
            st.success(f"🎉 Finished! Found {len(results)} clips. Scroll down to preview.")
            
            # Generate QA Report for Song Hunter mode
            if mode_now == "🎵 Song Hunter (Loops)":
                qa_report = {
                    "total_clips": len(results),
                    "refined_ok": sum(1 for r in results if r.get("refined", False)),
                    "refined_fail": sum(1 for r in results if not r.get("refined", False)),
                    "global_bpm": global_bpm if 'global_bpm' in locals() else 0,
                    "global_confidence": round(global_confidence, 2) if 'global_confidence' in locals() else 0.0,
                }
                
                # BPM statistics
                bpms = [r.get("bpm_used", 0) for r in results if r.get("bpm_used", 0) > 0]
                if bpms:
                    qa_report["bpm_min"] = min(bpms)
                    qa_report["bpm_max"] = max(bpms)
                    qa_report["bpm_median"] = int(pd.Series(bpms).median())
                
                # Top hooks by score
                sorted_results = sorted(results, key=lambda r: r.get("hook_score", 0), reverse=True)
                qa_report["top_hooks"] = [
                    {
                        "clip": r.get("clip"),
                        "score": round(r.get("hook_score", 0), 2),
                        "bpm": r.get("bpm_used"),
                        "bars": r.get("bars_used"),
                    }
                    for r in sorted_results[:10]
                ]
                
                # Refined fail reasons breakdown
                reasons = {}
                for r in results:
                    if not r.get("refined", False):
                        reason = r.get("refined_reason", "unknown")
                        reasons[reason] = reasons.get(reason, 0) + 1
                qa_report["refined_fail_reasons"] = reasons
                
                st.session_state.qa_report = qa_report

# ----------------------------
# Results Browser / Export
# ----------------------------
if "results" in st.session_state and st.session_state.results:
    st.divider()
    st.subheader("📊 Results Browser")
    
    # Display QA Report if available
    if "qa_report" in st.session_state:
        with st.expander("📊 QA Report", expanded=False):
            qa = st.session_state.qa_report
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Clips", qa.get("total_clips", 0))
            with col2:
                st.metric("Refined OK", qa.get("refined_ok", 0))
            with col3:
                st.metric("Refined Fail", qa.get("refined_fail", 0))
            with col4:
                st.metric("Global BPM", f"{qa.get('global_bpm', 0)} ({qa.get('global_confidence', 0):.2f})")
            
            if "bpm_min" in qa:
                st.write(f"**BPM Range:** {qa['bpm_min']} - {qa['bpm_max']} (median: {qa['bpm_median']})")
            
            if qa.get("refined_fail_reasons"):
                st.write("**Refined Fail Reasons:**")
                for reason, count in sorted(qa["refined_fail_reasons"].items(), key=lambda x: x[1], reverse=True):
                    st.write(f"  - {reason}: {count}")
            
            if qa.get("top_hooks"):
                st.write("**Top 10 Hooks by Score:**")
                top_df = pd.DataFrame(qa["top_hooks"])
                st.dataframe(top_df, use_container_width=True, hide_index=True)
    
    df = pd.DataFrame(st.session_state.results)
    if "pick" not in df.columns:
        df["pick"] = True
    
    edited = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "pick": st.column_config.CheckboxColumn("Select", default=True),
            "filename": st.column_config.TextColumn("Filename", width="large"),
            "bpm": st.column_config.NumberColumn("BPM", format="%d"),
            "tags": st.column_config.TextColumn("Tags", width="medium"),
            "themes": st.column_config.TextColumn("Themes", width="medium"),
            "dur_sec": st.column_config.NumberColumn("Duration", format="%.2f"),
            "hook_score": st.column_config.NumberColumn("Hook Score", format="%.2f"),
            "jingle_score": st.column_config.NumberColumn("Jingle Score", format="%.2f"),
            "text": st.column_config.TextColumn("Transcript", width="large"),
        }
    )
    
    selected = edited[edited["pick"] == True].copy()
    st.write(f"**Selected:** {len(selected)} clips")
    
    # Preview section
    with st.expander("🎧 Preview Selected (first 10)", expanded=True):
        for idx, r in selected.head(10).iterrows():
            p = Path(r["clip_path"])
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{r['filename']}**")
                st.caption(f"Duration: {float(r['dur_sec']):.2f}s | BPM: {r.get('bpm', 0)} | Tags: {r.get('tags', '')}")
                if r.get("text"):
                    st.write(f"📝 {r['text']}")
            with col2:
                if p.exists():
                    st.audio(p.read_bytes())
    
    # Export ZIP
    if st.button("📦 Export ZIP (Selected)", type="primary", use_container_width=True):
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
            # Add manifest
            z.writestr(
                "manifest_selected.csv",
                selected.to_csv(index=False).encode("utf-8")
            )
            
            # Add files
            for _, r in selected.iterrows():
                src_stem = Path(r["source"]).stem
                for k in ["clip_path", "txt", "json"]:
                    if k in r and r[k]:
                        fp = Path(r[k])
                        if fp.exists():
                            z.write(fp, arcname=f"{src_stem}/{fp.name}")
        
        zip_buf.seek(0)
        st.download_button(
            label="⬇️ Download ZIP",
            data=zip_buf,
            file_name="sample_machine_clips.zip",
            mime="application/zip",
            use_container_width=True,
        )
else:
    st.info("👆 Upload or download a file, load the Whisper model, and click Process to begin.")
