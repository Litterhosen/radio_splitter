# CRITICAL: st.set_page_config MUST be the VERY FIRST Streamlit call
import streamlit as st

# Import version first for page config
from config import VERSION

st.set_page_config(page_title=f"The Sample Machine v{VERSION}", layout="wide")

import io
import json
import zipfile
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from uuid import uuid4

import pandas as pd

# Local imports
from audio_split import (
    detect_non_silent_intervals,
    fixed_length_intervals,
    cut_segment_to_wav,
    cut_segment_to_mp3,
    cut_segment_with_fades,
    get_duration_seconds,
)
from transcribe import load_model, transcribe_wav
from downloaders import download_audio, DownloadError
from tagging import auto_tags
from jingle_finder import jingle_score
from hook_finder import ffmpeg_to_wav16k_mono, find_hooks
from beat_refine import refine_best_1_or_2_bars
from utils import ensure_dir, hhmmss_ms, mmss, safe_slug, safe_dirname, clip_uid, estimate_bars_from_duration, snap_bars_to_valid, extract_track_metadata
from audio_detection import (
    detect_audio_type,
    normalize_text_for_signature,
    extract_language_info,
    merge_language_votes,
    resolve_clip_language,
)

# Import configuration and UI utilities
from config import (
    OUTPUT_ROOT,
    OVERLAP_THRESHOLD,
    MIN_DURATION_SECONDS,
    MIN_CLIP_DURATION_SECONDS,
    DECAY_TAIL_DURATION,
    MAX_SLUG_LENGTH,
    MAX_FILENAME_LENGTH,
    MAX_STEM_LENGTH,
    MODE_OPTIONS,
    THEMES,
    DEFAULTS,
)
from ui_utils import (
    bars_ui_to_int,
    get_whisper_language,
    detect_themes,
    overlap_ratio,
    anti_overlap_keep_best,
    filter_by_duration,
    save_input_to_session_dir,
    export_clip_with_tail,
    build_groups,
)

# ----------------------------
# Initialize Output Directory
# ----------------------------
try:
    ensure_dir(OUTPUT_ROOT)
except (OSError, PermissionError):
    st.error("⚠️ Could not create output directory. Please check permissions and disk space.")
    st.stop()

# ----------------------------
# Initialize Session State
# ----------------------------
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ----------------------------
# Helper Function for Beat Refinement
# ----------------------------
def maybe_refine_barloop(wav_for_analysis: Path, a: float, b: float) -> Tuple[float, float, int, Optional[int], float, bool, str, int, float]:
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


def detect_file_language(model, in_path: Path, session_dir: Path, forced_language: Optional[str]) -> Dict[str, Any]:
    """Detect file-level language from longer samples or use forced policy."""
    if forced_language in {"da", "en"}:
        return {
            "language_policy": "forced",
            "language_forced": forced_language,
            "language_guess_file": forced_language,
            "language_confidence_file": 1.0,
        }

    sample_windows = [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0)]
    votes = []
    for idx, (a, b) in enumerate(sample_windows, start=1):
        sample_wav = session_dir / f"_lang_probe_{idx}.wav"
        try:
            cut_segment_to_wav(in_path, sample_wav, a, b)
            probe = transcribe_wav(model, sample_wav, language=None)
            votes.append(extract_language_info(probe))
        except Exception:
            continue

    merged = merge_language_votes(votes)
    return {
        "language_policy": "auto",
        "language_forced": "",
        "language_guess_file": merged.get("language_guess", "unknown"),
        "language_confidence_file": float(merged.get("language_confidence", 0.0) or 0.0),
    }


SONG_HUNTER_LONG_FILE_LIMIT_MINUTES = 20.0
SONG_HUNTER_LONG_FILE_LIMIT_SECONDS = SONG_HUNTER_LONG_FILE_LIMIT_MINUTES * 60.0


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
st.sidebar.checkbox("Debug mode (save crash trace)", key="debug_mode", value=True)

# Export settings
st.sidebar.subheader("📤 Export")
st.sidebar.selectbox("Format", ["wav (16000 mono)", "mp3 (192k)"], key="export_format")
st.sidebar.checkbox("Add slug from transcript", key="use_slug")
st.sidebar.slider("Slug words", 2, 12, key="slug_words")

# Mode-specific settings
if st.session_state["mode"] == "📻 Broadcast Hunter (Mix)":
    st.sidebar.subheader("📻 Broadcast Split")
    st.sidebar.selectbox(
        "Split method",
        ["VAD-first (recommended)", "Energy-first"],
        key="broadcast_split_method",
    )
    st.sidebar.caption("Minimum segment is fixed at 1.5s")
    st.sidebar.slider("Max segment (sec)", 10.0, 120.0, step=1.0, key="max_segment_s")
    st.sidebar.slider("Merge gap (sec)", 0.1, 1.0, step=0.05, key="merge_gap_s")
    st.sidebar.slider("Chunk size (sec)", 300.0, 900.0, step=60.0, key="broadcast_chunk_s")
    with st.sidebar.expander("Silence fallback tuning", expanded=False):
        st.slider("Silence threshold (dB)", -60.0, -10.0, step=1.0, key="noise_db")
        st.slider("Min silence (sec)", 0.2, 2.0, step=0.1, key="min_silence_s")
        st.slider("Padding (sec)", 0.0, 1.0, step=0.05, key="pad_s")
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
    st.sidebar.checkbox(f"Allow long Song Hunter files (>{int(SONG_HUNTER_LONG_FILE_LIMIT_MINUTES)} min)", key="allow_long_song_hunter", value=False)
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
            # Display structured error classification
            st.error(f"❌ Download failed: {e}")
            
            # Show error classification if available
            if hasattr(e, 'error_code') and e.error_code:
                st.error(f"🏷️ **Error Classification:** `{e.error_code.value}`")
            
            # Show log file location
            if e.log_file:
                st.info(f"📄 **Log file:** `{e.log_file}`")
            
            # Show hint
            if e.hint:
                st.warning(f"💡 **Diagnosis:** {e.hint}")

            if hasattr(e, 'error_code') and e.error_code and e.error_code.value == "ERR_JS_RUNTIME_MISSING":
                st.info("ℹ️ **Streamlit Cloud note:** some YouTube downloads require Node.js; Streamlit Cloud runtimes may not provide it.")
            
            # Show next steps in a structured way
            if hasattr(e, 'error_code') and e.error_code:
                from downloaders import classify_error, check_js_runtime
                _, _, next_steps = classify_error(str(e.last_error) if e.last_error else "", check_js_runtime() is not None)
                st.info("**Next Steps:**")
                st.markdown(next_steps)
            
            # Show full log in expander
            if e.log_file:
                try:
                    with open(e.log_file, 'r') as f:
                        log_content = f.read()
                    with st.expander("📋 View full log"):
                        st.code(log_content, language="text")
                except Exception as read_error:
                    st.warning(f"Could not read log file: {read_error}")
            
            # Show technical details in expander
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
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    run_scope_dir = ensure_dir(OUTPUT_ROOT / run_id)

    # Check if model is needed and loaded
    if st.session_state.model is None:
        st.warning("⚠️ Please click 'Load Whisper Model' first!")
        st.stop()

    try:
        with st.status("🚀 Starting processing...", expanded=True) as status:
            results = []
            st.session_state["source_artifacts"] = []
            st.session_state["active_run_id"] = run_id
            lang = get_whisper_language()
            total_files = len(input_paths)
    
            for idx_file, (src_type, name_or_path, maybe_bytes, youtube_info) in enumerate(input_paths):
                status.update(
                    label=f"📦 Processing file {idx_file+1} of {total_files}",
                    state="running"
                )
                
                st.write(f"### File {idx_file+1}/{total_files}: {name_or_path}")
                
                # Step 1: Save and convert to analysis format
                in_path, session_dir, in_name, yt_info = save_input_to_session_dir(src_type, name_or_path, maybe_bytes, youtube_info, run_id=run_id)
                
                # Extract track metadata (artist, title, track_id)
                track_artist, track_title, track_id = extract_track_metadata(in_path, yt_info)
                st.write(f"🎵 Track: **{track_artist} - {track_title}**")
    
                source_duration_sec = get_duration_seconds(in_path)
                if mode_now == "🎵 Song Hunter (Loops)" and source_duration_sec > SONG_HUNTER_LONG_FILE_LIMIT_SECONDS and not st.session_state.get("allow_long_song_hunter", False):
                    st.warning(
                        f"⚠️ Song Hunter is disabled for files longer than {int(SONG_HUNTER_LONG_FILE_LIMIT_MINUTES)} minutes "
                        f"({source_duration_sec/60.0:.1f} min detected). Use Broadcast mode, split input, or enable the long-file override in sidebar."
                    )
                    continue
    
                st.write("⏳ Converting audio...")
                wav16 = session_dir / "_analysis_16k_mono.wav"
                ffmpeg_to_wav16k_mono(in_path, wav16)
                st.write("✅ Conversion complete.")
                
                # Auto-detect audio type
                st.write("🔍 Detecting audio type...")
                audio_detection = detect_audio_type(wav16, sr=16000, duration=30.0)
                st.write(f"🎯 **Audio Type:** {audio_detection['audio_type_guess']} (confidence: {audio_detection['audio_type_confidence']:.2f})")
                st.write(f"💡 **Recommended Mode:** {audio_detection['recommended_mode']}")
    
                language_meta = detect_file_language(
                    st.session_state.model,
                    in_path,
                    session_dir,
                    forced_language=lang,
                )
                st.write(
                    f"🗣️ File language: {language_meta['language_guess_file']} "
                    f"(confidence: {language_meta['language_confidence_file']:.2f}, policy: {language_meta['language_policy']})"
                )
                
                # ----------------------------
                # Mode: Song Hunter (Loops)
                # ----------------------------
                if mode_now == "🎵 Song Hunter (Loops)":
                    st.write("🎵 Finding hooks...")
                    
                    # Get prefer_bars from session state
                    prefer_bars = int(st.session_state.get("prefer_bars", 2))
                    beats_per_bar = int(st.session_state.get("beats_per_bar", 4))
                for idx, cand in enumerate(candidates, start=1):
                    progress_bar.progress(idx / total_hooks)
                    
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
                        
                        # FIX: Don't override refinement based on min_duration
                        # The prefer_bars setting should control clip length, not a hard minimum
                        # Only reject clips that are genuinely too short (< MIN_CLIP_DURATION_SECONDS)
                        if refined_ok and dur < MIN_CLIP_DURATION_SECONDS:
                            # Clip is too short even after refinement - fall back to original
                            aa, bb = a, b
                            dur = max(0.0, bb - aa)
                            refined_ok = False
                            # Keep the original reason from refinement (likely "too_short" from beat_refine.py)
                        
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
                        
                        # Extract language info for this clip
                        clip_lang_info = extract_language_info(tjson)
                        clip_language = resolve_clip_language(
                            language_meta["language_guess_file"],
                            language_meta["language_confidence_file"],
                            clip_lang_info,
                            text,
                        )
                        
                        # Generate text signature for grouping
                        text_signature = normalize_text_for_signature(text, max_words=10)
                        
                        # Generate slug (max 24 chars, use __noslug if empty)
                        slug = ""
                        if st.session_state["use_slug"] and text:
                            slug = safe_slug(" ".join(text.split()[:int(st.session_state["slug_words"])]), max_len=MAX_SLUG_LENGTH)
                        
                        # Filename format: {artist}-{title}__{idx}__{bpm}bpm__{bars}bar__{slug}__{uid6}.mp3
                        # NO timestamps in main identifier per spec requirement
                        
                        # Build BPM/bars parts
                        if refined_ok:
                            bpm_part = f"{bpm_used}bpm"
                            bars_part = f"{bars_used}bar"
                        # Use the reason from refinement attempt (disabled, too_short, no_onsets, etc.)
                        refined_reason = rreason
                    
                    # Generate UID
                    uid = clip_uid(in_name, aa, bb, idx)
                    
                    # Transcribe
                    temp_wav = session_dir / f"temp_{idx:04d}__whisper.wav"
                    cut_segment_to_wav(in_path, temp_wav, aa, bb)
                    tjson = transcribe_wav(st.session_state.model, temp_wav, language=lang)
                    text = (tjson.get("text") or "").strip()
                    
                    # Extract language info for this clip
                    clip_lang_info = extract_language_info(tjson)
                    clip_language = resolve_clip_language(
                        language_meta["language_guess_file"],
                        language_meta["language_confidence_file"],
                        clip_lang_info,
                        text,
                    )
                    
                    # Generate text signature for grouping
                    text_signature = normalize_text_for_signature(text, max_words=10)
                    
                    # Generate slug (max 24 chars, use __noslug if empty)
                    slug = ""
                    if st.session_state["use_slug"] and text:
                        slug = safe_slug(" ".join(text.split()[:int(st.session_state["slug_words"])]), max_len=MAX_SLUG_LENGTH)
                    
                    # Filename format: {artist}-{title}__{idx}__{bpm}bpm__{bars}bar__{slug}__{uid6}.mp3
                    # NO timestamps in main identifier per spec requirement
                    
                    # Build BPM/bars parts
                    if refined_ok:
                        bpm_part = f"{bpm_used}bpm"
                        bars_part = f"{bars_used}bar"
                    else:
                        if final_bpm_conf < 0.4:
                            bpm_part = f"{bpm_used}bpm_low"
                        else:
                            if final_bpm_conf < 0.4:
                                bpm_part = f"{bpm_used}bpm_low"
                            else:
                                bpm_part = f"{bpm_used}bpm_est"
                            bars_part = f"{bars_used}bar_est"
                        
                        # slug_part
                        slug_part = slug if slug else "noslug"
                        
                        # Build complete filename stem (NO timestamps)
                        # Format: {artist}-{title}__{idx}__{bpm}bpm__{bars}bar__{slug}__{uid6}
                        # Max length enforcement (see MAX_FILENAME_LENGTH and MAX_STEM_LENGTH constants)
                        stem = f"{track_artist}-{track_title}__{idx:04d}__{bpm_part}__{bars_part}__{slug_part}__{uid}"
                        
                        # Enforce max length
                        if len(stem) > MAX_STEM_LENGTH:
                            # Truncate the slug part first, then title if needed
                            excess = len(stem) - MAX_STEM_LENGTH
                            if len(slug_part) > 10:
                                slug_part = slug_part[:max(4, len(slug_part) - excess)]
                                stem = f"{track_artist}-{track_title}__{idx:04d}__{bpm_part}__{bars_part}__{slug_part}__{uid}"
                            if len(stem) > MAX_STEM_LENGTH:
                                # Still too long, truncate title
                                title_len = len(track_title)
                                excess = len(stem) - MAX_STEM_LENGTH
                                track_title_trunc = track_title[:max(10, title_len - excess)]
                                stem = f"{track_artist}-{track_title_trunc}__{idx:04d}__{bpm_part}__{bars_part}__{slug_part}__{uid}"
                        
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
                            "bars_requested": prefer_bars,
                            "bars_policy": "prefer_bars",
                            "beats_per_bar": beats_per_bar,
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
                            "transcript_full_txt_path": "",
                            "transcript_full_json_path": "",
                            "language_detected": clip_language.get("language_guess", "unknown"),
                            "language_guess_file": language_meta["language_guess_file"],
                            "language_confidence_file": language_meta["language_confidence_file"],
                            "language_guess_clip": clip_language.get("language_guess", "unknown"),
                            "language_confidence_clip": clip_language.get("language_confidence", 0.0),
                            "language_source_clip": clip_language.get("language_source", "file"),
                            "language_policy": language_meta["language_policy"],
                            "language_forced": language_meta["language_forced"],
                            "audio_type_guess": audio_detection.get("audio_type_guess", "unknown"),
                            "audio_type_confidence": audio_detection.get("audio_type_confidence", 0.0),
                            "recommended_mode": audio_detection.get("recommended_mode", ""),
                            "clip_text_signature": text_signature,
                            "transcribe_model": st.session_state["model_size"],
                            "split_method_used": "hook",
                            "chunking_enabled": False,
                        })
                    
                    progress_bar.empty()
                
                # ----------------------------
                # Mode: Broadcast Hunter (Mix)
                # ----------------------------
                else:
                    from broadcast_splitter import detect_broadcast_segments
    
                    st.write("📻 Segmenting with VAD-first...")
                    intervals, split_method_used, chunking_enabled = detect_broadcast_segments(
                        wav16,
                        min_segment_sec=1.5,
                        max_segment_sec=float(st.session_state["max_segment_s"]),
                        merge_gap_sec=float(st.session_state["merge_gap_s"]),
                        chunk_sec=float(st.session_state["broadcast_chunk_s"]),
                        silence_noise_db=st.session_state["noise_db"],
                        silence_min_s=st.session_state["min_silence_s"],
                        silence_pad_s=st.session_state["pad_s"],
                        prefer_method="vad" if st.session_state["broadcast_split_method"].startswith("VAD") else "energy",
                    )
    
                    st.write(f"✂️ Found {len(intervals)} segments via **{split_method_used}** (chunking={chunking_enabled})")
    
                    progress_bar = st.progress(0)
                    total_int = len(intervals)
                    source_rows = []
                    full_transcript_segments = []
    
                    for idx, (a, b) in enumerate(intervals, start=1):
                        progress_bar.progress(idx / max(total_int, 1))
    
                        a, b = float(a), float(b)
                        dur = max(0.0, b - a)
    
                        temp_wav = session_dir / f"temp_{idx:04d}__whisper.wav"
                        cut_segment_to_wav(in_path, temp_wav, a, b)
                        t = transcribe_wav(st.session_state.model, temp_wav, language=lang)
                        text = (t.get("text") or "").strip()
                        # Extract language info for this clip
                        clip_lang_info = extract_language_info(t)
                        clip_language = resolve_clip_language(
                            language_meta["language_guess_file"],
                            language_meta["language_confidence_file"],
                            clip_lang_info,
                            text,
                        )
                        
                        # Generate text signature for grouping
                        text_signature = normalize_text_for_signature(text, max_words=10)
    
                        for seg in t.get("segments", []):
                            full_transcript_segments.append({
                                "start": float(a) + float(seg.get("start", 0.0)),
                                "end": float(a) + float(seg.get("end", 0.0)),
                                "text": (seg.get("text") or "").strip(),
                            })
    
                        slug = ""
                        if st.session_state["use_slug"] and text:
                            slug = safe_slug(" ".join(text.split()[:int(st.session_state["slug_words"])]), max_len=MAX_SLUG_LENGTH)
    
                        uid = clip_uid(in_name, a, b, idx)
                        start_mmss = mmss(a)
                        end_mmss = mmss(b)
                        slug_part = slug if slug else "noslug"
                        stem = f"{track_artist} - {track_title}__{idx:04d}__{start_mmss}-{end_mmss}__{slug_part}__{uid}"
    
                        clip_path, export_meta = export_clip_with_tail(
                            in_path, session_dir, stem, a, b,
                            st.session_state["export_format"],
                            add_tail=False,
                            add_fades=False
                        )
    
                        txt_path = session_dir / f"{stem}.txt"
                        json_path = session_dir / f"{stem}.json"
                        txt_path.write_text(text + "\n", encoding="utf-8")
                        json_path.write_text(json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8")
    
                        tags = auto_tags(text)
                        score = float(jingle_score(text, dur))
                        themes = detect_themes(text)
    
                        source_rows.append({
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
                            "transcript_full_txt_path": "",
                            "transcript_full_json_path": "",
                            "language_detected": clip_language.get("language_guess", "unknown"),
                            "language_guess_file": language_meta["language_guess_file"],
                            "language_confidence_file": language_meta["language_confidence_file"],
                            "language_guess_clip": clip_language.get("language_guess", "unknown"),
                            "language_confidence_clip": clip_language.get("language_confidence", 0.0),
                            "language_source_clip": clip_language.get("language_source", "file"),
                            "language_policy": language_meta["language_policy"],
                            "language_forced": language_meta["language_forced"],
                            "audio_type_guess": audio_detection.get("audio_type_guess", "unknown"),
                            "audio_type_confidence": audio_detection.get("audio_type_confidence", 0.0),
                            "recommended_mode": audio_detection.get("recommended_mode", ""),
                            "clip_text_signature": text_signature,
                            "transcribe_model": st.session_state["model_size"],
                            "split_method_used": "vad",
                            "chunking_enabled": False,
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
                        "bars_requested": prefer_bars,
                        "bars_policy": "prefer_bars",
                        "beats_per_bar": beats_per_bar,
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
                        "transcript_full_txt_path": "",
                        "transcript_full_json_path": "",
                        "language_detected": clip_language.get("language_guess", "unknown"),
                        "language_guess_file": language_meta["language_guess_file"],
                        "language_confidence_file": language_meta["language_confidence_file"],
                        "language_guess_clip": clip_language.get("language_guess", "unknown"),
                        "language_confidence_clip": clip_language.get("language_confidence", 0.0),
                        "language_source_clip": clip_language.get("language_source", "file"),
                        "language_policy": language_meta["language_policy"],
                        "language_forced": language_meta["language_forced"],
                        "audio_type_guess": audio_detection.get("audio_type_guess", "unknown"),
                        "audio_type_confidence": audio_detection.get("audio_type_confidence", 0.0),
                        "recommended_mode": audio_detection.get("recommended_mode", ""),
                        "clip_text_signature": text_signature,
                        "transcribe_model": st.session_state["model_size"],
                        "split_method_used": "hook",
                        "chunking_enabled": False,
                    })
                
                progress_bar.empty()
            
            # ----------------------------
            # Mode: Broadcast Hunter (Mix)
            # ----------------------------
            else:
                from broadcast_splitter import detect_broadcast_segments

                st.write("📻 Segmenting with VAD-first...")
                intervals, split_method_used, chunking_enabled = detect_broadcast_segments(
                    wav16,
                    min_segment_sec=1.5,
                    max_segment_sec=float(st.session_state["max_segment_s"]),
                    merge_gap_sec=float(st.session_state["merge_gap_s"]),
                    chunk_sec=float(st.session_state["broadcast_chunk_s"]),
                    silence_noise_db=st.session_state["noise_db"],
                    silence_min_s=st.session_state["min_silence_s"],
                    silence_pad_s=st.session_state["pad_s"],
                    prefer_method="vad" if st.session_state["broadcast_split_method"].startswith("VAD") else "energy",
                )

                st.write(f"✂️ Found {len(intervals)} segments via **{split_method_used}** (chunking={chunking_enabled})")

                progress_bar = st.progress(0)
                total_int = len(intervals)
                source_rows = []
                full_transcript_segments = []

                for idx, (a, b) in enumerate(intervals, start=1):
                    progress_bar.progress(idx / max(total_int, 1))

                    a, b = float(a), float(b)
                    dur = max(0.0, b - a)

                    temp_wav = session_dir / f"temp_{idx:04d}__whisper.wav"
                    cut_segment_to_wav(in_path, temp_wav, a, b)
                    t = transcribe_wav(st.session_state.model, temp_wav, language=lang)
                    text = (t.get("text") or "").strip()
                    # Extract language info for this clip
                    clip_lang_info = extract_language_info(t)
                    clip_language = resolve_clip_language(
                        language_meta["language_guess_file"],
                        language_meta["language_confidence_file"],
                        clip_lang_info,
                        text,
                    )
                    
                    # Generate text signature for grouping
                    text_signature = normalize_text_for_signature(text, max_words=10)

                    for seg in t.get("segments", []):
                        full_transcript_segments.append({
                            "start": float(a) + float(seg.get("start", 0.0)),
                            "end": float(a) + float(seg.get("end", 0.0)),
                            "text": (seg.get("text") or "").strip(),
                        })
    
                    progress_bar.empty()
    
                    full_transcript_segments = sorted(full_transcript_segments, key=lambda x: x["start"])
                    transcript_txt_path = session_dir / "transcript_full.txt"
                    transcript_json_path = session_dir / "transcript_full.json"
    
                    transcript_lines = [
                        f"[{hhmmss_ms(seg['start'])}] {seg['text']}" for seg in full_transcript_segments if seg.get("text")
                    ]
                    transcript_txt_path.write_text("\n".join(transcript_lines) + ("\n" if transcript_lines else ""), encoding="utf-8")
                    transcript_json_path.write_text(
                        json.dumps({
                            "source": in_name,
                            "language_detected": language_meta["language_guess_file"],
                            "language_confidence_file": language_meta["language_confidence_file"],
                            "language_policy": language_meta["language_policy"],
                            "language_forced": language_meta["language_forced"],
                            "transcribe_model": st.session_state["model_size"],
                            "segments": full_transcript_segments,
                        }, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
    
                    source_artifact = {
                        "source": in_name,
                        "audio_path": str(in_path),
                        "session_dir": str(session_dir),
                        "transcript_full_txt_path": str(transcript_txt_path),
                        "transcript_full_json_path": str(transcript_json_path),

                    txt_path = session_dir / f"{stem}.txt"
                    json_path = session_dir / f"{stem}.json"
                    txt_path.write_text(text + "\n", encoding="utf-8")
                    json_path.write_text(json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8")

                    tags = auto_tags(text)
                    score = float(jingle_score(text, dur))
                    themes = detect_themes(text)

                    source_rows.append({
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
                        "transcript_full_txt_path": "",
                        "transcript_full_json_path": "",
                        "language_detected": clip_language.get("language_guess", "unknown"),
                        "language_guess_file": language_meta["language_guess_file"],
                        "language_confidence_file": language_meta["language_confidence_file"],
                        "language_guess_clip": clip_language.get("language_guess", "unknown"),
                        "language_confidence_clip": clip_language.get("language_confidence", 0.0),
                        "language_source_clip": clip_language.get("language_source", "file"),
                        "language_policy": language_meta["language_policy"],
                        "language_forced": language_meta["language_forced"],
                        "audio_type_guess": audio_detection.get("audio_type_guess", "unknown"),
                        "audio_type_confidence": audio_detection.get("audio_type_confidence", 0.0),
                        "recommended_mode": audio_detection.get("recommended_mode", ""),
                        "clip_text_signature": text_signature,
                        "transcribe_model": st.session_state["model_size"],
                        "split_method_used": "vad",
                        "chunking_enabled": False,
                    })

                progress_bar.empty()

                full_transcript_segments = sorted(full_transcript_segments, key=lambda x: x["start"])
                transcript_txt_path = session_dir / "transcript_full.txt"
                transcript_json_path = session_dir / "transcript_full.json"

                transcript_lines = [
                    f"[{hhmmss_ms(seg['start'])}] {seg['text']}" for seg in full_transcript_segments if seg.get("text")
                ]
                transcript_txt_path.write_text("\n".join(transcript_lines) + ("\n" if transcript_lines else ""), encoding="utf-8")
                transcript_json_path.write_text(
                    json.dumps({
                        "source": in_name,
                        "language_detected": language_meta["language_guess_file"],
                        "language_confidence_file": language_meta["language_confidence_file"],
                        "language_policy": language_meta["language_policy"],
                        "language_forced": language_meta["language_forced"],
                        "transcribe_model": st.session_state["model_size"],
                        "split_method_used": split_method_used,
                        "chunking_enabled": bool(chunking_enabled),
                        "segments": full_transcript_segments,
                    }
                    st.session_state.setdefault("source_artifacts", []).append(source_artifact)
    
                    for row in source_rows:
                        row["transcript_full_txt_path"] = str(transcript_txt_path)
                        row["transcript_full_json_path"] = str(transcript_json_path)
                        row["language_policy"] = language_meta["language_policy"]
                        row["language_forced"] = language_meta["language_forced"]
                        row["transcribe_model"] = st.session_state["model_size"]
                        row["split_method_used"] = split_method_used
                        row["chunking_enabled"] = bool(chunking_enabled)
    
                    results.extend(source_rows)
                    }, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                source_artifact = {
                    "source": in_name,
                    "audio_path": str(in_path),
                    "session_dir": str(session_dir),
                    "transcript_full_txt_path": str(transcript_txt_path),
                    "transcript_full_json_path": str(transcript_json_path),
                    "language_detected": language_meta["language_guess_file"],
                    "language_confidence_file": language_meta["language_confidence_file"],
                    "language_policy": language_meta["language_policy"],
                    "language_forced": language_meta["language_forced"],
                    "transcribe_model": st.session_state["model_size"],
                    "split_method_used": split_method_used,
                    "chunking_enabled": bool(chunking_enabled),
                    "segments": full_transcript_segments,
                }
                st.session_state.setdefault("source_artifacts", []).append(source_artifact)

                for row in source_rows:
                    row["transcript_full_txt_path"] = str(transcript_txt_path)
                    row["transcript_full_json_path"] = str(transcript_json_path)
                    row["language_policy"] = language_meta["language_policy"]
                    row["language_forced"] = language_meta["language_forced"]
                    row["transcribe_model"] = st.session_state["model_size"]
                    row["split_method_used"] = split_method_used
                    row["chunking_enabled"] = bool(chunking_enabled)

                results.extend(source_rows)
        
        st.session_state.results = results
        
        if not results:
            status.update(label="⚠️ No clips found", state="error")
            st.error("No clips found with current settings. Try adjusting parameters.")
        else:
            status.update(label=f"✅ Success! Found {len(results)} clips", state="complete")
            st.balloons()
            st.success(f"🎉 Finished! Found {len(results)} clips. Scroll down to preview.")
            
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
    except Exception as e:
        crash_log_path = run_scope_dir / "crash.log"
        crash_payload = (
            f"run_id={run_id}\n"
            f"mode={mode_now}\n"
            f"error={type(e).__name__}: {e}\n\n"
            f"traceback:\n{traceback.format_exc()}"
        )
        crash_log_path.write_text(crash_payload, encoding="utf-8")
        st.error("❌ Run failed, see crash.log")
        st.download_button(
            "⬇️ Download crash.log",
            data=crash_payload.encode("utf-8"),
            file_name=f"{run_id}_crash.log",
            mime="text/plain",
            use_container_width=True,
        )
        if st.session_state.get("debug_mode", False):
            st.exception(e)


# ----------------------------
# Helper function for displaying clip cards
# ----------------------------
def _display_clip_card(r):
    """Display a clip preview card with enhanced metadata"""
    p = Path(r["clip_path"])
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**{r['filename']}**")
        
        # Build caption with available fields
        caption_parts = [f"⏱️ {float(r['dur_sec']):.2f}s"]
        if r.get('bpm_used', 0) > 0:
            caption_parts.append(f"🎵 {r.get('bpm_used', 0)} BPM")
        if r.get('bars_used'):
            caption_parts.append(f"📊 {r.get('bars_used')} bars")
        if r.get('hook_score', 0) > 0:
            caption_parts.append(f"⭐ {r.get('hook_score', 0):.2f}")
        
        # Add language label
        lang = r.get("language_guess_clip", r.get("language_detected", ""))
        if lang:
            lang_emoji = {"da": "🇩🇰", "en": "🇬🇧"}.get(lang, "🌐")
            caption_parts.append(f"{lang_emoji} {lang.upper()}")
        
        st.caption(" | ".join(caption_parts))
        
        # Display transcript snippet with language and themes
        if r.get("text"):
            text = r["text"]
            # Show first 150 chars as snippet
            snippet = text[:150] + "..." if len(text) > 150 else text
            st.write(f"📝 {snippet}")
            
            # Show full text button (text is already visible in manifest)
            if len(text) > 150 and st.button(f"📋 Show full text", key=f"show_{r.get('clip', 0)}_{r.get('filename', '')}"):
                st.code(text, language=None)
        
        # Show tags and themes
        if r.get('tags') or r.get('themes'):
            tag_theme_parts = []
            if r.get('tags'):
                tag_theme_parts.append(f"🏷️ {r.get('tags', '')}")
            if r.get('themes'):
                tag_theme_parts.append(f"🎨 {r.get('themes', '')}")
            st.caption(" | ".join(tag_theme_parts))
    
    with col2:
        if p.exists():
            st.audio(p.read_bytes())

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

    # Grouping controls
    col_group1, col_group2 = st.columns([1, 1])
    with col_group1:
        grouping_options = ["None", "Group by Phrase", "Group by Tag/Theme", "Group by Language"]
        group_by = st.selectbox("Grouping", grouping_options, index=0, key="group_by")
    with col_group2:
        # Preview controls: sort + show all/paginated
        sort_options = []
        if "hook_score" in selected.columns:
            sort_options.append("hook_score")
        if "energy" in selected.columns:
            sort_options.append("energy")
        sort_options = sort_options or ["clip"]
        preview_sort = st.selectbox("Sort by", sort_options, index=0, key="preview_sort")

    if preview_sort in selected.columns:
        selected = selected.sort_values(preview_sort, ascending=False, kind="stable")

    show_all_preview = st.checkbox("Show all selected clips", value=False, key="preview_show_all")

    selected_rows = [r.to_dict() for _, r in selected.iterrows()]
    total_clips = len(selected_rows)

    group_mode_map = {
        "None": "none",
        "Group by Phrase": "phrase",
        "Group by Tag/Theme": "tag_theme",
        "Group by Language": "language",
    }
    group_mode = group_mode_map.get(group_by, "none")
    groups = None if group_mode == "none" else build_groups(selected_rows, group_mode)
    group_count = len(groups) if groups else 0

    clips_per_page = 20
    total_pages = (total_clips + clips_per_page - 1) // clips_per_page

    run_preview_key = f"preview_page__{st.session_state.get('active_run_id', 'default')}"
    if run_preview_key not in st.session_state:
        st.session_state[run_preview_key] = 0

    title = f"🎧 Preview Selected ({total_clips} clips)"
    if groups:
        title += f" — {group_count} groups"

    with st.expander(title, expanded=True):
        if total_clips > 0:
            if groups:
                for group_key, items in sorted(groups.items(), key=lambda x: -len(x[1])):
                    if group_mode == "phrase":
                        group_title = f"📋 \"{group_key[:60]}...\" ({len(items)} clips)"
                    elif group_mode == "tag_theme":
                        group_title = f"🏷️ {group_key} ({len(items)} clips)"
                    else:
                        lang_label = {"da": "🇩🇰 Danish", "en": "🇬🇧 English", "unknown": "❓ Unknown"}.get(group_key, f"🌐 {group_key}")
                        group_title = f"{lang_label} ({len(items)} clips)"

                    with st.expander(group_title, expanded=len(items) <= 3):
                        for r in items:
                            _display_clip_card(r)
            elif show_all_preview:
                st.write(f"Showing all {total_clips} clips")
                for r in selected_rows:
                    _display_clip_card(r)
            else:
                # Pagination controls
                if total_pages > 1:
                    col_prev, col_info, col_next = st.columns([1, 2, 1])
                    with col_prev:
                        if st.button("⬅️ Previous", disabled=st.session_state[run_preview_key] == 0):
                            st.session_state[run_preview_key] = max(0, st.session_state[run_preview_key] - 1)
                            st.rerun()
                    with col_info:
                        st.write(f"Page {st.session_state[run_preview_key] + 1} of {total_pages}")
                    with col_next:
                        if st.button("Next ➡️", disabled=st.session_state[run_preview_key] >= total_pages - 1):
                            st.session_state[run_preview_key] = min(total_pages - 1, st.session_state[run_preview_key] + 1)
                            st.rerun()

                # Calculate page slice
                start_idx = st.session_state[run_preview_key] * clips_per_page
                end_idx = min(start_idx + clips_per_page, total_clips)
                page_clips = selected_rows[start_idx:end_idx]

                # Display clips
                for r in page_clips:
                    _display_clip_card(r)

    # Full transcript browser + download (Broadcast mode)
    if st.session_state.get("source_artifacts"):
        st.divider()
        st.subheader("🧾 Full Transcripts")

        run_transcript_bundle = io.BytesIO()
        with zipfile.ZipFile(run_transcript_bundle, "w", zipfile.ZIP_DEFLATED) as z_run:
            for artifact in st.session_state.get("source_artifacts", []):
                src_stem = Path(artifact["source"]).stem
                for key in ["transcript_full_txt_path", "transcript_full_json_path"]:
                    fp = Path(artifact.get(key, ""))
                    if fp.exists():
                        z_run.write(fp, arcname=f"{src_stem}/{fp.name}")
        run_transcript_bundle.seek(0)
        st.download_button(
            "⬇️ Download full transcript bundle (run)",
            data=run_transcript_bundle,
            file_name="transcripts_full_run.zip",
            mime="application/zip",
            use_container_width=True,
        )

        for artifact in st.session_state.get("source_artifacts", []):
            src = artifact["source"]
            src_stem = Path(src).stem
            with st.expander(f"📄 {src}", expanded=False):
                c1, c2 = st.columns(2)
                txt_path = Path(artifact["transcript_full_txt_path"])
                json_path = Path(artifact["transcript_full_json_path"])
                if txt_path.exists():
                    with c1:
                        st.download_button(
                            f"⬇️ Download full transcript (.txt) — {src_stem}",
                            data=txt_path.read_bytes(),
                            file_name=f"{src_stem}_transcript_full.txt",
                            mime="text/plain",
                            key=f"dl_txt_{src_stem}",
                        )
                if json_path.exists():
                    with c2:
                        st.download_button(
                            f"⬇️ Download full transcript (.json) — {src_stem}",
                            data=json_path.read_bytes(),
                            file_name=f"{src_stem}_transcript_full.json",
                            mime="application/json",
                            key=f"dl_json_{src_stem}",
                        )

                source_selected = selected[selected["source"] == src] if "source" in selected.columns else pd.DataFrame([])
                src_zip = io.BytesIO()
                with zipfile.ZipFile(src_zip, "w", zipfile.ZIP_DEFLATED) as z_src:
                    if txt_path.exists():
                        z_src.write(txt_path, arcname=f"{src_stem}/{txt_path.name}")
                    if json_path.exists():
                        z_src.write(json_path, arcname=f"{src_stem}/{json_path.name}")
                    for _, row in source_selected.iterrows():
                        for k in ["clip_path", "txt", "json", "transcript_full_txt_path", "transcript_full_json_path"]:
                            fp = Path(row.get(k, ""))
                            if fp.exists():
                                z_src.write(fp, arcname=f"{src_stem}/{fp.name}")
                src_zip.seek(0)
                st.download_button(
                    f"📦 Download per-source ZIP — {src_stem}",
                    data=src_zip,
                    file_name=f"{src_stem}_bundle.zip",
                    mime="application/zip",
                    key=f"dl_zip_{src_stem}",
                )

                q = st.text_input("Search transcript", key=f"search_{src_stem}")
                segs = artifact.get("segments", [])
                if q.strip():
                    segs = [seg for seg in segs if q.lower() in (seg.get("text", "").lower())]

                for seg_idx, seg in enumerate(segs[:200], start=1):
                    ts = float(seg.get("start", 0.0))
                    ttxt = seg.get("text", "")
                    cols = st.columns([4, 1])
                    cols[0].write(f"[{hhmmss_ms(ts)}] {ttxt}")
                    if cols[1].button("▶️ Play from", key=f"play_{src_stem}_{seg_idx}"):
                        st.session_state[f"play_from_{src_stem}"] = ts

                if Path(artifact["audio_path"]).exists():
                    play_from = int(st.session_state.get(f"play_from_{src_stem}", 0))
                    st.audio(str(artifact["audio_path"]), start_time=play_from)

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
                for k in ["clip_path", "txt", "json", "transcript_full_txt_path", "transcript_full_json_path"]:
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
