# CRITICAL: st.set_page_config MUST be the VERY FIRST Streamlit call
import os
from pathlib import Path

# Force-disable Streamlit source watcher to avoid torch.classes path probing noise.
os.environ.setdefault("STREAMLIT_SERVER_FILEWATCHERTYPE", "none")
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

# Keep temp/cache writes on local disk (faster/more stable than cloud-synced folders).
_runtime_root = Path(
    os.getenv(
        "RADIO_SPLITTER_RUNTIME_ROOT",
        str(Path.home() / "AppData" / "Local" / "radio_splitter2"),
    )
)
_runtime_temp = _runtime_root / "temp"
_runtime_temp.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMP", str(_runtime_temp))
os.environ.setdefault("TEMP", str(_runtime_temp))
os.environ.setdefault("TMPDIR", str(_runtime_temp))

import streamlit as st

# Import version first for page config
from config import VERSION

st.set_page_config(page_title=f"The Sample Machine v{VERSION}", layout="wide")

import json
import zipfile
import traceback
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from time import perf_counter
from uuid import uuid4

import pandas as pd

# Local imports
from Gradio.audio_split import (
    cut_segment_to_wav,
    get_duration_seconds,
)
from transcribe import load_model, transcribe_wav
from downloaders import download_audio, DownloadError, check_cookie_health
from tagging import auto_tags
from jingle_finder import jingle_score
from hook_finder import (
    ffmpeg_to_wav16k_mono,
    find_hooks,
    normalize_bpm_family,
    normalize_bpm_to_prior,
)
from beat_refine import refine_best_1_or_2_bars
from rs_utils import ensure_dir, hhmmss_ms, mmss, safe_slug, clip_uid, estimate_bars_from_duration, snap_bars_to_valid, extract_track_metadata
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
    MIN_DURATION_SECONDS,
    MIN_CLIP_DURATION_SECONDS,
    MAX_SLUG_LENGTH,
    MAX_STEM_LENGTH,
    MODE_OPTIONS,
    DEFAULTS,
)
from ui_utils import (
    bars_ui_to_int,
    get_whisper_language,
    detect_themes,
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




def stage_uploaded_files(files: Optional[List[Any]]) -> List[str]:
    """Persist uploaded files to disk and return stable staged paths."""
    if not files:
        return []

    if "upload_session_id" not in st.session_state:
        st.session_state["upload_session_id"] = uuid4().hex[:12]

    staging_dir = ensure_dir(OUTPUT_ROOT / "_staging" / st.session_state["upload_session_id"])
    staged_paths: List[str] = []
    for uf in files:
        staged = staging_dir / uf.name
        size = getattr(uf, "size", None)
        if (not staged.exists()) or (size is not None and staged.stat().st_size != size):
            staged.write_bytes(uf.getbuffer())
        staged_paths.append(str(staged))
    return staged_paths

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
BROADCAST_MIN_SEGMENT_SEC = max(float(MIN_DURATION_SECONDS), float(MIN_CLIP_DURATION_SECONDS), 4.0)


def _broadcast_diagnostics_template() -> Dict[str, Any]:
    return {
        "segments_total": 0,
        "segments_too_short": 0,
        "segments_too_quiet": 0,
        "segments_transcribe_failed": 0,
        "segments_transcript_empty": 0,
        "segments_export_failed": 0,
        "segments_exported": 0,
    }


def _quick_scan_windows(duration_sec: float, window_sec: float) -> List[Tuple[float, float, str]]:
    if duration_sec <= 0:
        return []
    ws = min(window_sec, duration_sec)
    starts = [0.0, max(0.0, duration_sec / 2.0 - ws / 2.0), max(0.0, duration_sec - ws)]
    labels = ["start", "middle", "end"]
    windows = []
    for start, label in zip(starts, labels):
        end = min(duration_sec, start + ws)
        windows.append((start, end, label))
    return windows


def _sample_intervals(intervals: List[Tuple[float, float]], max_samples: int) -> List[Tuple[float, float]]:
    """Pick evenly spaced interval samples to keep quick scan fast on dense files."""
    if max_samples <= 0 or not intervals:
        return []
    if len(intervals) <= max_samples:
        return intervals
    if max_samples == 1:
        return [intervals[len(intervals) // 2]]

    step = (len(intervals) - 1) / float(max_samples - 1)
    idxs = sorted({int(round(i * step)) for i in range(max_samples)})
    return [intervals[i] for i in idxs]


def _format_clock(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _update_progress_bar(bar, started_at: float, done: float, total: float, label: str) -> None:
    total_safe = max(float(total), 1.0)
    frac = min(max(float(done) / total_safe, 0.0), 1.0)
    elapsed = max(0.0, perf_counter() - started_at)
    eta = (elapsed / frac - elapsed) if frac > 1e-6 else 0.0
    text = f"{int(frac * 100):3d}% | {label} | elapsed {_format_clock(elapsed)} | ETA {_format_clock(eta)}"
    bar.progress(frac, text=text)


def _broadcast_runtime_settings() -> Tuple[float, float, float]:
    profile = str(st.session_state.get("broadcast_profile", "Balanced"))
    max_segment_sec = float(st.session_state.get("max_segment_s", 45.0))
    merge_gap_sec = float(st.session_state.get("merge_gap_s", 0.35))
    chunk_sec = float(st.session_state.get("broadcast_chunk_s", 600.0))
    if profile.startswith("Overview"):
        # Overview profile favors fewer/larger segments for long archival recordings.
        max_segment_sec = max(max_segment_sec, 300.0)
        merge_gap_sec = max(merge_gap_sec, 1.10)
        chunk_sec = max(chunk_sec, 900.0)
    return max_segment_sec, merge_gap_sec, chunk_sec


def _silence_runtime_settings() -> Dict[str, Any]:
    mode_ui = str(st.session_state.get("silence_threshold_mode", "Auto (noise floor + margin)"))
    mode = "auto" if mode_ui.startswith("Auto") else "manual"
    return {
        "threshold_mode": mode,
        "noise_db": float(st.session_state.get("noise_db", -28.0)),
        "margin_db": float(st.session_state.get("silence_margin_db", 10.0)),
        "quick_test_mode": bool(st.session_state.get("silence_quick_test_enabled", True)),
        "quick_test_seconds": float(st.session_state.get("silence_quick_test_seconds", 120.0)),
        "quick_test_retries": int(st.session_state.get("silence_quick_test_retries", 3)),
    }


def _resolve_download_dir() -> Path:
    run_id = str(st.session_state.get("active_run_id", "") or "").strip()
    if run_id:
        return ensure_dir(OUTPUT_ROOT / run_id / "_downloads")

    manifest_path = st.session_state.get("manifest_path", "")
    if manifest_path:
        try:
            manifest_parent = Path(str(manifest_path)).resolve().parent
            if manifest_parent.exists():
                return ensure_dir(manifest_parent / "_downloads")
        except Exception:
            pass

    return ensure_dir(OUTPUT_ROOT / "_downloads")


def _download_button_from_path(
    label: str,
    path: Path,
    mime: str,
    *,
    key: Optional[str] = None,
    use_container_width: bool = True,
    file_name: Optional[str] = None,
) -> None:
    p = Path(path)
    if not p.exists():
        st.warning(f"Download file not found: {p.name}")
        return
    with p.open("rb") as fh:
        st.download_button(
            label,
            data=fh,
            file_name=file_name or p.name,
            mime=mime,
            key=key,
            use_container_width=use_container_width,
        )


def _write_zip_to_path(
    zip_path: Path,
    file_entries: List[Tuple[Path, str]],
    *,
    memory_entries: Optional[List[Tuple[str, bytes]]] = None,
) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for file_path, arcname in file_entries:
            p = Path(file_path)
            if p.exists():
                z.write(p, arcname=arcname)
        for arcname, payload in (memory_entries or []):
            z.writestr(arcname, payload)
    return zip_path


@st.cache_data(show_spinner=False)
def cached_duration_seconds(path_str: str) -> float:
    """Cache duration lookup to avoid repeated ffprobe calls for same file."""
    return float(get_duration_seconds(Path(path_str)))


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
        "Broadcast profile",
        ["Balanced", "Overview (long segments)"],
        key="broadcast_profile",
    )
    st.sidebar.selectbox(
        "Split method",
        ["VAD-first (recommended)", "Energy-first"],
        key="broadcast_split_method",
    )
    st.sidebar.caption(f"Minimum segment is fixed at {BROADCAST_MIN_SEGMENT_SEC:.1f}s")
    st.sidebar.slider("Max segment (sec)", 10.0, 1800.0, step=5.0, key="max_segment_s")
    st.sidebar.slider("Merge gap (sec)", 0.1, 3.0, step=0.05, key="merge_gap_s")
    st.sidebar.slider("Chunk size (sec)", 300.0, 3600.0, step=60.0, key="broadcast_chunk_s")
    if st.session_state.get("broadcast_profile", "Balanced").startswith("Overview"):
        eff_max, eff_gap, eff_chunk = _broadcast_runtime_settings()
        st.sidebar.caption(
            f"Overview active: runtime max segment {eff_max:.0f}s, merge gap {eff_gap:.2f}s, chunk {eff_chunk:.0f}s."
        )
    st.sidebar.slider("Quick Scan probes/window", 2, 20, step=1, key="quick_scan_probe_segments")
    st.sidebar.checkbox("Export without transcript", key="export_without_transcript")
    with st.sidebar.expander("Silence fallback tuning", expanded=False):
        st.selectbox(
            "Threshold mode",
            ["Auto (noise floor + margin)", "Manual (fixed dB)"],
            key="silence_threshold_mode",
        )
        if str(st.session_state.get("silence_threshold_mode", "")).startswith("Auto"):
            st.slider("Auto margin (dB)", 4.0, 16.0, step=0.5, key="silence_margin_db")
            st.caption("Threshold = noise_floor + margin, then clamped to [-55, -18] dB.")
        else:
            st.slider("Silence threshold (dB)", -60.0, -10.0, step=1.0, key="noise_db")
        st.slider("Min silence (sec)", 0.2, 2.0, step=0.1, key="min_silence_s")
        st.slider("Padding (sec)", 0.0, 1.0, step=0.05, key="pad_s")
        st.checkbox("Quick test mode (first part only)", key="silence_quick_test_enabled")
        if st.session_state.get("silence_quick_test_enabled", True):
            st.slider("Quick test window (sec)", 30.0, 300.0, step=10.0, key="silence_quick_test_seconds")
            st.slider("Quick test retries", 1, 3, step=1, key="silence_quick_test_retries")
else:
    st.sidebar.subheader("🎵 Hook Detection")
    st.sidebar.slider("Min hook length (sec)", float(MIN_DURATION_SECONDS), 30.0, step=0.5, key="hook_len_range_min")
    st.sidebar.slider("Max hook length (sec)", float(MIN_DURATION_SECONDS), 30.0, step=0.5, key="hook_len_range_max")
    
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
    dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 3])
    with dl_col1:
        dl_btn = st.button("Download", type="primary", disabled=not url.strip())
    with dl_col2:
        cookie_check_btn = st.button("Tjek cookies", type="secondary")
    with dl_col3:
        st.caption("Downloads to: output/Downloads/")

    if cookie_check_btn:
        try:
            with st.spinner("Tjekker cookie health..."):
                health = check_cookie_health(OUTPUT_ROOT / "Downloads")

            if health.get("ok"):
                st.success("✅ Cookie health check: OK")
            else:
                st.warning(f"⚠️ Cookie health check: {health.get('summary', 'failed')}")

            auth_source = health.get("auth_source", "none")
            st.caption(f"Auth source: {auth_source}")

            geo_country = str(health.get("geo_bypass_country", "") or "").strip()
            if geo_country:
                st.caption(f"Geo bypass country: {geo_country}")

            cookie_file_summary = health.get("cookie_file_summary")
            if isinstance(cookie_file_summary, dict):
                st.info(
                    "Cookie rows: "
                    f"{cookie_file_summary.get('parsed_cookie_rows', 0)} parsed, "
                    f"{cookie_file_summary.get('youtube_google_rows', 0)} YouTube/Google, "
                    f"{cookie_file_summary.get('youtube_google_active_rows', 0)} active."
                )

            if health.get("hint"):
                st.warning(f"💡 {health['hint']}")

            if health.get("next_steps"):
                with st.expander("Next steps from checker"):
                    st.markdown(health["next_steps"])

            if health.get("probe_error"):
                with st.expander("🔍 Cookie check technical details"):
                    st.code(str(health["probe_error"]), language="text")

            log_file = str(health.get("log_file", "") or "").strip()
            if log_file:
                try:
                    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                        health_log = f.read()
                    with st.expander("📋 Cookie health log"):
                        st.code(health_log, language="text")
                except Exception as log_error:
                    st.warning(f"Could not read cookie health log: {log_error}")

        except Exception as e:
            st.error(f"❌ Cookie health check crashed: {e}")
    
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
            if getattr(e, "next_steps", None):
                st.info("**Next Steps:**")
                st.markdown(e.next_steps)
            
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
if files is not None:
    st.session_state["uploaded_files"] = stage_uploaded_files(files)

input_paths = []
for staged_path in st.session_state.get("uploaded_files", []):
    input_paths.append(("path", staged_path, None, None))
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
if "quick_scan_rows" not in st.session_state:
    st.session_state.quick_scan_rows = []

col1, col2 = st.columns([1, 1])
with col1:
    load_btn = st.button("🔧 Load Whisper Model", type="secondary", use_container_width=True)
with col2:
    run_btn = st.button("▶️ Process", type="primary", disabled=not input_paths, use_container_width=True)

quick_scan_btn = False
if st.session_state["mode"] == "📻 Broadcast Hunter (Mix)":
    quick_scan_btn = st.button(
        "⚡ Quick Scan (recommended for long files)",
        type="secondary",
        disabled=not input_paths,
        use_container_width=True,
    )

if load_btn:
    model_key = (
        st.session_state["model_size"],
        st.session_state["device"],
        st.session_state["compute_type"],
    )
    if st.session_state.model is not None and st.session_state.get("model_key") == model_key:
        st.info("ℹ️ Model already loaded with current settings.")
    else:
        try:
            with st.spinner("Loading model... (first time downloads it)"):
                st.session_state.model = load_model(
                    model_size=model_key[0],
                    device=model_key[1],
                    compute_type=model_key[2],
                )
                st.session_state["model_key"] = model_key
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Could not load model: {e}")

if quick_scan_btn:
    if st.session_state.model is None:
        st.warning("⚠️ Please click 'Load Whisper Model' first!")
    else:
        from broadcast_splitter import detect_broadcast_segments, get_last_broadcast_segmentation_debug

        st.session_state["quick_scan_ready_for_full_run"] = True
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        lang = get_whisper_language()
        rows = []
        quick_scan_started = perf_counter()
        quick_scan_progress = st.progress(0.0, text="  0% | Quick scan starting...")
        qs_done = 0.0
        qs_total = max(1.0, float(len(input_paths) * 4))
        max_probe_segments = int(st.session_state.get("quick_scan_probe_segments", 8))
        scan_max_segment_sec, scan_merge_gap_sec, _ = _broadcast_runtime_settings()
        silence_cfg = _silence_runtime_settings()

        for file_idx, (src_type, name_or_path, maybe_bytes, youtube_info) in enumerate(input_paths, start=1):
            qs_done += 1.0
            _update_progress_bar(
                quick_scan_progress,
                quick_scan_started,
                qs_done,
                qs_total,
                f"Quick Scan file {file_idx}/{len(input_paths)}: preparing source",
            )
            in_path, session_dir, in_name, yt_info = save_input_to_session_dir(src_type, name_or_path, maybe_bytes, youtube_info, run_id=run_id)
            wav16 = session_dir / "_analysis_16k_mono.wav"
            ffmpeg_to_wav16k_mono(in_path, wav16)
            track_artist, track_title, _ = extract_track_metadata(in_path, yt_info)
            duration_sec = cached_duration_seconds(str(in_path))
            windows = _quick_scan_windows(duration_sec, float(st.session_state.get("quick_scan_window_sec", 75.0)))

            for window_idx, (w_start, w_end, w_label) in enumerate(windows, start=1):
                qs_done += 1.0
                _update_progress_bar(
                    quick_scan_progress,
                    quick_scan_started,
                    qs_done,
                    qs_total,
                    f"Quick Scan file {file_idx}/{len(input_paths)} window {window_idx}/{max(len(windows), 1)}",
                )
                window_wav = session_dir / f"_quick_scan_{w_label}.wav"
                cut_segment_to_wav(in_path, window_wav, w_start, w_end)
                intervals, method_used, _ = detect_broadcast_segments(
                    window_wav,
                    min_segment_sec=BROADCAST_MIN_SEGMENT_SEC,
                    max_segment_sec=scan_max_segment_sec,
                    merge_gap_sec=scan_merge_gap_sec,
                    chunk_sec=99999.0,
                    silence_noise_db=silence_cfg["noise_db"],
                    silence_min_s=st.session_state["min_silence_s"],
                    silence_pad_s=st.session_state["pad_s"],
                    silence_threshold_mode=silence_cfg["threshold_mode"],
                    silence_margin_db=silence_cfg["margin_db"],
                    silence_quick_test_mode=silence_cfg["quick_test_mode"],
                    silence_quick_test_seconds=silence_cfg["quick_test_seconds"],
                    silence_quick_test_retries=silence_cfg["quick_test_retries"],
                    prefer_method="vad" if st.session_state["broadcast_split_method"].startswith("VAD") else "energy",
                )
                split_debug = get_last_broadcast_segmentation_debug()
                silence_debug = split_debug.get("silence_debug", {}) if isinstance(split_debug, dict) else {}
                nonempty = 0
                guesses = []
                sampled_intervals = _sample_intervals(intervals, max_probe_segments)
                qs_total += float(len(sampled_intervals))

                for probe_idx, (a, b) in enumerate(sampled_intervals, start=1):
                    probe_wav = session_dir / f"_quick_scan_probe_{w_label}_{int(a*1000)}.wav"
                    cut_segment_to_wav(window_wav, probe_wav, float(a), float(b))
                    try:
                        t = transcribe_wav(st.session_state.model, probe_wav, language=lang)
                    except Exception:
                        t = {}
                    txt = (t.get("text") or "").strip() if isinstance(t, dict) else ""
                    if txt:
                        nonempty += 1
                    if isinstance(t, dict):
                        guesses.append(extract_language_info(t).get("language_guess", "unknown"))

                    qs_done += 1.0
                    if probe_idx == 1 or probe_idx == len(sampled_intervals) or probe_idx % 3 == 0:
                        _update_progress_bar(
                            quick_scan_progress,
                            quick_scan_started,
                            qs_done,
                            qs_total,
                            f"Quick Scan probes ({probe_idx}/{max(len(sampled_intervals), 1)}) in window '{w_label}'",
                        )

                seg_per_min = (len(intervals) / max((w_end - w_start) / 60.0, 1e-9)) if (w_end - w_start) > 0 else 0.0
                probe_den = max(len(sampled_intervals), 1)
                rows.append({
                    "source": f"{track_artist} - {track_title}",
                    "window": w_label,
                    "method_used": method_used,
                    "segments": len(intervals),
                    "segments/min": round(seg_per_min, 2),
                    "transcript_nonempty_pct": round((nonempty / probe_den) * 100.0, 1),
                    "probes_used": len(sampled_intervals),
                    "language_guess": max(set(guesses), key=guesses.count) if guesses else "unknown",
                    "estimated_total_clips": int(round(seg_per_min * (duration_sec / 60.0))),
                    "noise_floor_db": silence_debug.get("noise_floor_db", None),
                    "silence_thresh_db": silence_debug.get("silence_thresh_db", None),
                    "margin_db": silence_debug.get("margin_db", None),
                    "quick_test_attempts": len(silence_debug.get("quick_test", {}).get("attempts", [])) if isinstance(silence_debug, dict) else 0,
                })

        _update_progress_bar(
            quick_scan_progress,
            quick_scan_started,
            qs_total,
            qs_total,
            "Quick scan finished",
        )
        st.session_state["quick_scan_rows"] = rows
        if not rows:
            st.warning("Quick Scan found no usable windows/segments with current settings.")

if st.session_state.get("quick_scan_rows"):
    st.subheader("Quick Scan Report")
    st.dataframe(pd.DataFrame(st.session_state["quick_scan_rows"]), use_container_width=True)
    if st.button("Proceed with Full Run", key="quick_scan_full_run_hint"):
        st.session_state["quick_scan_proceed"] = True

# ----------------------------
# Processing Logic
# ----------------------------
if run_btn or st.session_state.pop("quick_scan_proceed", False):
    mode_now = st.session_state["mode"]
    st.session_state["quick_scan_rows"] = []
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    run_scope_dir = ensure_dir(OUTPUT_ROOT / run_id)
    run_meta = {
        "run_id": run_id,
        "mode": mode_now,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "input_count": len(input_paths),
        "completed": False,
        "clips_exported": 0,
    }
    (run_scope_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

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
            run_started = perf_counter()
            run_progress = st.progress(0.0, text="  0% | Run started...")
            progress_state = {"done": 0.0, "total": max(1.0, float(total_files * 6))}

            def tick_progress(label: str) -> None:
                _update_progress_bar(
                    run_progress,
                    run_started,
                    progress_state["done"],
                    progress_state["total"],
                    label,
                )

            tick_progress("Preparing processing pipeline")
    
            for idx_file, (src_type, name_or_path, maybe_bytes, youtube_info) in enumerate(input_paths):
                status.update(
                    label=f"📦 Processing file {idx_file+1} of {total_files}",
                    state="running"
                )
                progress_state["done"] += 1.0
                tick_progress(f"File {idx_file+1}/{total_files}: staging input")
                
                st.write(f"### File {idx_file+1}/{total_files}: {name_or_path}")
                
                # Step 1: Save and convert to analysis format
                in_path, session_dir, in_name, yt_info = save_input_to_session_dir(src_type, name_or_path, maybe_bytes, youtube_info, run_id=run_id)
                
                # Extract track metadata (artist, title, track_id)
                track_artist, track_title, track_id = extract_track_metadata(in_path, yt_info)
                st.write(f"🎵 Track: **{track_artist} - {track_title}**")
    
                source_duration_sec = cached_duration_seconds(str(in_path))
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
                progress_state["done"] += 1.0
                tick_progress(f"File {idx_file+1}/{total_files}: audio converted")
                
                # Auto-detect audio type
                st.write("🔍 Detecting audio type...")
                audio_detection = detect_audio_type(
                    wav16,
                    sr=16000,
                    duration=30.0,
                    known_duration_sec=source_duration_sec,
                )
                st.write(f"🎯 **Audio Type:** {audio_detection['audio_type_guess']} (confidence: {audio_detection['audio_type_confidence']:.2f})")
                st.write(f"💡 **Recommended Mode:** {audio_detection['recommended_mode']}")
                progress_state["done"] += 1.0
                tick_progress(f"File {idx_file+1}/{total_files}: audio type detected")
    
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
                progress_state["done"] += 1.0
                tick_progress(f"File {idx_file+1}/{total_files}: language detected")
                
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
                    progress_state["done"] += 1.0
                    tick_progress(f"File {idx_file+1}/{total_files}: hooks detected")
                    
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
                    min_duration = max(float(st.session_state["hook_len_range_min"]), float(MIN_CLIP_DURATION_SECONDS))
                    candidates = filter_by_duration(candidates, min_duration=min_duration)
                    st.write(f"✂️ After {min_duration}s filter: {len(candidates)} hooks")
                    
                    # Apply anti-overlap
                    candidates = anti_overlap_keep_best(candidates)
                    st.write(f"🔄 After anti-overlap: {len(candidates)} hooks")
                    
                    # Process each hook
                    progress_bar = st.progress(0)
                    total_hooks = len(candidates)
                    progress_state["total"] += float(max(total_hooks, 1))
                    
                    for idx, cand in enumerate(candidates, start=1):
                        progress_bar.progress(idx / max(total_hooks, 1))
                        progress_state["done"] += 1.0
                        if idx == 1 or idx == total_hooks or idx % 3 == 0:
                            tick_progress(
                                f"File {idx_file+1}/{total_files}: processing hook {idx}/{max(total_hooks, 1)}"
                            )
                        
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
                        if dur < MIN_CLIP_DURATION_SECONDS:
                            # Hard safety floor for all exported clips.
                            continue
                        
                        # Get preference for bar snapping
                        prefer_bars = int(st.session_state.get("prefer_bars", 2))
                        
                        # Determine BPM and bars with proper tracking + half/double-time safeguards
                        bpm_clip_raw = float(cand.get("bpm", global_bpm))
                        bpm_clip_confidence = cand.get("bpm_clip_confidence", 0.0)
                        bpm_clip = int(normalize_bpm_family(bpm_clip_raw, bpm_clip_confidence))
                        if int(global_bpm) > 0:
                            bpm_clip_aligned, bpm_clip_alignment_source = normalize_bpm_to_prior(
                                bpm_clip,
                                int(global_bpm),
                                tolerance=0.20,
                            )
                            if bpm_clip_alignment_source == "track_global":
                                bpm_clip = int(global_bpm)
                        else:
                            bpm_clip_alignment_source = "segment_estimate"
                        
                        if refined_ok:
                            refined_candidate = int(
                                normalize_bpm_family(
                                    (bpm_refined if bpm_refined else global_bpm),
                                    float(bpm_conf),
                                )
                            )
                            if int(global_bpm) > 0:
                                refined_aligned, refined_source = normalize_bpm_to_prior(
                                    refined_candidate,
                                    int(global_bpm),
                                    tolerance=0.20,
                                )
                                if refined_source == "track_global":
                                    bpm_used = int(global_bpm)
                                    bpm_used_source = "refined_global_aligned"
                                else:
                                    bpm_used = int(refined_aligned)
                                    bpm_used_source = "refined_grid"
                            else:
                                bpm_used = refined_candidate
                                bpm_used_source = "refined_grid"
                            raw_bars_estimate = int(bars_est) if bars_est else estimate_bars_from_duration(dur, max(int(bpm_used), 1), beats_per_bar)
                            bars_used = int(bars) if bars else int(prefer_bars)
                            final_bpm_conf = float(bpm_conf)
                            refined_reason = ""
                        else:
                            raw_bars_estimate = int(bars_est) if bars_est else estimate_bars_from_duration(dur, max(int(bpm_clip), 1), beats_per_bar)
                            bars_used = int(snap_bars_to_valid(raw_bars_estimate, prefer_bars, tolerance=0.25))
                            bpm_used = int(bpm_clip)
                            if cand.get("bpm_source") == "track_global" or bpm_clip_alignment_source == "track_global":
                                bpm_used_source = "global_snapped"
                                final_bpm_conf = float(cand.get("bpm_confidence", global_confidence))
                            else:
                                bpm_used_source = "segment_estimate"
                                final_bpm_conf = float(bpm_clip_confidence)
                            refined_reason = rreason

                        uid = clip_uid(in_name, aa, bb, idx)
                        temp_wav = session_dir / f"temp_{idx:04d}__whisper.wav"
                        cut_segment_to_wav(in_path, temp_wav, aa, bb)
                        tjson = transcribe_wav(st.session_state.model, temp_wav, language=lang)
                        text = (tjson.get("text") or "").strip()

                        clip_lang_info = extract_language_info(tjson)
                        clip_language = resolve_clip_language(
                            language_meta["language_guess_file"],
                            language_meta["language_confidence_file"],
                            clip_lang_info,
                            text,
                        )

                        text_signature = normalize_text_for_signature(text, max_words=10)
                        slug = ""
                        if st.session_state["use_slug"] and text:
                            slug = safe_slug(" ".join(text.split()[:int(st.session_state["slug_words"])]), max_len=MAX_SLUG_LENGTH)

                        if refined_ok:
                            bpm_part = f"{bpm_used}bpm"
                            bars_part = f"{bars_used}bar"
                        else:
                            bpm_part = f"{bpm_used}bpm_low" if final_bpm_conf < 0.4 else f"{bpm_used}bpm_est"
                            bars_part = f"{bars_used}bar_est"

                        slug_part = slug if slug else "noslug"
                        stem = f"{track_artist}-{track_title}__{idx:04d}__{bpm_part}__{bars_part}__{slug_part}__{uid}"

                        if len(stem) > MAX_STEM_LENGTH:
                            excess = len(stem) - MAX_STEM_LENGTH
                            if len(slug_part) > 10:
                                slug_part = slug_part[:max(4, len(slug_part) - excess)]
                                stem = f"{track_artist}-{track_title}__{idx:04d}__{bpm_part}__{bars_part}__{slug_part}__{uid}"
                            if len(stem) > MAX_STEM_LENGTH:
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
                    progress_state["done"] += 1.0
                    tick_progress(f"File {idx_file+1}/{total_files}: Song Hunter completed")
                
                # ----------------------------
                # Mode: Broadcast Hunter (Mix)
                # ----------------------------
                else:
                    from broadcast_splitter import detect_broadcast_segments, get_last_broadcast_segmentation_debug

                    export_without_transcript = bool(st.session_state.get("export_without_transcript", True))
                    min_segment_sec = BROADCAST_MIN_SEGMENT_SEC
                    max_segment_sec, merge_gap_sec, runtime_chunk_sec = _broadcast_runtime_settings()
                    silence_cfg = _silence_runtime_settings()
                    prefer_method = "vad" if st.session_state["broadcast_split_method"].startswith("VAD") else "energy"
                    transcribe_segments = not export_without_transcript

                    if not transcribe_segments:
                        st.info("ℹ️ Transcript skipped in Broadcast mode (faster long-file processing).")

                    def _run_broadcast_pass(pass_name: str, min_seg: float, max_seg: float, quiet_filter_enabled: bool, retry_used: bool = False):
                        st.write(f"📻 Segmenting ({pass_name})...")
                        intervals, split_method_used, chunking_enabled = detect_broadcast_segments(
                            wav16,
                            min_segment_sec=min_seg,
                            max_segment_sec=max_seg,
                            merge_gap_sec=merge_gap_sec,
                            chunk_sec=runtime_chunk_sec,
                            silence_noise_db=silence_cfg["noise_db"],
                            silence_min_s=st.session_state["min_silence_s"],
                            silence_pad_s=st.session_state["pad_s"],
                            silence_threshold_mode=silence_cfg["threshold_mode"],
                            silence_margin_db=silence_cfg["margin_db"],
                            silence_quick_test_mode=silence_cfg["quick_test_mode"],
                            silence_quick_test_seconds=silence_cfg["quick_test_seconds"],
                            silence_quick_test_retries=silence_cfg["quick_test_retries"],
                            prefer_method=prefer_method,
                        )
                        split_debug = get_last_broadcast_segmentation_debug()
                        silence_debug = split_debug.get("silence_debug", {}) if isinstance(split_debug, dict) else {}
                        st.write(f"✂️ Found {len(intervals)} segments via **{split_method_used}** (chunking={chunking_enabled})")
                        if split_method_used == "silence" and isinstance(silence_debug, dict) and silence_debug:
                            nf = silence_debug.get("noise_floor_db", None)
                            thr = silence_debug.get("silence_thresh_db", None)
                            margin = silence_debug.get("margin_db", None)
                            st.info(
                                "Silence auto-threshold: "
                                f"noise_floor={nf} dB, threshold={thr} dB, margin={margin} dB"
                            )
                            warning_txt = str(silence_debug.get("warning", "") or "").strip()
                            if warning_txt:
                                st.warning(warning_txt)
                            attempts = silence_debug.get("quick_test", {}).get("attempts", [])
                            if attempts:
                                with st.expander("Silence quick-test attempts", expanded=False):
                                    st.dataframe(pd.DataFrame(attempts), use_container_width=True)
                        progress_state["total"] += float(max(len(intervals), 1))

                        diagnostics = _broadcast_diagnostics_template()
                        diagnostics["segments_total"] = len(intervals)
                        if split_method_used == "silence" and isinstance(silence_debug, dict):
                            diagnostics["silence_noise_floor_db"] = silence_debug.get("noise_floor_db", None)
                            diagnostics["silence_thresh_db"] = silence_debug.get("silence_thresh_db", None)
                            diagnostics["silence_margin_db"] = silence_debug.get("margin_db", None)
                            diagnostics["silence_quick_test_attempts"] = len(
                                silence_debug.get("quick_test", {}).get("attempts", [])
                            )
                        progress_bar = st.progress(0)
                        source_rows = []

                        for idx, (a, b) in enumerate(intervals, start=1):
                            progress_bar.progress(idx / max(len(intervals), 1))
                            progress_state["done"] += 1.0
                            if idx == 1 or idx == len(intervals) or idx % 8 == 0:
                                tick_progress(
                                    f"File {idx_file+1}/{total_files}: broadcast {pass_name} segment {idx}/{max(len(intervals), 1)}"
                                )
                            a, b = float(a), float(b)
                            dur = max(0.0, b - a)
                            if dur < MIN_CLIP_DURATION_SECONDS:
                                diagnostics["segments_too_short"] += 1
                                continue

                            try:
                                t: Dict[str, Any] = {}
                                if transcribe_segments:
                                    temp_wav = session_dir / f"temp_{pass_name}_{idx:04d}__whisper.wav"
                                    cut_segment_to_wav(in_path, temp_wav, a, b)
                                    try:
                                        t = transcribe_wav(st.session_state.model, temp_wav, language=lang)
                                    except Exception:
                                        diagnostics["segments_transcribe_failed"] += 1
                                        t = {}

                                text = (t.get("text") or "").strip() if isinstance(t, dict) else ""
                                if transcribe_segments and not text:
                                    diagnostics["segments_transcript_empty"] += 1
                                    if not export_without_transcript:
                                        continue

                                if quiet_filter_enabled and not text:
                                    diagnostics["segments_too_quiet"] += 1
                                    continue

                                if transcribe_segments and isinstance(t, dict):
                                    clip_lang_info = extract_language_info(t)
                                    clip_language = resolve_clip_language(
                                        language_meta["language_guess_file"],
                                        language_meta["language_confidence_file"],
                                        clip_lang_info,
                                        text,
                                    )
                                else:
                                    clip_language = {
                                        "language_guess": language_meta["language_guess_file"],
                                        "language_confidence": language_meta["language_confidence_file"],
                                        "language_source": "file",
                                    }

                                text_signature = normalize_text_for_signature(text, max_words=10)
                                slug = ""
                                if st.session_state["use_slug"] and text:
                                    slug = safe_slug(" ".join(text.split()[:int(st.session_state["slug_words"])]), max_len=MAX_SLUG_LENGTH)

                                uid = clip_uid(in_name, a, b, idx)
                                stem = f"{track_artist} - {track_title}__{idx:04d}__{mmss(a)}-{mmss(b)}__{(slug if slug else 'noslug')}__{uid}"
                                clip_path, export_meta = export_clip_with_tail(
                                    in_path, session_dir, stem, a, b,
                                    st.session_state["export_format"],
                                    add_tail=False,
                                    add_fades=True,
                                    apply_zero_crossing=False,
                                )
                                txt_path = session_dir / f"{stem}.txt"
                                json_path = session_dir / f"{stem}.json"
                                txt_path.write_text((text or "") + "\n", encoding="utf-8")
                                json_path.write_text(json.dumps(t if isinstance(t, dict) else {}, ensure_ascii=False, indent=2), encoding="utf-8")

                                tags = auto_tags(text) if text else ["ukendt"]
                                score = float(jingle_score(text, dur))
                                themes = detect_themes(text)
                                diagnostics["segments_exported"] += 1
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
                                    "split_method_used": split_method_used,
                                    "chunking_enabled": chunking_enabled,
                                    "retry_used": retry_used,
                                })
                            except Exception:
                                diagnostics["segments_export_failed"] += 1
                                continue
                        progress_bar.empty()
                        return source_rows, diagnostics, split_method_used

                    source_rows, broadcast_diag, split_method_used = _run_broadcast_pass(
                        "primary",
                        min_seg=min_segment_sec,
                        max_seg=max_segment_sec,
                        quiet_filter_enabled=False,
                        retry_used=False,
                    )

                    if split_method_used == "energy" and broadcast_diag["segments_exported"] == 0 and broadcast_diag["segments_total"] > 0:
                        st.warning("No clips exported on energy pass. Retrying once with relaxed thresholds...")
                        relaxed_min = BROADCAST_MIN_SEGMENT_SEC
                        relaxed_max = min(90.0, max_segment_sec * 1.3)
                        source_rows, broadcast_diag, split_method_used = _run_broadcast_pass(
                            "retry",
                            min_seg=relaxed_min,
                            max_seg=relaxed_max,
                            quiet_filter_enabled=False,
                            retry_used=True,
                        )

                    if broadcast_diag["segments_total"] > 0 and broadcast_diag["segments_exported"] == 0:
                        st.error("0 clips exported")
                        diag_df = pd.DataFrame([{"metric": k, "value": v} for k, v in broadcast_diag.items()])
                        st.dataframe(diag_df, use_container_width=True)
                        (session_dir / "diagnostics.json").write_text(
                            json.dumps({"diagnostics": broadcast_diag, "split_method_used": split_method_used}, indent=2),
                            encoding="utf-8",
                        )
                        if st.button("Retry with relaxed thresholds", key=f"retry_relaxed_{idx_file}"):
                            relaxed_min = BROADCAST_MIN_SEGMENT_SEC
                            relaxed_max = min(90.0, max_segment_sec * 1.3)
                            source_rows, broadcast_diag, split_method_used = _run_broadcast_pass(
                                "manual_retry",
                                min_seg=relaxed_min,
                                max_seg=relaxed_max,
                                quiet_filter_enabled=False,
                                retry_used=True,
                            )
                            (session_dir / "diagnostics.json").write_text(
                                json.dumps({"diagnostics": broadcast_diag, "split_method_used": split_method_used}, indent=2),
                                encoding="utf-8",
                            )

                    (session_dir / "diagnostics.json").write_text(
                        json.dumps({"diagnostics": broadcast_diag, "split_method_used": split_method_used}, indent=2),
                        encoding="utf-8",
                    )
                    results.extend(source_rows)
                    progress_state["done"] += 1.0
                    tick_progress(f"File {idx_file+1}/{total_files}: Broadcast Hunter completed")

            
        
        tick_progress("Finalizing run outputs")
        st.session_state.results = results

        if results:
            result_df = pd.DataFrame(results)
            manifest_path = run_scope_dir / "manifest.csv"
            result_df.to_csv(manifest_path, index=False)
            manifest_recording_path = run_scope_dir / "manifest_by_recording.csv"
            manifest_length_path = run_scope_dir / "manifest_by_length_desc.csv"

            if {"source", "start_sec"}.issubset(result_df.columns):
                rec_cols = ["source", "start_sec"]
                rec_asc = [True, True]
                if "dur_sec" in result_df.columns:
                    rec_cols.append("dur_sec")
                    rec_asc.append(False)
                if "clip" in result_df.columns:
                    rec_cols.append("clip")
                    rec_asc.append(True)
                by_recording = result_df.sort_values(
                    rec_cols,
                    ascending=rec_asc,
                    kind="stable",
                )
                by_recording.to_csv(manifest_recording_path, index=False)
            else:
                result_df.to_csv(manifest_recording_path, index=False)

            if "dur_sec" in result_df.columns:
                len_cols = ["dur_sec"]
                len_asc = [False]
                if "source" in result_df.columns:
                    len_cols.append("source")
                    len_asc.append(True)
                if "start_sec" in result_df.columns:
                    len_cols.append("start_sec")
                    len_asc.append(True)
                if "clip" in result_df.columns:
                    len_cols.append("clip")
                    len_asc.append(True)
                by_length = result_df.sort_values(
                    len_cols,
                    ascending=len_asc,
                    kind="stable",
                )
                by_length.to_csv(manifest_length_path, index=False)
            else:
                result_df.to_csv(manifest_length_path, index=False)

            st.session_state["manifest_path"] = str(manifest_path)
            st.session_state["manifest_recording_path"] = str(manifest_recording_path)
            st.session_state["manifest_length_path"] = str(manifest_length_path)
        else:
            st.session_state.pop("manifest_recording_path", None)
            st.session_state.pop("manifest_length_path", None)

        if not results:
            status.update(label="⚠️ No clips found", state="error")
            st.error("No clips found with current settings. Try adjusting parameters.")
            st.session_state.pop("qa_report", None)
            _update_progress_bar(run_progress, run_started, progress_state["total"], progress_state["total"], "Run finished (no clips)")
        else:
            status.update(label=f"✅ Success! Found {len(results)} clips", state="complete")
            st.balloons()
            st.success(f"🎉 Finished! Found {len(results)} clips. Scroll down to preview.")
            _update_progress_bar(run_progress, run_started, progress_state["total"], progress_state["total"], "Run finished")

            if mode_now == "🎵 Song Hunter (Loops)":
                song_rows = [r for r in results if r.get("split_method_used") == "hook"]
                qa_report = {
                    "total_clips": len(song_rows),
                    "refined_ok": sum(1 for r in song_rows if r.get("refined", False)),
                    "refined_fail": sum(1 for r in song_rows if not r.get("refined", False)),
                    "global_bpm": int(song_rows[0].get("bpm_global", 0)) if song_rows else 0,
                    "global_confidence": float(song_rows[0].get("bpm_global_confidence", 0.0)) if song_rows else 0.0,
                }

                bpms_raw = [int(r.get("bpm_used", 0) or 0) for r in song_rows if int(r.get("bpm_used", 0) or 0) > 0]
                if bpms_raw:
                    qa_report["bpm_min_raw"] = min(bpms_raw)
                    qa_report["bpm_max_raw"] = max(bpms_raw)
                    qa_report["bpm_median_raw"] = int(pd.Series(bpms_raw).median())

                bpm_global_for_report = int(qa_report.get("global_bpm", 0) or 0)
                bpms_report: List[int] = []
                for r in song_rows:
                    bpm_val = int(r.get("bpm_used", 0) or 0)
                    if bpm_val <= 0:
                        continue
                    bpm_conf = float(r.get("bpm_clip_confidence", 0.0) or 0.0)
                    bpm_norm = int(normalize_bpm_family(bpm_val, bpm_conf))
                    if bpm_global_for_report > 0:
                        aligned_bpm, aligned_source = normalize_bpm_to_prior(
                            bpm_norm,
                            bpm_global_for_report,
                            tolerance=0.20,
                        )
                        if aligned_source == "track_global":
                            bpm_norm = bpm_global_for_report
                        else:
                            bpm_norm = int(aligned_bpm)
                    bpms_report.append(int(bpm_norm))

                if bpms_report:
                    qa_report["bpm_min"] = min(bpms_report)
                    qa_report["bpm_max"] = max(bpms_report)
                    qa_report["bpm_median"] = int(pd.Series(bpms_report).median())

                sorted_results = sorted(song_rows, key=lambda r: r.get("hook_score", 0), reverse=True)
                qa_report["top_hooks"] = [
                    {
                        "clip": r.get("clip"),
                        "score": round(r.get("hook_score", 0), 2),
                        "bpm": r.get("bpm_used"),
                        "bars": r.get("bars_used"),
                    }
                    for r in sorted_results[:10]
                ]

                reasons = {}
                for r in song_rows:
                    if not r.get("refined", False):
                        reason = r.get("refined_reason", "unknown")
                        reasons[reason] = reasons.get(reason, 0) + 1
                qa_report["refined_fail_reasons"] = reasons
                st.session_state["qa_report"] = qa_report
            else:
                st.session_state.pop("qa_report", None)

        run_meta["completed"] = True
        run_meta["finished_at"] = datetime.utcnow().isoformat() + "Z"
        run_meta["clips_exported"] = len(results)
        run_meta["manifest_path"] = st.session_state.get("manifest_path", "")
        (run_scope_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    except Exception as e:
        if "run_progress" in locals() and "run_started" in locals():
            try:
                _update_progress_bar(run_progress, run_started, 1.0, 1.0, "Run failed")
            except Exception:
                pass
        run_meta["completed"] = False
        run_meta["finished_at"] = datetime.utcnow().isoformat() + "Z"
        run_meta["error"] = f"{type(e).__name__}: {e}"
        (run_scope_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        (run_scope_dir / "diagnostics.json").write_text(
            json.dumps({"run_id": run_id, "mode": mode_now, "error": run_meta["error"]}, indent=2),
            encoding="utf-8",
        )
        crash_log_path = run_scope_dir / "crash.log"
        crash_payload = (
            f"run_id={run_id}\n"
            f"mode={mode_now}\n"
            f"error={type(e).__name__}: {e}\n\n"
            f"traceback:\n{traceback.format_exc()}"
        )
        crash_log_path.write_text(crash_payload, encoding="utf-8")
        st.error("❌ Run failed, see crash.log")
        _download_button_from_path(
            "⬇️ Download crash.log",
            crash_log_path,
            mime="text/plain",
            key=f"dl_crash_{run_id}",
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
            st.audio(str(p))

# Restore last manifest on rerun if in-memory results are missing
if (not st.session_state.get("results")) and st.session_state.get("manifest_path"):
    manifest_path = Path(st.session_state["manifest_path"])
    if manifest_path.exists():
        try:
            restored_df = pd.read_csv(manifest_path)
            restored_df = restored_df.where(pd.notna(restored_df), None)
            st.session_state["results"] = restored_df.to_dict(orient="records")
        except Exception:
            pass

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
                if (
                    "bpm_min_raw" in qa
                    and (qa.get("bpm_min_raw"), qa.get("bpm_max_raw"), qa.get("bpm_median_raw"))
                    != (qa.get("bpm_min"), qa.get("bpm_max"), qa.get("bpm_median"))
                ):
                    st.caption(
                        f"Raw BPM range before global alignment: "
                        f"{qa['bpm_min_raw']} - {qa['bpm_max_raw']} (median: {qa['bpm_median_raw']})"
                    )
            
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

    table_sort_map: Dict[str, Tuple[List[str], List[bool]]] = {
        "As captured": ([], []),
    }
    if {"source", "start_sec"}.issubset(df.columns):
        table_rec_cols = ["source", "start_sec"]
        table_rec_asc = [True, True]
        if "dur_sec" in df.columns:
            table_rec_cols.append("dur_sec")
            table_rec_asc.append(False)
        if "clip" in df.columns:
            table_rec_cols.append("clip")
            table_rec_asc.append(True)
        table_sort_map["Recording order (source + time)"] = (
            table_rec_cols,
            table_rec_asc,
        )
    if "dur_sec" in df.columns:
        table_len_cols = ["dur_sec"]
        table_len_desc = [False]
        table_len_asc = [True]
        if "source" in df.columns:
            table_len_cols.append("source")
            table_len_desc.append(True)
            table_len_asc.append(True)
        if "start_sec" in df.columns:
            table_len_cols.append("start_sec")
            table_len_desc.append(True)
            table_len_asc.append(True)
        if "clip" in df.columns:
            table_len_cols.append("clip")
            table_len_desc.append(True)
            table_len_asc.append(True)
        table_sort_map["Length (longest first)"] = (
            table_len_cols,
            table_len_desc,
        )
        table_sort_map["Length (shortest first)"] = (
            table_len_cols,
            table_len_asc,
        )

    default_table_sort = "Recording order (source + time)" if "Recording order (source + time)" in table_sort_map else "As captured"
    table_sort_label = st.selectbox(
        "Table order",
        list(table_sort_map.keys()),
        index=list(table_sort_map.keys()).index(default_table_sort),
        key="results_table_sort",
    )
    table_sort_cols, table_sort_asc = table_sort_map[table_sort_label]
    if table_sort_cols:
        df_view = df.sort_values(table_sort_cols, ascending=table_sort_asc, kind="stable")
    else:
        df_view = df.copy()
    
    edited = st.data_editor(
        df_view,
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

    manifest_recording_path = Path(str(st.session_state.get("manifest_recording_path", "")))
    manifest_length_path = Path(str(st.session_state.get("manifest_length_path", "")))
    if manifest_recording_path.exists() or manifest_length_path.exists():
        dl_manifest_col1, dl_manifest_col2 = st.columns(2)
        if manifest_recording_path.exists():
            with dl_manifest_col1:
                _download_button_from_path(
                    "⬇️ Download manifest (recording order)",
                    manifest_recording_path,
                    mime="text/csv",
                    key="dl_manifest_recording_order",
                )
        if manifest_length_path.exists():
            with dl_manifest_col2:
                _download_button_from_path(
                    "⬇️ Download manifest (length desc)",
                    manifest_length_path,
                    mime="text/csv",
                    key="dl_manifest_length_desc",
                )

    # Grouping controls
    col_group1, col_group2 = st.columns([1, 1])
    with col_group1:
        grouping_options = ["None", "Group by Phrase", "Group by Tag/Theme", "Group by Language"]
        group_by = st.selectbox("Grouping", grouping_options, index=0, key="group_by")
    with col_group2:
        # Preview controls: sort + show all/paginated
        preview_sort_map: Dict[str, Tuple[List[str], List[bool]]] = {}
        if {"source", "start_sec"}.issubset(selected.columns):
            preview_rec_cols = ["source", "start_sec"]
            preview_rec_asc = [True, True]
            if "dur_sec" in selected.columns:
                preview_rec_cols.append("dur_sec")
                preview_rec_asc.append(False)
            if "clip" in selected.columns:
                preview_rec_cols.append("clip")
                preview_rec_asc.append(True)
            preview_sort_map["Recording order"] = (
                preview_rec_cols,
                preview_rec_asc,
            )
        if "dur_sec" in selected.columns:
            preview_len_cols = ["dur_sec"]
            preview_len_desc = [False]
            preview_len_asc = [True]
            if "source" in selected.columns:
                preview_len_cols.append("source")
                preview_len_desc.append(True)
                preview_len_asc.append(True)
            if "start_sec" in selected.columns:
                preview_len_cols.append("start_sec")
                preview_len_desc.append(True)
                preview_len_asc.append(True)
            if "clip" in selected.columns:
                preview_len_cols.append("clip")
                preview_len_desc.append(True)
                preview_len_asc.append(True)
            preview_sort_map["Length (longest first)"] = (
                preview_len_cols,
                preview_len_desc,
            )
            preview_sort_map["Length (shortest first)"] = (
                preview_len_cols,
                preview_len_asc,
            )
        if "hook_score" in selected.columns:
            preview_sort_map["Hook score"] = (["hook_score"], [False])
        if "jingle_score" in selected.columns:
            preview_sort_map["Jingle score"] = (["jingle_score"], [False])
        if "energy" in selected.columns:
            preview_sort_map["Energy"] = (["energy"], [False])
        if "clip" in selected.columns:
            preview_sort_map["Clip #"] = (["clip"], [True])
        if not preview_sort_map:
            preview_sort_map["As listed"] = ([], [])

        preview_sort = st.selectbox(
            "Sort by",
            list(preview_sort_map.keys()),
            index=0,
            key="preview_sort",
        )
        preview_sort_cols, preview_sort_asc = preview_sort_map[preview_sort]

    if preview_sort_cols:
        selected = selected.sort_values(preview_sort_cols, ascending=preview_sort_asc, kind="stable")

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
                # Keep insertion order from selected_rows so preview sort drives group order too.
                for group_key, items in groups.items():
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

        source_artifacts = st.session_state.get("source_artifacts", [])
        download_dir = _resolve_download_dir()

        run_zip_entries: List[Tuple[Path, str]] = []
        for artifact in source_artifacts:
            src_stem = Path(artifact["source"]).stem
            for key in ["transcript_full_txt_path", "transcript_full_json_path"]:
                fp = Path(artifact.get(key, ""))
                if fp.exists():
                    run_zip_entries.append((fp, f"{src_stem}/{fp.name}"))

        if run_zip_entries:
            run_transcript_zip_path = download_dir / "transcripts_full_run.zip"
            _write_zip_to_path(run_transcript_zip_path, run_zip_entries)
            _download_button_from_path(
                "⬇️ Download full transcript bundle (run)",
                run_transcript_zip_path,
                mime="application/zip",
                key="dl_transcripts_full_run",
                use_container_width=True,
            )

        for artifact_idx, artifact in enumerate(source_artifacts, start=1):
            src = artifact["source"]
            src_stem = Path(src).stem
            with st.expander(f"📄 {src}", expanded=False):
                c1, c2 = st.columns(2)
                txt_path = Path(artifact["transcript_full_txt_path"])
                json_path = Path(artifact["transcript_full_json_path"])
                if txt_path.exists():
                    with c1:
                        _download_button_from_path(
                            f"⬇️ Download full transcript (.txt) — {src_stem}",
                            txt_path,
                            mime="text/plain",
                            key=f"dl_txt_{artifact_idx}",
                            file_name=f"{src_stem}_transcript_full.txt",
                        )
                if json_path.exists():
                    with c2:
                        _download_button_from_path(
                            f"⬇️ Download full transcript (.json) — {src_stem}",
                            json_path,
                            mime="application/json",
                            key=f"dl_json_{artifact_idx}",
                            file_name=f"{src_stem}_transcript_full.json",
                        )

                source_selected = selected[selected["source"] == src] if "source" in selected.columns else pd.DataFrame([])
                src_zip_entries: List[Tuple[Path, str]] = []
                if txt_path.exists():
                    src_zip_entries.append((txt_path, f"{src_stem}/{txt_path.name}"))
                if json_path.exists():
                    src_zip_entries.append((json_path, f"{src_stem}/{json_path.name}"))
                for _, row in source_selected.iterrows():
                    for k in ["clip_path", "txt", "json", "transcript_full_txt_path", "transcript_full_json_path"]:
                        raw = row.get(k, "")
                        if raw is None or (isinstance(raw, float) and pd.isna(raw)) or str(raw).strip() == "":
                            continue
                        fp = Path(str(raw))
                        if fp.exists():
                            src_zip_entries.append((fp, f"{src_stem}/{fp.name}"))

                if src_zip_entries:
                    src_slug = safe_slug(src_stem, max_len=80) or f"source_{artifact_idx:03d}"
                    src_zip_path = download_dir / f"{src_slug}_bundle.zip"
                    _write_zip_to_path(src_zip_path, src_zip_entries)
                    _download_button_from_path(
                        f"📦 Download per-source ZIP — {src_stem}",
                        src_zip_path,
                        mime="application/zip",
                        key=f"dl_zip_{artifact_idx}",
                        file_name=f"{src_stem}_bundle.zip",
                    )
                else:
                    st.caption("No files available for source ZIP.")

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
        download_dir = _resolve_download_dir()
        selected_zip_path = download_dir / "sample_machine_clips_selected.zip"
        selected_entries: List[Tuple[Path, str]] = []
        for _, r in selected.iterrows():
            src_stem = Path(r["source"]).stem
            for k in ["clip_path", "txt", "json", "transcript_full_txt_path", "transcript_full_json_path"]:
                raw = r.get(k, "")
                if raw is None or (isinstance(raw, float) and pd.isna(raw)) or str(raw).strip() == "":
                    continue
                fp = Path(str(raw))
                if fp.exists():
                    selected_entries.append((fp, f"{src_stem}/{fp.name}"))

        _write_zip_to_path(
            selected_zip_path,
            selected_entries,
            memory_entries=[
                ("manifest_selected.csv", selected.to_csv(index=False).encode("utf-8")),
            ],
        )
        _download_button_from_path(
            label="⬇️ Download ZIP",
            path=selected_zip_path,
            file_name="sample_machine_clips.zip",
            mime="application/zip",
            use_container_width=True,
            key=f"dl_selected_zip_{st.session_state.get('active_run_id', 'run')}",
        )
else:
    st.info("👆 Upload or download a file, load the Whisper model, and click Process to begin.")
