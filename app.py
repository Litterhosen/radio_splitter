# CRITICAL: st.set_page_config MUST be the VERY FIRST Streamlit call
import streamlit as st
st.set_page_config(page_title="The Sample Machine", layout="wide")

import io
import json
import zipfile
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

# Local imports
from audio_split import (
    detect_non_silent_intervals,
    fixed_length_intervals,
    cut_segment_to_wav,
    cut_segment_to_mp3,
)
from transcribe import load_model, transcribe_wav
from downloaders import download_audio
from tagging import auto_tags
from jingle_finder import jingle_score
from hook_finder import ffmpeg_to_wav16k_mono, find_hooks
from beat_refine import refine_best_1_or_2_bars
from utils import ensure_dir, hhmmss_ms, safe_slug

# ----------------------------
# Constants and Config
# ----------------------------
OUTPUT_ROOT = Path("output")
ensure_dir(OUTPUT_ROOT)

# Anti-overlap and filter thresholds
OVERLAP_THRESHOLD = 0.30  # 30% overlap threshold for duplicate detection
MIN_DURATION_SECONDS = 4.0  # Minimum clip duration
DECAY_TAIL_DURATION = 0.75  # Extra audio tail for loops (seconds)

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
    "noise_db": -35.0,
    "min_silence_s": 0.7,
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
    "prefer_bars": 1,
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


def save_input_to_session_dir(src_type: str, name_or_path: str, maybe_bytes):
    """Save uploaded or downloaded file to session directory"""
    if src_type == "upload":
        in_name = name_or_path
        session_dir = ensure_dir(OUTPUT_ROOT / Path(in_name).stem)
        in_path = session_dir / in_name
        in_path.write_bytes(maybe_bytes)
        return in_path, session_dir, in_name

    p = Path(name_or_path)
    in_name = p.name
    session_dir = ensure_dir(OUTPUT_ROOT / p.stem)
    local_in = session_dir / in_name
    if not local_in.exists():
        local_in.write_bytes(p.read_bytes())
    return local_in, session_dir, in_name


def export_clip_with_tail(
    in_path: Path, 
    session_dir: Path, 
    stem: str, 
    a: float, 
    b: float, 
    want_format: str,
    add_tail: bool = True
) -> Path:
    """Export clip with optional decay tail"""
    tail_duration = DECAY_TAIL_DURATION if add_tail else 0.0
    b_with_tail = b + tail_duration
    
    # Determine file extension and suffix
    is_wav = want_format.startswith("wav")
    ext = "wav" if is_wav else "mp3"
    suffix = "_tail" if add_tail else ""
    outp = session_dir / f"{stem}{suffix}.{ext}"
    
    if is_wav:
        cut_segment_to_wav(in_path, outp, a, b_with_tail)
    else:
        cut_segment_to_mp3(in_path, outp, a, b_with_tail, bitrate="192k")
    
    return outp


def maybe_refine_barloop(wav_for_analysis: Path, a: float, b: float):
    """Refine loop to beat grid if enabled"""
    if not st.session_state["beat_refine"]:
        return a, b, 0, 0, 0.0, False, "disabled"

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
            ""
        )

    return a, b, rr.bpm, rr.bars, rr.score, False, rr.reason


# ----------------------------
# UI - Title and Description
# ----------------------------
st.title("🎛️ The Sample Machine")
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
    
    st.sidebar.subheader("🎼 Chorus Detection")
    st.sidebar.slider("Min chorus length (sec)", 10.0, 60.0, step=1.0, key="chorus_len_range_min")
    st.sidebar.slider("Max chorus length (sec)", 15.0, 90.0, step=1.0, key="chorus_len_range_max")
    
    # Validate that chorus min <= max
    if st.session_state["chorus_len_range_min"] > st.session_state["chorus_len_range_max"]:
        st.sidebar.error("⚠️ Min chorus length cannot exceed max chorus length")
    
    st.sidebar.slider("Chorus scan hop (sec)", 0.5, 10.0, step=0.5, key="chorus_hop")
    st.sidebar.slider("Top N choruses", 1, 10, step=1, key="chorus_topn")
    st.sidebar.slider("Min chorus gap (sec)", 2.0, 20.0, step=1.0, key="chorus_gap")
    st.sidebar.slider("Loops per chorus", 1, 20, step=1, key="loops_per_chorus")
    
    st.sidebar.subheader("🎼 Beat Refinement")
    st.sidebar.checkbox("Refine to beat grid", key="beat_refine")
    st.sidebar.number_input("Beats per bar", min_value=3, max_value=7, step=1, key="beats_per_bar")
    st.sidebar.radio("Preferred loop length", ["1 bar", "2 bars"], key="prefer_bars_ui")
    st.session_state["prefer_bars"] = 1 if st.session_state.get("prefer_bars_ui", "1 bar") == "1 bar" else 2
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
                p = download_audio(url.strip(), OUTPUT_ROOT / "Downloads")
            st.success(f"✅ Downloaded: {p.name}")
            st.session_state["downloaded_files"] = list(st.session_state.get("downloaded_files", [])) + [str(p)]
        except Exception as e:
            st.error(f"❌ Download error: {e}")

# Prepare input paths
input_paths = []
if files:
    for uf in files:
        input_paths.append(("upload", uf.name, uf.getvalue()))
for p in st.session_state.get("downloaded_files", []):
    input_paths.append(("path", p, None))

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

        for idx_file, (src_type, name_or_path, maybe_bytes) in enumerate(input_paths):
            status.update(
                label=f"📦 Processing file {idx_file+1} of {total_files}",
                state="running"
            )
            
            st.write(f"### File {idx_file+1}/{total_files}: {name_or_path}")
            
            # Step 1: Save and convert to analysis format
            in_path, session_dir, in_name = save_input_to_session_dir(src_type, name_or_path, maybe_bytes)
            st.write("⏳ Converting audio...")
            wav16 = session_dir / "_analysis_16k_mono.wav"
            ffmpeg_to_wav16k_mono(in_path, wav16)
            st.write("✅ Conversion complete.")
            
            # ----------------------------
            # Mode: Song Hunter (Loops)
            # ----------------------------
            if mode_now == "🎵 Song Hunter (Loops)":
                # Try Chorus detection first, fallback to Hooks
                st.write("🎼 Trying chorus detection...")
                chorus_hooks = find_hooks(
                    wav16,
                    hook_len_range=(
                        st.session_state["chorus_len_range_min"],
                        st.session_state["chorus_len_range_max"]
                    ),
                    prefer_len=(st.session_state["chorus_len_range_min"] + st.session_state["chorus_len_range_max"]) / 2.0,
                    hop_s=st.session_state["chorus_hop"],
                    topn=st.session_state["chorus_topn"],
                    min_gap_s=st.session_state["chorus_gap"],
                )
                
                if chorus_hooks:
                    st.write(f"🎼 Found {len(chorus_hooks)} chorus sections!")
                    hooks = chorus_hooks
                else:
                    st.write("⚠️ Chorus detection failed, falling back to Hooks logic...")
                    hooks = find_hooks(
                        wav16,
                        hook_len_range=(
                            st.session_state["hook_len_range_min"],
                            st.session_state["hook_len_range_max"]
                        ),
                        prefer_len=st.session_state["prefer_len"],
                        hop_s=st.session_state["hook_hop"],
                        topn=st.session_state["hook_topn"],
                        min_gap_s=st.session_state["hook_gap"],
                    )
                st.write(f"🎉 Found {len(hooks)} potential hooks!")
                
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
                    })
                
                # Apply 4-second filter
                candidates = filter_by_duration(candidates, min_duration=4.0)
                st.write(f"✂️ After 4s filter: {len(candidates)} hooks")
                
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
                    a2, b2, bpm, bars, rscore, refined_ok, rreason = maybe_refine_barloop(wav16, a, b)
                    aa, bb = (a2, b2) if refined_ok else (a, b)
                    dur = max(0.0, bb - aa)
                    
                    # If refined segment < 4s, fall back to original hook window
                    if refined_ok and dur < MIN_DURATION_SECONDS:
                        aa, bb = a, b
                        dur = max(0.0, bb - aa)
                        refined_ok = False
                    
                    # Final 4-second minimum check
                    if dur < MIN_DURATION_SECONDS:
                        continue
                    
                    # BPM: prefer refined BPM, fall back to hook finder BPM
                    clip_bpm = int(round(float(bpm))) if int(round(float(bpm))) > 0 else int(round(float(cand.get("bpm", 0))))
                    clip_bars = int(bars) if refined_ok else 0
                    
                    # Transcribe
                    base = f"{idx:04d}_{hhmmss_ms(aa)}_to_{hhmmss_ms(bb)}"
                    wav_for_whisper = session_dir / f"{base}__whisper.wav"
                    cut_segment_to_wav(in_path, wav_for_whisper, aa, bb)
                    tjson = transcribe_wav(st.session_state.model, wav_for_whisper, language=lang)
                    text = (tjson.get("text") or "").strip()
                    
                    # Generate slug
                    slug = ""
                    if st.session_state["use_slug"] and text:
                        slug = safe_slug(" ".join(text.split()[:int(st.session_state["slug_words"])]))
                    
                    # Build filename with BPM and bar info
                    bpm_tag = f"_bpm{clip_bpm}" if clip_bpm > 0 else ""
                    bar_tag = f"_{clip_bars}bar" if refined_ok and clip_bars > 0 else ""
                    stem = f"{base}{bpm_tag}{bar_tag}__{slug}" if slug else f"{base}{bpm_tag}{bar_tag}"
                    
                    # Export with tail
                    clip_path = export_clip_with_tail(
                        in_path, session_dir, stem, aa, bb,
                        st.session_state["export_format"],
                        add_tail=True
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
                    tags += [f"{clip_bars}bar"] if refined_ok else ["unrefined"]
                    if text:
                        tags = list(set(tags + auto_tags(text)))
                    themes = detect_themes(text)
                    
                    results.append({
                        "source": in_name,
                        "filename": clip_path.name,
                        "pick": True,
                        "clip": idx,
                        "start_sec": aa,
                        "end_sec": bb,
                        "dur_sec": dur,
                        "bpm": clip_bpm,
                        "bars": clip_bars,
                        "refined": bool(refined_ok),
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
                
                # Apply 4-second filter
                intervals = [(a, b) for a, b in intervals if (b - a) >= 4.0]
                st.write(f"✂️ Found {len(intervals)} segments (after 4s filter)")
                
                progress_bar = st.progress(0)
                total_int = len(intervals)
                
                for idx, (a, b) in enumerate(intervals, start=1):
                    progress_bar.progress(idx / total_int)
                    
                    a, b = float(a), float(b)
                    dur = max(0.0, b - a)
                    base = f"{idx:04d}_{hhmmss_ms(a)}_to_{hhmmss_ms(b)}"
                    
                    # Transcribe
                    wav_for_whisper = session_dir / f"{base}__whisper.wav"
                    cut_segment_to_wav(in_path, wav_for_whisper, a, b)
                    t = transcribe_wav(st.session_state.model, wav_for_whisper, language=lang)
                    text = (t.get("text") or "").strip()
                    
                    # Generate slug
                    slug = ""
                    if st.session_state["use_slug"] and text:
                        slug = safe_slug(" ".join(text.split()[:int(st.session_state["slug_words"])]))
                    
                    stem = f"{base}__{slug}" if slug else base
                    
                    # Export without tail for broadcast
                    clip_path = export_clip_with_tail(
                        in_path, session_dir, stem, a, b,
                        st.session_state["export_format"],
                        add_tail=False
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
                        "pick": True,
                        "clip": idx,
                        "start_sec": a,
                        "end_sec": b,
                        "dur_sec": dur,
                        "bpm": 0,
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

# ----------------------------
# Results Browser / Export
# ----------------------------
if "results" in st.session_state and st.session_state.results:
    st.divider()
    st.subheader("📊 Results Browser")
    
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
                    st.audio(p.read_bytes(), key=f"audio_{idx}_{r['filename']}")
    
    # Export ZIP
    if st.button("📦 Export ZIP (Selected)", type="primary", use_container_width=True, key="export_zip_btn"):
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
            key="dl_zip_selected",
        )
else:
    st.info("👆 Upload or download a file, load the Whisper model, and click Process to begin.")
