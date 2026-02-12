import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from utils import ensure_dir, hhmmss_ms, safe_slug
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

from hook_finder import (
    find_hooks,
    find_chorus_windows,
    refine_loops_within_window,
    beat_aligned_windows,
)
from beat_refine import refine_best_1_or_2_bars

st.set_page_config(page_title="Radio Splitter + Hooks + Whisper", layout="wide")
st.title("Radio Splitter + Hooks + Whisper (gratis, lokalt / cloud)")
st.caption("Upload/Link → analyse → vælg klip → eksport ZIP")

output_root = ensure_dir("output")


# ---------------- Mode + Presets ----------------
MODES = [
    "Radio: Split på stilhed + Whisper",
    "Jingles: Fast length + sortering",
    "Song: Find hooks/loops (4–15s) (beat-aware)",
]

if "mode" not in st.session_state:
    st.session_state["mode"] = MODES[0]

st.sidebar.header("Mode")
mode = st.sidebar.selectbox("Vælg mode", MODES, index=MODES.index(st.session_state["mode"]), key="mode")
st.session_state["mode"] = mode


# ---------------- Whisper settings ----------------
st.sidebar.header("Whisper")
model_size = st.sidebar.selectbox("Model", ["tiny", "base", "small", "medium"], index=2)
language = st.sidebar.selectbox("Sprog", ["da", "en", "auto"], index=0)
device = st.sidebar.selectbox("Device", ["cpu"], index=0)
compute_type = st.sidebar.selectbox("Compute", ["int8", "float32"], index=0)

use_whisper_for_naming = st.sidebar.checkbox("Brug Whisper til navngivning/tekst", value=True)
use_slug = st.sidebar.checkbox("Tilføj slug fra transskription", value=True)
slug_words = st.sidebar.slider("Slug ord (ca.)", 2, 12, 6)

export_format = st.sidebar.selectbox("Eksportformat", ["wav (16000 mono)", "mp3 (192k)"], index=0)


# ---------------- Mode-specific controls ----------------
if mode.startswith("Radio"):
    st.sidebar.header("Radio split (stilhed)")
    noise_db = st.sidebar.slider("Silence threshold (dB)", -60.0, -10.0, -35.0, 1.0)
    min_silence_s = st.sidebar.slider("Min stilhed (sek)", 0.2, 2.0, 0.7, 0.1)
    pad_s = st.sidebar.slider("Padding før/efter (sek)", 0.0, 1.0, 0.15, 0.05)
    min_segment_s = st.sidebar.slider("Min kliplængde (sek)", 0.5, 10.0, 1.2, 0.1)

elif mode.startswith("Jingles"):
    st.sidebar.header("Jingle split (fast length)")
    fixed_len = st.sidebar.slider("Klip-længde (sek)", 2.0, 30.0, 8.0, 1.0)
    min_score = st.sidebar.slider("Min jingle-score (filter)", 0.0, 10.0, 3.0, 0.25)

else:
    st.sidebar.header("Song hooks")
    hook_lo = st.sidebar.slider("Min hook (sek)", 2.0, 12.0, 4.0, 0.5)
    hook_hi = st.sidebar.slider("Max hook (sek)", 6.0, 25.0, 15.0, 0.5)
    prefer_len = st.sidebar.slider("Foretrukken længde (sek)", 3.0, 18.0, 8.0, 0.5)
    topn = st.sidebar.slider("Top hooks", 4, 30, 12, 1)

    st.sidebar.subheader("Chorus-aware")
    chorus_mode = st.sidebar.checkbox("Find chorus-vinduer (30–45s) først", value=True)
    chorus_lo = st.sidebar.slider("Chorus min (sek)", 10.0, 40.0, 30.0, 1.0)
    chorus_hi = st.sidebar.slider("Chorus max (sek)", 20.0, 70.0, 45.0, 1.0)

    st.sidebar.subheader("Beat refine (bar-grid)")
    do_refine = st.sidebar.checkbox("Auto: refine til 1–2 bar loop", value=True)
    prefer_bars = st.sidebar.selectbox("Foretræk bars", [1, 2], index=0)
    beats_per_bar = st.sidebar.selectbox("Beats pr bar", [3, 4], index=1)


# ---------------- Input: Link download ----------------
st.subheader("A) Link download (Ctrl+V)")
url = st.text_input("Link", placeholder="https://... (YouTube / Archive / etc.)")
dl_col1, dl_col2 = st.columns([1, 3])
with dl_col1:
    dl_btn = st.button("Download", type="primary", disabled=not url.strip())
with dl_col2:
    st.caption("Gemmer i: output/Downloads/")

if dl_btn:
    try:
        with st.spinner("Downloader..."):
            p = download_audio(url.strip(), Path("output") / "Downloads")
        st.success(f"Downloaded: {p.name}")
        st.session_state.setdefault("downloaded_files", [])
        st.session_state["downloaded_files"].append(str(p))
    except Exception as e:
        st.error(f"Download fejl: {e}")


# ---------------- Input: Upload ----------------
st.subheader("B) Upload MP3/WAV")
files = st.file_uploader("Upload", type=["mp3", "wav"], accept_multiple_files=True)

input_items = []
if files:
    for uf in files:
        input_items.append(("upload", uf.name, uf.getvalue()))
for p in st.session_state.get("downloaded_files", []):
    input_items.append(("path", p, None))


# ---------------- Whisper model state ----------------
if "model_obj" not in st.session_state:
    st.session_state["model_obj"] = None

colA, colB = st.columns([1, 1])
with colA:
    load_btn = st.button("Load Whisper-model", type="primary", disabled=not use_whisper_for_naming)
with colB:
    run_btn = st.button("Kør analyse", type="primary", disabled=not input_items)

if load_btn:
    try:
        with st.spinner("Loader model... (første gang downloader den)"):
            st.session_state["model_obj"] = load_model(model_size=model_size, device=device, compute_type=compute_type)
        st.success("Model loaded.")
    except Exception as e:
        st.error(f"Kunne ikke loade model: {e}")


def _transcribe_if_needed(wav_path: Path, lang_opt: str):
    if not use_whisper_for_naming:
        return {"text": "", "segments": []}
    if st.session_state["model_obj"] is None:
        return {"text": "", "segments": []}
    lang = None if lang_opt == "auto" else lang_opt
    return transcribe_wav(st.session_state["model_obj"], wav_path, language=lang)


def _export_clip(src_path: Path, out_path: Path, a: float, b: float, bitrate="192k"):
    if export_format.startswith("wav"):
        cut_segment_to_wav(src_path, out_path.with_suffix(".wav"), a, b)
        return out_path.with_suffix(".wav")
    else:
        cut_segment_to_mp3(src_path, out_path.with_suffix(".mp3"), a, b, bitrate=bitrate)
        return out_path.with_suffix(".mp3")


# ---------------- Run pipeline ----------------
if run_btn:
    results = []
    for src_type, name_or_path, maybe_bytes in input_items:
        if src_type == "upload":
            in_name = name_or_path
            session_dir = ensure_dir(output_root / Path(in_name).stem)
            in_path = session_dir / in_name
            in_path.write_bytes(maybe_bytes)
        else:
            in_path = Path(name_or_path)
            in_name = in_path.name
            session_dir = ensure_dir(output_root / in_path.stem)
            local_in = session_dir / in_name
            if not local_in.exists():
                local_in.write_bytes(in_path.read_bytes())
            in_path = local_in

        st.write(f"### Processing: {in_name}")

        # 1) Build candidate intervals per mode
        if mode.startswith("Radio"):
            intervals = detect_non_silent_intervals(
                in_path,
                noise_db=noise_db,
                min_silence_s=min_silence_s,
                pad_s=pad_s,
                min_segment_s=min_segment_s,
            )

        elif mode.startswith("Jingles"):
            intervals = fixed_length_intervals(in_path, segment_len=fixed_len)

        else:
            # Song mode: make a 16k wav for analysis, then find windows there
            analysis_wav = session_dir / "__analysis_16k.wav"
            cut_segment_to_wav(in_path, analysis_wav, 0.0, 10**9)

            if chorus_mode:
                choruses = find_chorus_windows(
                    analysis_wav,
                    chorus_len_range=(chorus_lo, chorus_hi),
                    topn=4,
                )
                hooks_all = []
                for cw in choruses:
                    hooks_all.extend(
                        refine_loops_within_window(
                            analysis_wav,
                            cw,
                            hook_len_range=(hook_lo, hook_hi),
                            prefer_len=prefer_len,
                            topn=max(6, topn),
                        )
                    )
                # sort and take topn with simple gap
                hooks_all.sort(key=lambda w: w.score, reverse=True)
                picked = []
                for w in hooks_all:
                    if len(picked) >= topn:
                        break
                    if all(abs(w.start - p.start) >= 1.5 for p in picked):
                        picked.append(w)
                intervals = [(w.start, w.end, w) for w in picked]
            else:
                hooks = beat_aligned_windows(
                    analysis_wav,
                    len_range=(hook_lo, hook_hi),
                    topn=topn,
                )
                intervals = [(w.start, w.end, w) for w in hooks]

        # normalize interval format
        if mode.startswith("Song"):
            st.info(f"Fundet {len(intervals)} hooks")
        else:
            st.info(f"Fundet {len(intervals)} klip")

        # 2) For each interval: cut -> whisper -> tag -> score -> (optional) beat-refine
        for idx, item in enumerate(intervals, start=1):
            if mode.startswith("Song"):
                a, b, w = item
            else:
                a, b = item
                w = None

            dur = max(0.0, float(b) - float(a))
            base = f"{idx:04d}_{hhmmss_ms(a)}_to_{hhmmss_ms(b)}"

            wav_for_whisper = session_dir / f"{base}__whisper.wav"
            cut_segment_to_wav(in_path, wav_for_whisper, a, b)

            t = _transcribe_if_needed(wav_for_whisper, language)
            text = (t.get("text") or "").strip()

            slug = ""
            if use_slug and text:
                slug = safe_slug(" ".join(text.split()[:slug_words]))

            final_stem = f"{base}__{slug}" if slug else base
            out_stem_path = session_dir / final_stem

            # optional refine for song mode (1-2 bar snap)
            refined_a, refined_b, refined_bpm, refined_bars, refine_score = None, None, None, None, None
            use_refined = False
            if mode.startswith("Song") and do_refine:
                rr = refine_best_1_or_2_bars(
                    str(wav_for_whisper),
                    0.0,
                    dur,
                    beats_per_bar=beats_per_bar,
                    prefer_bars=prefer_bars,
                )
                if rr.ok and (rr.end - rr.start) > 0.5:
                    refined_a = float(a + rr.start)
                    refined_b = float(a + rr.end)
                    refined_bpm = float(rr.bpm)
                    refined_bars = int(rr.bars)
                    refine_score = float(rr.score)
                    use_refined = True

            # export clip (use refined if present)
            exp_a = refined_a if use_refined else a
            exp_b = refined_b if use_refined else b
            clip_path = _export_clip(in_path, out_stem_path, exp_a, exp_b)

            txt_path = session_dir / f"{final_stem}.txt"
            json_path = session_dir / f"{final_stem}.json"
            txt_path.write_text(text + "\n", encoding="utf-8")
            json_path.write_text(json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8")

            tags = auto_tags(text)
            js = jingle_score(text, dur)

            row = {
                "source": in_name,
                "pick": True,
                "clip": idx,
                "start_sec": float(a),
                "end_sec": float(b),
                "dur_sec": float(dur),
                "tags": ", ".join(tags),
                "jingle_score": float(js),
                "clip_path": str(clip_path),
                "txt": str(txt_path),
                "json": str(json_path),
                "text": text[:240] if text else "",
                "mode": mode,
            }

            if mode.startswith("Song") and w is not None:
                row.update({
                    "hook_score": float(w.score),
                    "hook_energy": float(w.energy),
                    "hook_loopability": float(w.loopability),
                    "hook_stability": float(w.stability),
                    "hook_bpm_est": float(w.bpm),
                })

            if mode.startswith("Song") and do_refine:
                row.update({
                    "refined_used": bool(use_refined),
                    "refined_start": refined_a if refined_a is not None else "",
                    "refined_end": refined_b if refined_b is not None else "",
                    "refined_bpm": refined_bpm if refined_bpm is not None else "",
                    "refined_bars": refined_bars if refined_bars is not None else "",
                    "refine_score": refine_score if refine_score is not None else "",
                })

            results.append(row)

    st.session_state["results"] = results


# ---------------- Browser / selection / export ----------------
if "results" in st.session_state and st.session_state["results"]:
    st.divider()
    st.subheader("C) Klip-browser (vælg de gode)")

    df = pd.DataFrame(st.session_state["results"])

    # Sorting: song hooks by hook_score, else by jingle_score
    if mode.startswith("Song") and "hook_score" in df.columns:
        df = df.sort_values(["hook_score", "dur_sec"], ascending=[False, True])
    else:
        df = df.sort_values(["jingle_score", "dur_sec"], ascending=[False, True])

    edited = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "pick": st.column_config.CheckboxColumn("Gem", default=True),
            "text": st.column_config.TextColumn("Transcript (kort)", width="large"),
            "clip_path": st.column_config.TextColumn("Fil", width="large"),
        },
    )

    selected = edited[edited["pick"] == True].copy()
    st.write(f"**Valgt:** {len(selected)} klip")

    with st.expander("Preview af valgte (første 30)"):
        for _, r in selected.head(30).iterrows():
            p = Path(r["clip_path"])
            st.write(f"**{p.name}**  ({float(r['dur_sec']):.1f}s) | tags: {r.get('tags','')}")

            if mode.startswith("Song") and "hook_score" in r:
                st.caption(
                    f"hook_score={float(r.get('hook_score',0)):.2f}  "
                    f"loop={float(r.get('hook_loopability',0)):.2f}  "
                    f"stab={float(r.get('hook_stability',0)):.2f}  "
                    f"bpm≈{float(r.get('hook_bpm_est',0)):.1f}"
                )
                if str(r.get("refined_used", "")) == "True":
                    st.caption(
                        f"refined: bars={r.get('refined_bars','')} bpm={r.get('refined_bpm','')} score={r.get('refine_score','')}"
                    )

            if p.exists():
                st.audio(p.read_bytes())
            if r.get("text", ""):
                st.write(r["text"])

    if st.button("Eksportér ZIP (valgte)", type="primary"):
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest_selected.csv", selected.to_csv(index=False).encode("utf-8"))

            for _, r in selected.iterrows():
                src_stem = Path(r["source"]).stem
                for k in ["clip_path", "txt", "json"]:
                    p = Path(str(r[k]))
                    if p.exists():
                        z.write(p, arcname=f"{src_stem}/{p.name}")

        zip_buf.seek(0)
        st.download_button(
            "Download ZIP",
            data=zip_buf,
            file_name="radio_clips_selected.zip",
            mime="application/zip",
        )
else:
    st.info("Upload/download → (optional) load Whisper → Kør analyse.")
