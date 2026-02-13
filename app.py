# -*- coding: utf-8 -*-
import io
import json
import zipfile
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st

from audio_split import (
    detect_non_silent_intervals,
    cut_segment_to_wav,
    cut_segment_to_mp3,
    mean_volume_db,
)
from beat_refine import refine_best_1_or_2_bars
from downloaders import download_audio, DownloadError
from hook_finder import ffmpeg_to_wav16k_mono, find_hooks, find_chorus_candidates
from tagging import auto_tags
from transcribe import load_model, transcribe_wav


OUTPUT_ROOT = Path("output")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MIN_EXPORT_DUR = 4.0
TAIL_PAD = 0.75

DEFAULTS = {
    "mode": "song",
    "model_size": "small",
    "language": "da",
    "device": "cpu",
    "compute_type": "int8",
    "whisper_on_hooks": False,
    "use_slug": True,
    "slug_words": 6,
    "export_format": "mp3 (192k)",
    "noise_db": -35.0,
    "min_silence_s": 0.7,
    "pad_s": 0.15,
    "min_segment_s": 4.0,
    "voice_min_words": 6,
    "music_min_db": -28.0,
    "hook_len_range": (4.0, 15.0),
    "prefer_len": 8.0,
    "hook_hop": 1.0,
    "hook_topn": 30,
    "hook_gap": 2.0,
    "beat_refine": True,
    "beats_per_bar": 4,
    "prefer_bars": 1,
    "try_both_bars": True,
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

st.set_page_config(page_title="Radio Splitter: The Sample Machine", layout="wide")

st.title("Radio Splitter: The Sample Machine")
st.caption("Upload lyd -> Find loops & temaer -> Byg dit arkiv.")


st.sidebar.header("Indstillinger")
mode_choice = st.sidebar.radio(
    "Mode",
    ["🎵 Song Hunter (Loops)", "📻 Broadcast Hunter (Mix)"],
    index=0 if st.session_state["mode"] == "song" else 1,
)
st.session_state["mode"] = "song" if "Song Hunter" in mode_choice else "broadcast"

st.sidebar.subheader("Whisper")
model_labels = [
    "Tiny (Hurtig)",
    "Base",
    "Small (Balance)",
    "Medium (Høj kvalitet/Langsom)",
]
model_map = {
    "Tiny (Hurtig)": "tiny",
    "Base": "base",
    "Small (Balance)": "small",
    "Medium (Høj kvalitet/Langsom)": "medium",
}
current_label = next(
    (k for k, v in model_map.items() if v == st.session_state["model_size"]),
    "Small (Balance)",
)
model_choice = st.sidebar.selectbox(
    "Whisper-model",
    model_labels,
    index=model_labels.index(current_label),
)
st.session_state["model_size"] = model_map[model_choice]

with st.sidebar.expander("🔧 Avancerede indstillinger"):
    st.subheader("Whisper avanceret")
    st.selectbox("Sprog", ["da", "en", "auto"], key="language")
    st.selectbox("Device", ["cpu"], key="device")
    st.selectbox("Compute type", ["int8", "float32"], key="compute_type")

    st.subheader("Navngivning")
    st.checkbox("Tilføj slug fra transskription", key="use_slug")
    st.slider("Slug ord (ca.)", 2, 12, key="slug_words")

    st.subheader("Eksport")
    st.selectbox("Klipformat", ["mp3 (192k)", "wav (16000 mono)"], key="export_format")

    st.subheader("Broadcast: split på stilhed")
    st.slider("Silence threshold (dB)", -60.0, -10.0, step=1.0, key="noise_db")
    st.slider("Min stilhed (sek)", 0.2, 2.0, step=0.1, key="min_silence_s")
    st.slider("Padding for/efter (sek)", 0.0, 1.0, step=0.05, key="pad_s")
    st.slider("Min kliplængde (sek)", 0.5, 10.0, step=0.1, key="min_segment_s")
    st.slider("Min ord for tale", 2, 30, step=1, key="voice_min_words")
    st.slider("Min lydstyrke for musik (dB)", -60.0, -10.0, step=1.0, key="music_min_db")

    st.subheader("Song: hooks")
    st.slider("Hook-længde (sek)", 2.0, 30.0, step=0.5, key="hook_len_range")
    st.slider("Foretrukket længde (sek)", 2.0, 20.0, step=0.5, key="prefer_len")
    st.slider("Scan-hop (sek)", 0.25, 5.0, step=0.25, key="hook_hop")
    st.slider("Antal hook-forslag", 3, 30, step=1, key="hook_topn")
    st.slider("Min afstand mellem hooks (sek)", 0.0, 10.0, step=0.5, key="hook_gap")
    st.checkbox("Transskriber hooks (langsommere)", key="whisper_on_hooks")

    st.subheader("Beat-grid")
    st.checkbox("Refine hooks/loops til beat-grid", key="beat_refine")
    st.number_input("Beats per bar", min_value=3, max_value=7, step=1, key="beats_per_bar")
    prefer_txt = "1 bar" if int(st.session_state["prefer_bars"]) == 1 else "2 bars"
    prefer_ui = st.radio(
        "Foretrækkes loop-længde",
        ["1 bar", "2 bars"],
        index=0 if prefer_txt == "1 bar" else 1,
    )
    st.session_state["prefer_bars"] = 1 if prefer_ui == "1 bar" else 2
    st.checkbox("Prøv både 1 og 2 bars", key="try_both_bars")


def normalize_bpm(bpm) -> int:
    try:
        return int(round(float(bpm)))
    except Exception:
        return 0


def safe_slug(text: str) -> str:
    if not text:
        return ""
    out = []
    for ch in text.strip():
        if ch.isalnum() or ch in "_æøåÆØÅ- ":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).replace(" ", "_").strip("_")


def build_stem(slug_text: str, type_label: str, bpm, fallback: str) -> str:
    slug = safe_slug(slug_text) if slug_text else ""
    if not slug:
        slug = fallback
    bpm_tag = f"{normalize_bpm(bpm)}bpm"
    return f"{slug}_{type_label}_{bpm_tag}"


def export_clip(
    in_path: Path,
    session_dir: Path,
    stem: str,
    start_s: float,
    end_s: float,
    want_format: str,
    min_duration: float,
) -> Path | None:
    dur = max(0.0, float(end_s) - float(start_s))
    if dur < float(min_duration):
        return None

    export_end = float(end_s) + TAIL_PAD
    wav_tmp = session_dir / f"{stem}__tmp.wav"
    cut_segment_to_wav(in_path, wav_tmp, start_s, export_end)

    if want_format.startswith("wav"):
        outp = session_dir / f"{stem}.wav"
        if outp.exists():
            outp.unlink()
        wav_tmp.rename(outp)
        return outp

    outp = session_dir / f"{stem}.mp3"
    cut_segment_to_mp3(in_path, outp, start_s, export_end, bitrate="192k")
    try:
        wav_tmp.unlink()
    except Exception:
        pass
    return outp


def maybe_refine_barloop(wav_for_analysis: Path, start_s: float, end_s: float):
    if not st.session_state["beat_refine"]:
        return start_s, end_s, 0.0, 0, 0.0, False, "disabled"

    beats_per_bar = int(st.session_state["beats_per_bar"])
    prefer_bars = int(st.session_state["prefer_bars"])
    try_both = bool(st.session_state["try_both_bars"])

    def _try_bars(bars: int):
        return refine_best_1_or_2_bars(
            str(wav_for_analysis),
            window_start=float(start_s),
            window_end=float(end_s),
            beats_per_bar=beats_per_bar,
            prefer_bars=bars,
            sr=22050,
        )

    rr = _try_bars(prefer_bars)
    if rr.ok and (rr.end - rr.start) >= MIN_EXPORT_DUR:
        return rr.start, rr.end, rr.bpm, rr.bars, rr.score, True, ""

    if try_both:
        for bars in (2, 4):
            if bars == prefer_bars:
                continue
            rr_alt = _try_bars(bars)
            if rr_alt.ok and (rr_alt.end - rr_alt.start) >= MIN_EXPORT_DUR:
                return rr_alt.start, rr_alt.end, rr_alt.bpm, rr_alt.bars, rr_alt.score, True, ""

    if rr.ok and (rr.end - rr.start) > 0.5:
        return rr.start, rr.end, rr.bpm, rr.bars, rr.score, False, "too short"

    return start_s, end_s, rr.bpm, rr.bars, rr.score, False, rr.reason


def classify_broadcast(text: str, mean_db: float) -> str:
    word_count = len((text or "").split())
    if word_count >= int(st.session_state["voice_min_words"]):
        return "Tale"
    if mean_db >= float(st.session_state["music_min_db"]):
        return "Musik/Jingle"
    return "Ukendt"


def calculate_overlap(s1: float, e1: float, s2: float, e2: float) -> float:
    overlap_start = max(s1, s2)
    overlap_end = min(e1, e2)
    overlap_len = max(0.0, overlap_end - overlap_start)
    if overlap_len == 0:
        return 0.0
    d1 = max(0.001, e1 - s1)
    d2 = max(0.001, e2 - s2)
    return overlap_len / min(d1, d2)


def process_song_hunter(
    in_path: Path,
    session_dir: Path,
    in_name: str,
    wav16: Path,
    lang: str | None,
    hooks: List,
    status=None,
):
    st.write("Finder loops (Song Hunter)...")
    hooks = sorted(hooks, key=lambda h: h.score, reverse=True)

    results = []
    kept_intervals: List[Tuple[float, float]] = []
    progress_bar = st.progress(0)

    for idx, h in enumerate(hooks, start=1):
        progress_bar.progress(idx / max(1, len(hooks)))
        start_s, end_s = float(h.start), float(h.end)

        r_start, r_end, bpm, bars, rscore, refined_ok, rreason = maybe_refine_barloop(
            wav16, start_s, end_s
        )
        use_start, use_end = (r_start, r_end) if refined_ok else (start_s, end_s)
        dur = max(0.0, use_end - use_start)
        raw_dur = max(0.0, end_s - start_s)

        if dur < MIN_EXPORT_DUR and raw_dur >= MIN_EXPORT_DUR:
            use_start, use_end = start_s, end_s
            dur = raw_dur
            refined_ok = False
            rreason = "fallback_raw"

        if dur < MIN_EXPORT_DUR:
            continue

        # Anti-overlap: keep highest-scored windows first, skip >30% overlaps.
        is_overlap = any(
            calculate_overlap(use_start, use_end, ka, kb) > 0.30
            for ka, kb in kept_intervals
        )
        if is_overlap:
            continue

        base_id = f"{idx:04d}_{use_start:.3f}_{use_end:.3f}"
        text = ""
        tjson = {"text": "", "segments": []}

        if st.session_state["whisper_on_hooks"]:
            wav_for_whisper = session_dir / f"{base_id}__whisper.wav"
            cut_segment_to_wav(in_path, wav_for_whisper, use_start, use_end)
            call_lang = lang if lang and lang != "auto" else "da"
            tjson = transcribe_wav(st.session_state.model, wav_for_whisper, language=call_lang)
            text = (tjson.get("text") or "").strip()

        slug_text = ""
        if st.session_state["use_slug"] and text:
            slug_text = " ".join(text.split()[: int(st.session_state["slug_words"])])

        stem = build_stem(slug_text, "loop", bpm, fallback=f"clip_{idx:04d}")
        if status is not None:
            status.update(label="Gemmer klip...", state="running")

        clip_path = export_clip(
            in_path,
            session_dir,
            stem,
            use_start,
            use_end,
            st.session_state["export_format"],
            min_duration=dur,
        )
        if clip_path is None:
            continue

        kept_intervals.append((use_start, use_end))

        txt_path = session_dir / f"{stem}.txt"
        json_path = session_dir / f"{stem}.json"
        txt_path.write_text((text or "") + "\n", encoding="utf-8")
        json_path.write_text(json.dumps(tjson, ensure_ascii=False, indent=2), encoding="utf-8")

        tags = ["musik", "loop"]
        tags += [f"{bars}bar"] if refined_ok else ["unrefined"]
        if text:
            tags = list(set(tags + auto_tags(text)))

        results.append(
            {
                "source": in_name,
                "group": "song",
                "clip": idx,
                "start_sec": use_start,
                "end_sec": use_end,
                "dur_sec": dur,
                "tags": ", ".join(tags),
                "hook_score": float(h.score),
                "energy": float(h.energy),
                "loopability": float(h.loopability),
                "stability": float(h.stability),
                "refined": bool(refined_ok),
                "refine_bpm": normalize_bpm(bpm),
                "refine_bars": int(bars or 0),
                "refine_score": float(rscore or 0.0),
                "refine_reason": rreason,
                "clip_path": str(clip_path),
                "filename": clip_path.name,
                "txt": str(txt_path),
                "json": str(json_path),
                "text": (text[:240] if text else ""),
            }
        )

    progress_bar.empty()
    return results


def process_broadcast(
    in_path: Path,
    session_dir: Path,
    in_name: str,
    lang: str | None,
    status=None,
):
    intervals = detect_non_silent_intervals(
        in_path,
        noise_db=st.session_state["noise_db"],
        min_silence_s=st.session_state["min_silence_s"],
        pad_s=st.session_state["pad_s"],
        min_segment_s=st.session_state["min_segment_s"],
    )

    results = []
    progress_bar = st.progress(0)

    for idx, (start_s, end_s) in enumerate(intervals, start=1):
        progress_bar.progress(idx / max(1, len(intervals)))
        start_s, end_s = float(start_s), float(end_s)
        dur = max(0.0, end_s - start_s)
        if dur < MIN_EXPORT_DUR:
            continue

        base_id = f"{idx:04d}_{start_s:.3f}_{end_s:.3f}"
        wav_for_whisper = session_dir / f"{base_id}__whisper.wav"
        cut_segment_to_wav(in_path, wav_for_whisper, start_s, end_s)

        t = transcribe_wav(st.session_state.model, wav_for_whisper, language=lang or "da")
        text = (t.get("text") or "").strip()
        mean_db = mean_volume_db(wav_for_whisper)
        category = classify_broadcast(text, mean_db)

        slug_text = ""
        if st.session_state["use_slug"] and text:
            slug_text = " ".join(text.split()[: int(st.session_state["slug_words"])])

        type_label = "tale" if category == "Tale" else "musik" if category == "Musik/Jingle" else "mix"
        stem = build_stem(slug_text, type_label, 0, fallback=f"clip_{idx:04d}")
        if status is not None:
            status.update(label="Gemmer klip...", state="running")

        clip_path = export_clip(
            in_path,
            session_dir,
            stem,
            start_s,
            end_s,
            st.session_state["export_format"],
            min_duration=dur,
        )
        if clip_path is None:
            continue

        txt_path = session_dir / f"{stem}.txt"
        json_path = session_dir / f"{stem}.json"
        txt_path.write_text(text + "\n", encoding="utf-8")
        json_path.write_text(json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8")

        tags = auto_tags(text)

        results.append(
            {
                "source": in_name,
                "group": "broadcast",
                "Category": category,
                "clip": idx,
                "start_sec": start_s,
                "end_sec": end_s,
                "dur_sec": dur,
                "tags": ", ".join(tags),
                "refine_bpm": 0,
                "clip_path": str(clip_path),
                "filename": clip_path.name,
                "txt": str(txt_path),
                "json": str(json_path),
                "text": text[:240] if text else "",
            }
        )

    progress_bar.empty()
    return results


# ----------------------------
# Input Tabs
# ----------------------------
input_paths = []

upload_tab, link_tab = st.tabs(["📂 Upload Filer", "🔗 Hent fra Link"])

with upload_tab:
    files = st.file_uploader("Upload", type=["mp3", "wav"], accept_multiple_files=True)
    if files:
        for uf in files:
            input_paths.append(("upload", uf.name, uf.getvalue()))

with link_tab:
    url = st.text_input(
        "Link",
        placeholder="https://... (YouTube / SoundCloud / etc.)",
        key="url_input",
    )
    dl_btn = st.button("Download", type="primary", disabled=not url.strip(), key="btn_download")
    st.caption("Gemmer i: output/Downloads/")
    if dl_btn:
        try:
            with st.spinner("Downloader..."):
                p = download_audio(url.strip(), OUTPUT_ROOT / "Downloads")
            st.success(f"Downloaded: {p.name}")
            st.session_state["downloaded_files"] = list(
                st.session_state.get("downloaded_files", [])
            ) + [str(p)]
        except DownloadError as e:
            st.error(str(e))
            st.markdown("Hent via Cobalt.tools: https://cobalt.tools")

for p in st.session_state.get("downloaded_files", []):
    input_paths.append(("path", p, None))


# ----------------------------
# Actions
# ----------------------------
if "model" not in st.session_state:
    st.session_state.model = None

colA, colB = st.columns([1, 1])
with colA:
    load_btn = st.button("Load Whisper-model", type="primary", key="btn_load_model")
with colB:
    run_btn = st.button("Process", type="primary", disabled=not input_paths, key="btn_process")

if load_btn:
    try:
        with st.spinner("Loader model... (første gang downloader den)"):
            st.session_state.model = load_model(
                model_size=st.session_state["model_size"],
                device=st.session_state["device"],
                compute_type=st.session_state["compute_type"],
            )
        st.success("Model loaded.")
    except Exception as e:
        st.error(f"Kunne ikke loade model: {e}")


# ----------------------------
# Processing
# ----------------------------
if run_btn:
    needs_whisper = st.session_state["mode"] == "broadcast" or st.session_state["whisper_on_hooks"]
    if needs_whisper and st.session_state.model is None:
        st.warning("Husk at klikke 'Load Whisper-model' først!")
        st.stop()

    results = []
    lang = None if st.session_state["language"] == "auto" else st.session_state["language"]

    with st.status("Starter op...", expanded=True) as status:
        total_files = len(input_paths)

        for idx_file, (src_type, name_or_path, maybe_bytes) in enumerate(input_paths):
            status.update(
                label=f"Behandler fil {idx_file + 1} af {total_files}: {name_or_path}",
                state="running",
            )
            st.write(f"### Fil {idx_file + 1}/{total_files}: {name_or_path}")

            if src_type == "upload":
                in_name = name_or_path
                session_dir = OUTPUT_ROOT / Path(in_name).stem
                session_dir.mkdir(parents=True, exist_ok=True)
                in_path = session_dir / in_name
                in_path.write_bytes(maybe_bytes)
            else:
                p = Path(name_or_path)
                in_name = p.name
                session_dir = OUTPUT_ROOT / p.stem
                session_dir.mkdir(parents=True, exist_ok=True)
                in_path = session_dir / in_name
                if not in_path.exists():
                    in_path.write_bytes(p.read_bytes())

            wav16 = session_dir / "_analysis_16k_mono.wav"
            status.update(label="Konverterer til analyseformat...", state="running")
            ffmpeg_to_wav16k_mono(in_path, wav16)

            status.update(label="Analyserer...", state="running")
            if st.session_state["mode"] == "song":
                st.write("Kører Chorus-aware...")
                chorus_results = find_chorus_candidates(
                    wav16,
                    window_len=st.session_state["prefer_len"],
                    hop_s=st.session_state["hook_hop"],
                    topn=max(4, int(st.session_state["hook_topn"]) // 2),
                    min_gap_s=st.session_state["hook_gap"],
                )
                if len(chorus_results) == 0:
                    st.write("Ingen omkvæd fundet - skifter til Hooks...")
                    hooks = find_hooks(
                        wav16,
                        hook_len_range=st.session_state["hook_len_range"],
                        prefer_len=st.session_state["prefer_len"],
                        hop_s=st.session_state["hook_hop"],
                        topn=st.session_state["hook_topn"],
                        min_gap_s=st.session_state["hook_gap"],
                    )
                else:
                    hooks = chorus_results

                results.extend(
                    process_song_hunter(
                        in_path, session_dir, in_name, wav16, lang, hooks=hooks, status=status
                    )
                )
            else:
                results.extend(process_broadcast(in_path, session_dir, in_name, lang, status=status))

        if not results:
            status.update(label="Færdig, men fandt ingen klip.", state="error", expanded=True)
            st.error("Modellen fandt ingen klip med de nuværende indstillinger.")
        else:
            status.update(
                label=f"Succes! Fandt {len(results)} klip.",
                state="complete",
                expanded=False,
            )
            st.success(f"Færdig! Scroll ned for at se dine {len(results)} klip.")
            st.session_state.results = results


# ----------------------------
# Results browser / export
# ----------------------------
if "results" in st.session_state and st.session_state.results:
    st.divider()
    st.subheader("Klip-browser")

    df = pd.DataFrame(st.session_state.results).copy()

    if "loopability" in df.columns and "stability" in df.columns:
        df = df.sort_values(by=["loopability", "stability"], ascending=[False, False])

    df["row_id"] = range(len(df))
    view_cols = ["row_id", "refine_bpm", "tags", "filename"]
    view_df = df[view_cols].copy()

    edited = st.data_editor(
        view_df,
        use_container_width=True,
        hide_index=True,
        key="editor_results",
        column_config={
            "refine_bpm": st.column_config.NumberColumn("BPM", format="%d"),
            "tags": st.column_config.TextColumn("Tags", width="large"),
            "filename": st.column_config.TextColumn("Filnavn", width="large"),
        },
        disabled=["filename", "row_id"],
    )

    edited_map = edited.set_index("row_id")
    df.loc[edited_map.index, "refine_bpm"] = edited_map["refine_bpm"]
    df.loc[edited_map.index, "tags"] = edited_map["tags"]

    options = list(df[["row_id", "filename"]].itertuples(index=False, name=None))
    selected_ids = st.multiselect(
        "Vælg klip til eksport",
        options=options,
        default=options,
        format_func=lambda x: x[1],
        key="select_clips",
    )
    selected_rows = {x[0] for x in selected_ids}
    selected = df[df["row_id"].isin(selected_rows)].copy()

    st.write(f"Valgt: {len(selected)} klip")

    st.subheader("Download valgte klip")
    for dl_idx, r in enumerate(selected.itertuples()):
        p = Path(r.clip_path)
        if not p.exists():
            continue
        mime = "audio/mpeg" if p.suffix.lower() == ".mp3" else "audio/wav"
        st.download_button(
            label=f"Download {p.name}",
            data=p.read_bytes(),
            file_name=p.name,
            mime=mime,
            key=f"dl_clip_{dl_idx}_{p.name}",
        )

    with st.expander("Preview af valgte (første 50)"):
        for pv_idx, r in enumerate(selected.head(50).itertuples()):
            p = Path(r.clip_path)
            st.write(f"{p.name} ({float(r.dur_sec):.2f}s) | tags={getattr(r, 'tags', '')}")
            if p.exists():
                st.audio(p.read_bytes())
            if getattr(r, "text", ""):
                st.write(r.text)

    if st.button("Eksporter ZIP (valgte)", type="primary", key="btn_export_zip"):
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest_selected.csv", selected.to_csv(index=False).encode("utf-8"))
            for _, r in selected.iterrows():
                src_stem = Path(r["source"]).stem
                for k in ["clip_path", "txt", "json"]:
                    if k in r and r[k]:
                        fp = Path(r[k])
                        if fp.exists():
                            z.write(fp, arcname=f"{src_stem}/{fp.name}")

        zip_buf.seek(0)
        st.download_button(
            label="Download ZIP",
            data=zip_buf,
            file_name="radio_clips_selected.zip",
            mime="application/zip",
            key="dl_zip_export",
        )
else:
    st.info("Upload eller hent en fil, load modellen, og kør Process.")


