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
    ffmpeg_to_wav16k_mono,
    find_hooks,
    find_chorus_windows,
    refine_loops_within_window,
)

from beat_refine import refine_best_1_or_2_bars


# ----------------------------
# Config / helpers
# ----------------------------
st.set_page_config(page_title="Radio Splitter + Whisper", layout="wide")

OUTPUT_ROOT = Path("output")
ensure_dir(OUTPUT_ROOT)

MODE_OPTIONS = [
    "Radio/tale (split på stilhed)",
    "Jingles/mikro (fast length 6–12s)",
    "Song: Find hooks/loops (4–15s)",
    "Song: Chorus-aware (30–45s → hooks/loops)",
]


DEFAULTS = {
    # UI/state
    "mode": MODE_OPTIONS[0],
    "mode_ui": MODE_OPTIONS[0],
    "_pending": None,

    # Whisper
    "model_size": "small",
    "language": "da",
    "device": "cpu",
    "compute_type": "int8",
    "whisper_on_hooks": False,

    # Radio split
    "noise_db": -35.0,
    "min_silence_s": 0.7,
    "pad_s": 0.15,
    "min_segment_s": 1.2,

    # Fixed split (jingles)
    "fixed_len": 8.0,

    # Hooks
    "hook_len_range": (4.0, 15.0),
    "prefer_len": 8.0,
    "hook_hop": 1.0,
    "hook_topn": 12,
    "hook_gap": 2.0,

    # Chorus-aware
    "chorus_len_range": (30.0, 45.0),
    "chorus_hop": 2.0,
    "chorus_topn": 4,
    "chorus_gap": 8.0,
    "loops_per_chorus": 10,

    # Beat refine (1–2 bars)
    "beat_refine": True,
    "beats_per_bar": 4,
    "prefer_bars": 1,         # 1 or 2
    "try_both_bars": True,    # choose best

    # Jingle filter
    "jingle_mode": False,
    "min_score": 0.0,

    # Naming/export
    "use_slug": True,
    "slug_words": 6,
    "export_format": "wav (16000 mono)",

    # Downloads
    "downloaded_files": [],
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_pending_updates_before_widgets():
    """
    Streamlit-regel:
    Du må IKKE ændre st.session_state.<widget_key> efter widget er oprettet.
    Derfor bruger vi _pending og anvender den KUN i starten (før widgets).
    """
    pending = st.session_state.get("_pending")
    if isinstance(pending, dict) and pending:
        for k, v in pending.items():
            st.session_state[k] = v
        st.session_state["_pending"] = None


apply_pending_updates_before_widgets()


def request_preset(preset: dict):
    """
    Sætter _pending (safe) og rerunner.
    _pending anvendes i starten af næste run (før widgets), så ingen session_state-crash.
    """
    st.session_state["_pending"] = preset
    st.rerun()


def language_value():
    return None if st.session_state["language"] == "auto" else st.session_state["language"]


def save_input_to_session_dir(src_type, name_or_path, maybe_bytes):
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


def export_clip(in_path: Path, session_dir: Path, stem: str, a: float, b: float, want_format: str) -> Path:
    wav_tmp = session_dir / f"{stem}__tmp.wav"
    cut_segment_to_wav(in_path, wav_tmp, a, b)

    if want_format.startswith("wav"):
        outp = session_dir / f"{stem}.wav"
        if outp.exists():
            outp.unlink()
        wav_tmp.rename(outp)
        return outp

    outp = session_dir / f"{stem}.mp3"
    cut_segment_to_mp3(in_path, outp, a, b, bitrate="192k")
    try:
        wav_tmp.unlink()
    except Exception:
        pass
    return outp


def maybe_refine_barloop(wav_for_analysis: Path, a: float, b: float):
    """
    Returnerer:
    (start, end, bpm, bars, refine_score, refined_ok, reason)
    """
    if not st.session_state["beat_refine"]:
        return a, b, 0.0, 0, 0.0, False, "disabled"

    beats_per_bar = int(st.session_state["beats_per_bar"])
    prefer_bars = int(st.session_state["prefer_bars"])
    try_both = bool(st.session_state["try_both_bars"])

    rr = refine_best_1_or_2_bars(
        str(wav_for_analysis),
        window_start=float(a),
        window_end=float(b),
        beats_per_bar=beats_per_bar,
        prefer_bars=prefer_bars,
        sr=22050,
    )

    if rr.ok and try_both:
        return rr.start, rr.end, rr.bpm, rr.bars, rr.score, True, ""

    # Hvis try_both er off: vi bruger stadig refine_best_1_or_2_bars for enkelhed,
    # men resultatet bliver ofte samme. (Godt nok i praksis.)
    if rr.ok and (rr.end - rr.start) > 0.5:
        return rr.start, rr.end, rr.bpm, rr.bars, rr.score, True, ""

    return a, b, rr.bpm, rr.bars, rr.score, False, rr.reason


# ----------------------------
# UI
# ----------------------------
st.title("Radio Splitter + Whisper (gratis, lokalt)")
st.caption("Upload/Link → split / hooks → (valgfri whisper) → vælg → eksportér ZIP")

# Sidebar
st.sidebar.header("Indstillinger")

mode_idx = MODE_OPTIONS.index(st.session_state["mode_ui"]) if st.session_state["mode_ui"] in MODE_OPTIONS else 0
st.sidebar.selectbox("Mode", MODE_OPTIONS, index=mode_idx, key="mode_ui")

# hold internal mode in sync (safe: we are not writing to mode_ui)
st.session_state["mode"] = st.session_state["mode_ui"]

st.sidebar.subheader("Whisper")
st.sidebar.selectbox("Whisper-model", ["tiny", "base", "small", "medium"], key="model_size")
st.sidebar.selectbox("Sprog", ["da", "en", "auto"], key="language")
st.sidebar.selectbox("Device", ["cpu"], key="device")
st.sidebar.selectbox("Compute type", ["int8", "float32"], key="compute_type")

st.sidebar.subheader("Navngivning")
st.sidebar.checkbox("Tilføj slug fra transskription", key="use_slug")
st.sidebar.slider("Slug ord (ca.)", 2, 12, key="slug_words")

st.sidebar.subheader("Eksport")
st.sidebar.selectbox("Klipformat", ["wav (16000 mono)", "mp3 (192k)"], key="export_format")

# Radio
st.sidebar.subheader("Radio/tale: split på stilhed")
st.sidebar.slider("Silence threshold (dB)", -60.0, -10.0, step=1.0, key="noise_db")
st.sidebar.slider("Min stilhed (sek)", 0.2, 2.0, step=0.1, key="min_silence_s")
st.sidebar.slider("Padding før/efter (sek)", 0.0, 1.0, step=0.05, key="pad_s")
st.sidebar.slider("Min kliplængde (sek)", 0.5, 10.0, step=0.1, key="min_segment_s")

# Fixed
st.sidebar.subheader("Jingles/mikro: fast length")
st.sidebar.slider("Klip-længde (sek)", 2.0, 60.0, step=1.0, key="fixed_len")

# Hooks
st.sidebar.subheader("Song hooks/loops")
st.sidebar.slider("Hook-længde (sek)", 2.0, 30.0, step=0.5, key="hook_len_range")
st.sidebar.slider("Foretrukket længde (sek)", 2.0, 20.0, step=0.5, key="prefer_len")
st.sidebar.slider("Scan-hop (sek)", 0.25, 5.0, step=0.25, key="hook_hop")
st.sidebar.slider("Antal hook-forslag", 3, 30, step=1, key="hook_topn")
st.sidebar.slider("Min afstand mellem hooks (sek)", 0.0, 10.0, step=0.5, key="hook_gap")
st.sidebar.checkbox("Transskriber hooks (langsommere)", key="whisper_on_hooks")

# Chorus-aware
st.sidebar.subheader("Chorus-aware")
st.sidebar.slider("Chorus-længde (sek)", 15.0, 90.0, step=1.0, key="chorus_len_range")
st.sidebar.slider("Chorus scan-hop (sek)", 1.0, 6.0, step=0.5, key="chorus_hop")
st.sidebar.slider("Antal chorus-vinduer", 1, 8, step=1, key="chorus_topn")
st.sidebar.slider("Min afstand mellem chorus (sek)", 0.0, 30.0, step=1.0, key="chorus_gap")
st.sidebar.slider("Loops pr chorus", 3, 20, step=1, key="loops_per_chorus")

# Beat refine
st.sidebar.subheader("Beat-grid: 1–2 bar perfekte loops")
st.sidebar.checkbox("Refine hooks/loops til beat-grid", key="beat_refine")
st.sidebar.number_input("Beats per bar (standard 4)", min_value=3, max_value=7, step=1, key="beats_per_bar")
prefer_txt = "1 bar" if int(st.session_state["prefer_bars"]) == 1 else "2 bars"
prefer_ui = st.sidebar.radio("Foretræk loop-længde", ["1 bar", "2 bars"], index=0 if prefer_txt == "1 bar" else 1)
st.session_state["prefer_bars"] = 1 if prefer_ui == "1 bar" else 2
st.sidebar.checkbox("Prøv både 1 og 2 bars (vælg bedste)", key="try_both_bars")
st.sidebar.caption(
    "Til musik: snapper loop start/slut til beats og giver præcis 1–2 bars loops.\n"
    "Hvis tempo er ustabilt (rubato), kan den falde tilbage til dit hook."
)

# Jingle filter
st.sidebar.subheader("Jingle-mode")
st.sidebar.checkbox("Find jingles (sortér + filter)", key="jingle_mode")
st.sidebar.slider("Min jingle score (filter)", 0.0, 10.0, step=0.25, key="min_score")

# Quick preset buttons (SAFE via request_preset)
st.sidebar.subheader("Presets (hurtigt)")
c1, c2 = st.sidebar.columns(2)

with c1:
    if st.button("Preset: Radio", use_container_width=True):
        request_preset({
            "mode_ui": "Radio/tale (split på stilhed)",
            "mode": "Radio/tale (split på stilhed)",
            "noise_db": -35.0,
            "min_silence_s": 0.7,
            "pad_s": 0.15,
            "min_segment_s": 1.2,
            "jingle_mode": False,
            "beat_refine": False,
            "whisper_on_hooks": False,
            "model_size": "small",
            "language": "da",
        })

with c2:
    if st.button("Preset: Chorus", use_container_width=True):
        request_preset({
            "mode_ui": "Song: Chorus-aware (30–45s → hooks/loops)",
            "mode": "Song: Chorus-aware (30–45s → hooks/loops)",
            "hook_len_range": (4.0, 15.0),
            "prefer_len": 8.0,
            "hook_hop": 1.0,
            "hook_topn": 12,
            "hook_gap": 2.0,
            "chorus_len_range": (30.0, 45.0),
            "chorus_hop": 2.0,
            "chorus_topn": 4,
            "chorus_gap": 8.0,
            "loops_per_chorus": 10,
            "beat_refine": True,
            "beats_per_bar": 4,
            "prefer_bars": 1,
            "try_both_bars": True,
            "whisper_on_hooks": False,
            "jingle_mode": False,
            "model_size": "small",
            "language": "da",
        })


# ----------------------------
# Input: link download
# ----------------------------
st.subheader("A) Indsæt link (Ctrl+V) og download lyd")
url = st.text_input("Link", placeholder="https://... (YouTube / Archive / etc.)")

dl_col1, dl_col2 = st.columns([1, 3])
with dl_col1:
    dl_btn = st.button("Download", type="primary", disabled=not url.strip())
with dl_col2:
    st.caption("Gemmer i: output/Downloads/")

if dl_btn:
    try:
        with st.spinner("Downloader..."):
            p = download_audio(url.strip(), OUTPUT_ROOT / "Downloads")
        st.success(f"Downloaded: {p.name}")
        st.session_state["downloaded_files"] = list(st.session_state.get("downloaded_files", [])) + [str(p)]
    except Exception as e:
        st.error(f"Download fejl: {e}")

# ----------------------------
# Input: upload
# ----------------------------
st.subheader("B) Upload MP3/WAV (du kan vælge flere)")
files = st.file_uploader("Upload", type=["mp3", "wav"], accept_multiple_files=True)

input_paths = []
if files:
    for uf in files:
        input_paths.append(("upload", uf.name, uf.getvalue()))
for p in st.session_state.get("downloaded_files", []):
    input_paths.append(("path", p, None))

# ----------------------------
# Actions
# ----------------------------
if "model" not in st.session_state:
    st.session_state.model = None

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    load_btn = st.button("Load Whisper-model", type="primary")
with colB:
    run_btn = st.button("Process", type="primary", disabled=not input_paths)
with colC:
    auto_btn = st.button("Analyze → auto-sæt preset", disabled=not input_paths)

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

if auto_btn and input_paths:
    st.info("Analyserer første fil og auto-sætter mode + sliders...")
    src_type, name_or_path, maybe_bytes = input_paths[0]
    in_path, session_dir, in_name = save_input_to_session_dir(src_type, name_or_path, maybe_bytes)

    # Heuristik: hvis den ser ud som "én lang non-silent" => musik
    intervals = detect_non_silent_intervals(in_path, noise_db=-35.0, min_silence_s=0.7, pad_s=0.0, min_segment_s=1.0)
    looks_like_song = (len(intervals) == 1 and (intervals[0][1] - intervals[0][0]) > 60.0)

    if looks_like_song:
        request_preset({
            "mode_ui": "Song: Chorus-aware (30–45s → hooks/loops)",
            "mode": "Song: Chorus-aware (30–45s → hooks/loops)",
            "beat_refine": True,
            "whisper_on_hooks": False,
            "jingle_mode": False,
            "model_size": "small",
            "language": "da",
        })
    else:
        request_preset({
            "mode_ui": "Radio/tale (split på stilhed)",
            "mode": "Radio/tale (split på stilhed)",
            "beat_refine": False,
            "whisper_on_hooks": True,
            "jingle_mode": False,
            "model_size": "small",
            "language": "da",
        })


# ----------------------------
# Processing
# ----------------------------
if run_btn:
    mode_now = st.session_state["mode"]

    needs_whisper = True
    if mode_now in ("Song: Find hooks/loops (4–15s)", "Song: Chorus-aware (30–45s → hooks/loops)"):
        if not st.session_state["whisper_on_hooks"]:
            needs_whisper = False

    if needs_whisper and st.session_state.model is None:
        st.warning("Klik først: Load Whisper-model.")
        st.stop()

    results = []
    lang = language_value()

    for src_type, name_or_path, maybe_bytes in input_paths:
        in_path, session_dir, in_name = save_input_to_session_dir(src_type, name_or_path, maybe_bytes)
        st.write(f"### Processing: {in_name}")

        wav16 = session_dir / "_analysis_16k_mono.wav"
        ffmpeg_to_wav16k_mono(in_path, wav16)

        # ---------------- Song: hooks only ----------------
        if mode_now == "Song: Find hooks/loops (4–15s)":
            hooks = find_hooks(
                wav16,
                hook_len_range=st.session_state["hook_len_range"],
                prefer_len=st.session_state["prefer_len"],
                hop_s=st.session_state["hook_hop"],
                topn=st.session_state["hook_topn"],
                min_gap_s=st.session_state["hook_gap"],
            )
            st.info(f"Fandt {len(hooks)} hook/loop-forslag (beat_refine={st.session_state['beat_refine']})")

            for idx, h in enumerate(hooks, start=1):
                a, b = float(h.start), float(h.end)

                a2, b2, bpm, bars, rscore, refined_ok, rreason = maybe_refine_barloop(wav16, a, b)
                aa, bb = (a2, b2) if refined_ok else (a, b)
                dur = max(0.0, bb - aa)

                base = f"{idx:04d}_{hhmmss_ms(aa)}_to_{hhmmss_ms(bb)}"
                text = ""
                tjson = {"text": "", "segments": []}

                if st.session_state["whisper_on_hooks"]:
                    wav_for_whisper = session_dir / f"{base}__whisper.wav"
                    cut_segment_to_wav(in_path, wav_for_whisper, aa, bb)
                    tjson = transcribe_wav(st.session_state.model, wav_for_whisper, language=lang or "da")
                    text = (tjson.get("text") or "").strip()

                slug = ""
                if st.session_state["use_slug"] and text:
                    slug = safe_slug(" ".join(text.split()[: int(st.session_state["slug_words"]) ]))

                stem = f"{base}__{slug}" if slug else base
                clip_path = export_clip(in_path, session_dir, stem, aa, bb, st.session_state["export_format"])

                txt_path = session_dir / f"{stem}.txt"
                json_path = session_dir / f"{stem}.json"
                txt_path.write_text((text or "") + "\n", encoding="utf-8")
                json_path.write_text(json.dumps(tjson, ensure_ascii=False, indent=2), encoding="utf-8")

                tags = ["musik", "hook"]
                tags += [f"{bars}bar"] if refined_ok else ["unrefined"]
                if text:
                    tags = list(set(tags + auto_tags(text)))

                results.append({
                    "source": in_name,
                    "pick": True,
                    "group": "hooks",
                    "clip": idx,
                    "start_sec": aa,
                    "end_sec": bb,
                    "dur_sec": dur,
                    "tags": ", ".join(tags),
                    "hook_score": float(h.score),
                    "energy": float(h.energy),
                    "loopability": float(h.loopability),
                    "stability": float(h.stability),
                    "refined": bool(refined_ok),
                    "refine_bpm": float(bpm or 0.0),
                    "refine_bars": int(bars or 0),
                    "refine_score": float(rscore or 0.0),
                    "refine_reason": rreason,
                    "jingle_score": 0.0,
                    "clip_path": str(clip_path),
                    "txt": str(txt_path),
                    "json": str(json_path),
                    "text": (text[:240] if text else "")
                })

            st.session_state.results = results
            continue

        # ---------------- Song: chorus-aware ----------------
        if mode_now == "Song: Chorus-aware (30–45s → hooks/loops)":
            chorus_windows = find_chorus_windows(
                wav16,
                chorus_len_range=st.session_state["chorus_len_range"],
                hop_s=st.session_state["chorus_hop"],
                topn=st.session_state["chorus_topn"],
                min_gap_s=st.session_state["chorus_gap"],
            )

            st.info(f"Fandt {len(chorus_windows)} chorus-vinduer. Udtrækker hooks/loops inde i dem...")

            out_idx = 0
            for c_idx, cw in enumerate(chorus_windows, start=1):
                ca, cb = float(cw.start), float(cw.end)

                # (valgfrit) gem chorus-vinduet som reference (ikke auto-picked)
                c_base = f"C{c_idx:02d}_{hhmmss_ms(ca)}_to_{hhmmss_ms(cb)}"
                c_stem = f"{c_base}__CHORUS"
                chorus_path = export_clip(in_path, session_dir, c_stem, ca, cb, st.session_state["export_format"])

                results.append({
                    "source": in_name,
                    "pick": False,
                    "group": "chorus",
                    "clip": c_idx,
                    "start_sec": ca,
                    "end_sec": cb,
                    "dur_sec": max(0.0, cb - ca),
                    "tags": "chorus",
                    "hook_score": float(cw.score),
                    "energy": float(cw.energy),
                    "loopability": float(cw.loopability),
                    "stability": float(cw.stability),
                    "refined": False,
                    "refine_bpm": 0.0,
                    "refine_bars": 0,
                    "refine_score": 0.0,
                    "refine_reason": "",
                    "jingle_score": 0.0,
                    "clip_path": str(chorus_path),
                    "txt": "",
                    "json": "",
                    "text": ""
                })

                loops = refine_loops_within_window(
                    wav16,
                    cw,
                    hook_len_range=st.session_state["hook_len_range"],
                    prefer_len=st.session_state["prefer_len"],
                    hop_s=max(0.25, float(st.session_state["hook_hop"]) / 2.0),
                    topn=int(st.session_state["loops_per_chorus"]),
                    min_gap_s=max(0.5, float(st.session_state["hook_gap"]) / 2.0),
                )

                for l_idx, h in enumerate(loops, start=1):
                    out_idx += 1
                    a, b = float(h.start), float(h.end)

                    a2, b2, bpm, bars, rscore, refined_ok, rreason = maybe_refine_barloop(wav16, a, b)
                    aa, bb = (a2, b2) if refined_ok else (a, b)
                    dur = max(0.0, bb - aa)

                    base = f"{out_idx:04d}_C{c_idx:02d}L{l_idx:02d}_{hhmmss_ms(aa)}_to_{hhmmss_ms(bb)}"
                    text = ""
                    tjson = {"text": "", "segments": []}

                    if st.session_state["whisper_on_hooks"]:
                        wav_for_whisper = session_dir / f"{base}__whisper.wav"
                        cut_segment_to_wav(in_path, wav_for_whisper, aa, bb)
                        tjson = transcribe_wav(st.session_state.model, wav_for_whisper, language=lang or "da")
                        text = (tjson.get("text") or "").strip()

                    slug = ""
                    if st.session_state["use_slug"] and text:
                        slug = safe_slug(" ".join(text.split()[: int(st.session_state["slug_words"]) ]))

                    stem = f"{base}__{slug}" if slug else base
                    clip_path = export_clip(in_path, session_dir, stem, aa, bb, st.session_state["export_format"])

                    txt_path = session_dir / f"{stem}.txt"
                    json_path = session_dir / f"{stem}.json"
                    txt_path.write_text((text or "") + "\n", encoding="utf-8")
                    json_path.write_text(json.dumps(tjson, ensure_ascii=False, indent=2), encoding="utf-8")

                    tags = ["musik", "chorus-aware", "loop"]
                    tags += [f"{bars}bar"] if refined_ok else ["unrefined"]
                    if text:
                        tags = list(set(tags + auto_tags(text)))

                    results.append({
                        "source": in_name,
                        "pick": True,
                        "group": f"chorus_{c_idx:02d}",
                        "clip": out_idx,
                        "start_sec": aa,
                        "end_sec": bb,
                        "dur_sec": dur,
                        "tags": ", ".join(tags),
                        "hook_score": float(h.score),
                        "energy": float(h.energy),
                        "loopability": float(h.loopability),
                        "stability": float(h.stability),
                        "refined": bool(refined_ok),
                        "refine_bpm": float(bpm or 0.0),
                        "refine_bars": int(bars or 0),
                        "refine_score": float(rscore or 0.0),
                        "refine_reason": rreason,
                        "jingle_score": 0.0,
                        "clip_path": str(clip_path),
                        "txt": str(txt_path),
                        "json": str(json_path),
                        "text": (text[:240] if text else "")
                    })

            st.session_state.results = results
            continue

        # ---------------- Radio / jingles ----------------
        if mode_now == "Radio/tale (split på stilhed)":
            intervals = detect_non_silent_intervals(
                in_path,
                noise_db=st.session_state["noise_db"],
                min_silence_s=st.session_state["min_silence_s"],
                pad_s=st.session_state["pad_s"],
                min_segment_s=st.session_state["min_segment_s"],
            )
        else:
            intervals = fixed_length_intervals(in_path, segment_len=st.session_state["fixed_len"])

        st.info(f"Fundet {len(intervals)} klip")
        for idx, (a, b) in enumerate(intervals, start=1):
            a, b = float(a), float(b)
            dur = max(0.0, b - a)
            base = f"{idx:04d}_{hhmmss_ms(a)}_to_{hhmmss_ms(b)}"

            wav_for_whisper = session_dir / f"{base}__whisper.wav"
            cut_segment_to_wav(in_path, wav_for_whisper, a, b)

            t = transcribe_wav(st.session_state.model, wav_for_whisper, language=lang or "da")
            text = (t.get("text") or "").strip()

            slug = ""
            if st.session_state["use_slug"] and text:
                slug = safe_slug(" ".join(text.split()[: int(st.session_state["slug_words"]) ]))

            stem = f"{base}__{slug}" if slug else base
            clip_path = export_clip(in_path, session_dir, stem, a, b, st.session_state["export_format"])

            txt_path = session_dir / f"{stem}.txt"
            json_path = session_dir / f"{stem}.json"
            txt_path.write_text(text + "\n", encoding="utf-8")
            json_path.write_text(json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8")

            tags = auto_tags(text)
            score = float(jingle_score(text, dur))

            results.append({
                "source": in_name,
                "pick": True,
                "group": "radio",
                "clip": idx,
                "start_sec": a,
                "end_sec": b,
                "dur_sec": dur,
                "tags": ", ".join(tags),
                "jingle_score": score,
                "clip_path": str(clip_path),
                "txt": str(txt_path),
                "json": str(json_path),
                "text": text[:240] if text else ""
            })

    st.session_state.results = results


# ----------------------------
# Results browser / export
# ----------------------------
if "results" in st.session_state and st.session_state.results:
    st.divider()
    st.subheader("C) Klip-browser (vælg de gode)")

    df = pd.DataFrame(st.session_state.results)
    if "pick" not in df.columns:
        df["pick"] = True

    # Optional filter for jingles
    if st.session_state["jingle_mode"] and "jingle_score" in df.columns:
        df = df[df["jingle_score"] >= float(st.session_state["min_score"])]

    edited = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "pick": st.column_config.CheckboxColumn("Gem", default=True),
            "group": st.column_config.TextColumn("Gruppe"),
            "refined": st.column_config.CheckboxColumn("Beat-refined"),
            "refine_bpm": st.column_config.NumberColumn("BPM", format="%.1f"),
            "refine_bars": st.column_config.NumberColumn("Bars", format="%d"),
            "refine_score": st.column_config.NumberColumn("Refine score", format="%.2f"),
            "hook_score": st.column_config.NumberColumn("Hook score", format="%.2f"),
            "jingle_score": st.column_config.NumberColumn("Jingle score", format="%.2f"),
            "text": st.column_config.TextColumn("Transcript (kort)", width="large"),
            "clip_path": st.column_config.TextColumn("Fil", width="large"),
        }
    )

    selected = edited[edited["pick"] == True].copy()
    st.write(f"**Valgt:** {len(selected)} klip")

    with st.expander("Preview af valgte (første 50)"):
        for _, r in selected.head(50).iterrows():
            p = Path(r["clip_path"])
            st.write(f"**{p.name}**  ({float(r['dur_sec']):.2f}s) | group={r.get('group','')} | tags={r.get('tags','')}")
            if p.exists():
                st.audio(p.read_bytes())
            if r.get("text"):
                st.write(r["text"])

    if st.button("Eksportér ZIP (valgte)", type="primary"):
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
            mime="application/zip"
        )
else:
    st.info("Upload eller download en fil, load model (hvis nødvendigt), og kør Process.")
