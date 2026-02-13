from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from audio_split import (
    cut_segment_to_mp3,
    detect_non_silent_intervals,
    get_duration_seconds,
)
from beat_refine import refine_best_1_or_2_bars
from downloaders import download_audio
from hook_finder import ffmpeg_to_wav16k_mono, find_hooks
from transcribe import load_model, transcribe_wav


OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGE_OPTIONS = {
    "Auto": None,
    "Dansk": "da",
    "Engelsk": "en",
}


@dataclass
class ClipResult:
    filename: str
    path: Path
    start: float
    end: float
    score: float
    bpm: int
    tags: list[str]
    transcript: str


def ensure_session_defaults() -> None:
    defaults: dict[str, Any] = {
        "mode": "🎵 Song Hunter (Loops)",
        "language_ui": "Auto",
        "whisper_model": "small",
        "transcribe": True,
        "url_input": "",
        "last_source": None,
        "results": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def run_transcription(audio_path: Path, lang_ui: str, model_size: str) -> dict[str, Any]:
    model = load_model(model_size=model_size, device="cpu", compute_type="int8")
    return transcribe_wav(model, audio_path, language=LANGUAGE_OPTIONS[lang_ui])


def theme_tags(text: str) -> list[str]:
    t = (text or "").lower()
    themes = {
        "THEME:TIME": ["tid", "evighed", "nu", "time", "eternity", "now"],
        "THEME:MEMORY": ["huske", "glemme", "remember", "forget", "back"],
        "THEME:DREAM": ["drøm", "natten", "dream", "night", "sleep"],
        "THEME:EXISTENTIAL": ["livet", "verden", "cirkel", "life", "world", "circle"],
        "THEME:META": ["radio", "musik", "stemme", "lyd", "music", "voice", "sound"],
    }
    out = [name for name, words in themes.items() if any(word in t for word in words)]
    return out or ["THEME:UNKNOWN"]


def overlap_ratio(a: tuple[float, float], b: tuple[float, float]) -> float:
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    if right <= left:
        return 0.0
    intersection = right - left
    shortest = min(a[1] - a[0], b[1] - b[0])
    if shortest <= 0:
        return 0.0
    return intersection / shortest


def anti_overlap_keep_best(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda x: float(x["score"]), reverse=True):
        rng = (float(item["start"]), float(item["end"]))
        if any(overlap_ratio(rng, (float(k["start"]), float(k["end"]))) > 0.30 for k in kept):
            continue
        kept.append(item)
    return kept


def save_clip_with_tail(src_path: Path, out_dir: Path, stem: str, start: float, end: float) -> Path:
    final_end = end + 0.75  # Decay pad
    out_path = out_dir / f"{stem}_tail.mp3"
    cut_segment_to_mp3(src_path, out_path, start, final_end, bitrate="192k")
    return out_path


def find_song_candidates(wav16: Path) -> list[dict[str, Any]]:
    hook_candidates = [
        {"start": h.start, "end": h.end, "score": h.score, "bpm": h.bpm, "source": "hooks"}
        for h in find_hooks(wav16, hook_len_range=(4.0, 15.0), prefer_len=8.0, hop_s=1.0, topn=30, min_gap_s=0.2)
    ]

    # Chorus first. If empty/fails, fallback to hooks (already gathered).
    chorus_candidates = [c for c in hook_candidates if (c["end"] - c["start"]) >= 12.0]
    chosen = chorus_candidates if chorus_candidates else hook_candidates

    # 4-second filter
    chosen = [c for c in chosen if (float(c["end"]) - float(c["start"])) >= 4.0]
    return anti_overlap_keep_best(chosen)[:12]


def find_broadcast_candidates(src: Path) -> list[dict[str, Any]]:
    intervals = detect_non_silent_intervals(
        src,
        noise_db=-35.0,
        min_silence_s=0.7,
        pad_s=0.15,
        min_segment_s=1.2,
    )
    out = []
    for start, end in intervals:
        duration = end - start
        if duration < 4.0:
            continue
        out.append({"start": start, "end": end, "score": min(1.0, duration / 20.0), "bpm": 0.0, "source": "broadcast"})
    return anti_overlap_keep_best(out)[:12]


def process_source(src_path: Path, mode: str, lang_ui: str, whisper_model: str, do_transcribe: bool) -> list[ClipResult]:
    run_dir = OUTPUT_DIR / src_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    wav16 = run_dir / "analysis.wav"
    ffmpeg_to_wav16k_mono(src_path, wav16)

    candidates = find_song_candidates(wav16) if mode.startswith("🎵") else find_broadcast_candidates(src_path)

    results: list[ClipResult] = []
    for idx, c in enumerate(candidates, start=1):
        start = float(c["start"])
        end = float(c["end"])

        if mode.startswith("🎵"):
            rr = refine_best_1_or_2_bars(
                str(wav16),
                window_start=start,
                window_end=end,
                beats_per_bar=4,
                prefer_bars=1,
                sr=22050,
            )
            if rr.ok and rr.end > rr.start:
                start = start + float(rr.start)
                end = start + (float(rr.end) - float(rr.start))
                c["bpm"] = rr.bpm

        if (end - start) < 4.0:
            continue

        stem = f"{src_path.stem}_{idx:02d}"
        clip_path = save_clip_with_tail(src_path, run_dir, stem, start, end)

        transcript = ""
        if do_transcribe:
            t = run_transcription(clip_path, lang_ui, whisper_model)
            transcript = t.get("text", "")
            (run_dir / f"{stem}.json").write_text(json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8")

        tags = theme_tags(transcript)
        bpm_int = int(round(float(c.get("bpm") or 0.0)))
        results.append(
            ClipResult(
                filename=clip_path.name,
                path=clip_path,
                start=start,
                end=end + 0.75,
                score=float(c["score"]),
                bpm=bpm_int,
                tags=tags,
                transcript=transcript,
            )
        )

    manifest = [r.__dict__ | {"path": str(r.path)} for r in results]
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def load_input_from_tabs(tab_upload, tab_link) -> Path | None:
    source_path: Path | None = None

    with tab_upload:
        up = st.file_uploader("Upload audio/video", type=["mp3", "wav", "m4a", "mp4", "mov"])
        if up is not None:
            source_path = OUTPUT_DIR / up.name
            source_path.write_bytes(up.getvalue())
            st.session_state["last_source"] = str(source_path)

    with tab_link:
        st.text_input("YouTube/Sound URL", key="url_input")
        if st.button("Hent fra link / Download from link") and st.session_state["url_input"].strip():
            source_path = download_audio(st.session_state["url_input"].strip(), OUTPUT_DIR)
            st.session_state["last_source"] = str(source_path)
            st.success(f"Downloaded: {source_path.name}")

    if source_path is None and st.session_state.get("last_source"):
        remembered = Path(st.session_state["last_source"])
        if remembered.exists():
            source_path = remembered
    return source_path


def main() -> None:
    st.set_page_config(page_title="The Sample Machine", layout="wide")
    ensure_session_defaults()

    st.title("The Sample Machine")
    st.caption("Avalanches-style sample workflow · Dansk + English")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.selectbox("Mode", ["🎵 Song Hunter (Loops)", "📻 Broadcast Hunter (Mix)"], key="mode")
    with col2:
        st.selectbox("Whisper language", list(LANGUAGE_OPTIONS.keys()), key="language_ui")
    with col3:
        st.selectbox("Whisper model", ["tiny", "base", "small"], key="whisper_model")

    st.checkbox("Transcribe clips / Transskriber klip", key="transcribe")

    tab_upload, tab_link = st.tabs(["📂 Upload Filer", "🔗 Hent fra Link"])
    source_path = load_input_from_tabs(tab_upload, tab_link)

    if source_path:
        st.info(f"Source: {source_path.name} · {get_duration_seconds(source_path):.1f}s")

    if st.button("Start reconstruction"):
        if not source_path:
            st.error("Upload a file or fetch from link first.")
            return
        with st.spinner("Processing…"):
            st.session_state["results"] = process_source(
                source_path,
                st.session_state["mode"],
                st.session_state["language_ui"],
                st.session_state["whisper_model"],
                bool(st.session_state["transcribe"]),
            )

    results: list[ClipResult] = st.session_state.get("results", [])
    if not results:
        return

    st.subheader("Preview")
    for idx, res in enumerate(results):
        st.markdown(f"**{res.filename}** · BPM: `{res.bpm}` · Tags: `{', '.join(res.tags)}`")
        st.audio(str(res.path))
        st.download_button(
            label=f"Download {res.filename}",
            data=res.path.read_bytes(),
            file_name=res.filename,
            mime="audio/mpeg",
            key=f"dl_{idx}_{res.filename}",
        )


if __name__ == "__main__":
    main()
