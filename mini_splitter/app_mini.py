import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from time import perf_counter
import sys

import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ms_audio_utils import ensure_ffmpeg, save_uploaded_file, probe_audio, to_wav, read_bytes
from ms_transcribe import transcribe_audio
from ms_clipper import export_clips_from_segments
from ms_loops import export_bar_loops
from downloaders import download_audio, DownloadError

RUNTIME_ROOT = Path(
    os.getenv(
        "RADIO_SPLITTER_RUNTIME_ROOT",
        str(Path.home() / "AppData" / "Local" / "radio_splitter2"),
    )
).resolve()
OUTPUT_ROOT = Path(
    os.getenv("RADIO_SPLITTER_MINI_OUTPUT_ROOT", str(RUNTIME_ROOT / "mini_output"))
).resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

_runtime_temp = RUNTIME_ROOT / "temp"
_runtime_temp.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMP", str(_runtime_temp))
os.environ.setdefault("TEMP", str(_runtime_temp))
os.environ.setdefault("TMPDIR", str(_runtime_temp))

st.set_page_config(page_title="Mini Splitter+", layout="wide")
st.title("Mini Splitter+ (Focused Samples)")

ensure_ffmpeg()

# --- Styling: make downloads more obvious ---
st.markdown(
    """
<style>
/* Bigger, clearer download buttons */
div[data-testid="stDownloadButton"] > button {
  font-size: 1.05rem;
  padding: 0.7rem 1.0rem;
  border-radius: 12px;
  width: 100%;
}
/* Emphasize the main zip download */
.mainzip div[data-testid="stDownloadButton"] > button {
  font-size: 1.15rem !important;
  padding: 0.85rem 1.0rem !important;
  font-weight: 700 !important;
}
/* Make section headers a bit tighter */
h2, h3 { margin-top: 0.6rem; }
</style>
""",
    unsafe_allow_html=True,
)

@dataclass
class JobOutputs:
    job_id: str
    out_dir: Path
    src_path: Path
    wav_speech: Path
    wav_music: Path
    transcript_txt: Optional[Path] = None
    transcript_srt: Optional[Path] = None
    transcript_json: Optional[Path] = None
    transcript_words_json: Optional[Path] = None
    clips_dir: Optional[Path] = None
    loops_dir: Optional[Path] = None
    focus_report_json: Optional[Path] = None
    zip_path: Optional[Path] = None

def _zip_folder_to_file(folder: Path, zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(folder))
    return zip_path

def _list_audio_files(folder: Path) -> List[Path]:
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])

def _safe_rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except Exception:
        return p.name


def _find_recent_runs(root: Path, limit: int = 20) -> List[Path]:
    if not root.exists():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[:limit]


def _build_job_from_output_dir(out_dir: Path) -> Optional["JobOutputs"]:
    if not out_dir.exists():
        return None

    src_candidates = [p for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"}]
    src_path = src_candidates[0] if src_candidates else (out_dir / "source_unknown.wav")
    wav_speech = out_dir / "input_speech.wav"
    wav_music = out_dir / "input_music.wav"
    zip_path = out_dir / f"{out_dir.name}_mini_splitter.zip"
    if not zip_path.exists():
        try:
            _zip_folder_to_file(out_dir, zip_path)
        except Exception:
            pass

    job = JobOutputs(
        job_id=out_dir.name,
        out_dir=out_dir,
        src_path=src_path,
        wav_speech=wav_speech,
        wav_music=wav_music,
    )
    if zip_path.exists():
        job.zip_path = zip_path

    t_txt = out_dir / "transcript.txt"
    t_srt = out_dir / "transcript.srt"
    t_json = out_dir / "transcript.json"
    t_words = out_dir / "transcript_words.json"
    if t_txt.exists():
        job.transcript_txt = t_txt
    if t_srt.exists():
        job.transcript_srt = t_srt
    if t_json.exists():
        job.transcript_json = t_json
    if t_words.exists():
        job.transcript_words_json = t_words

    clips_dir = out_dir / "clips"
    loops_dir = out_dir / "loops"
    focus_report = out_dir / "clip_focus_report.json"
    if clips_dir.exists():
        job.clips_dir = clips_dir
    if loops_dir.exists():
        job.loops_dir = loops_dir
    if focus_report.exists():
        job.focus_report_json = focus_report

    return job


def _format_clock(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _profile_config(profile: str) -> Tuple[float, float]:
    if profile == "Punchy":
        return 1.8, 0.35
    if profile == "Longform":
        return 3.8, 0.55
    return 2.8, 0.45


def _segment_score(seg: Dict[str, Any], target_duration: float, word_weight: float) -> float:
    start = float(seg.get("start", 0.0) or 0.0)
    end = float(seg.get("end", 0.0) or 0.0)
    dur = max(0.0, end - start)
    text = (seg.get("text") or "").strip()
    words = len([w for w in text.split() if w.strip()])
    dur_fit = 1.0 - min(abs(dur - target_duration) / max(target_duration, 1e-6), 1.0)
    word_score = min(words / 12.0, 1.0)
    punct_bonus = 0.08 if any(ch in text for ch in "!?") else 0.0
    return max(0.0, 0.62 * dur_fit + word_weight * word_score + punct_bonus)


def _overlap_or_too_close(a: Dict[str, Any], b: Dict[str, Any], min_gap: float) -> bool:
    a_start, a_end = float(a["start"]), float(a["end"])
    b_start, b_end = float(b["start"]), float(b["end"])
    return not (a_end + min_gap <= b_start or a_start >= b_end + min_gap)


def _select_focused_segments(
    segments: List[Dict[str, Any]],
    max_items: int,
    min_gap: float,
    target_duration: float,
    word_weight: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    enriched: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments, start=1):
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", 0.0) or 0.0)
        if end <= start:
            continue
        scored = dict(seg)
        scored["score"] = round(_segment_score(seg, target_duration, word_weight), 4)
        scored["idx"] = i
        enriched.append(scored)

    ranked = sorted(
        enriched,
        key=lambda s: (
            float(s.get("score", 0.0)),
            float(s.get("end", 0.0)) - float(s.get("start", 0.0)),
        ),
        reverse=True,
    )

    selected: List[Dict[str, Any]] = []
    for cand in ranked:
        if any(_overlap_or_too_close(cand, keep, min_gap=min_gap) for keep in selected):
            continue
        selected.append(cand)
        if len(selected) >= max_items:
            break

    selected = sorted(selected, key=lambda s: float(s.get("start", 0.0)))
    return selected, ranked

# Persist last run results so downloads don't "disappear"
if "job" not in st.session_state:
    st.session_state.job = None

with st.sidebar:
    st.header("Settings")

    input_mode = st.radio("Inputkilde", ["Upload fil", "YouTube/URL"], index=0)

    mode = st.selectbox("Mode", ["All", "Transcribe", "Clips (Speech)", "Loops (Music)"])
    language = st.selectbox("Language", ["auto", "da", "en"])

    st.subheader("ASR backend")
    backend = st.selectbox(
        "Transcription engine",
        ["Fast (faster-whisper)", "Precise (WhisperX)"],
        index=0,
        help="WhisperX giver bedre word-level timings. Hvis WhisperX ikke er installeret, falder appen tilbage til faster-whisper.",
    )

    model_size = st.selectbox("Model", ["tiny", "base", "small"], index=2)
    beam_size = st.slider("Beam size", 1, 10, 5)

    st.divider()
    st.subheader("Focused Sample Selection")
    sample_mode = st.selectbox("Clip selection", ["Focused", "All segments"], index=0)
    focus_profile = st.selectbox("Focus profile", ["Balanced", "Punchy", "Longform"], index=0)
    max_clip_samples = st.slider("Max clips", 3, 80, 18, 1)
    min_gap_focus = st.slider("Min gap between clips (sec)", 0.0, 4.0, 0.75, 0.05)

    st.divider()
    st.subheader("Clips (speech)")
    min_clip_sec = st.slider("Min clip length (sec)", 0.5, 10.0, 2.0, 0.5)
    pad_sec = st.slider("Pad around clip (sec)", 0.0, 2.0, 0.25, 0.05)
    fade_ms = st.slider("Fade (ms)", 0, 300, 30, 10)

    st.divider()
    st.subheader("Loops (music)")
    bars = st.slider("Bars (4/4)", 4, 16, 8, 1)
    prefer_beats = st.checkbox("Snap to detected beats", value=True)
    loop_hop = st.selectbox("Loop hop", ["every beat", "every bar"], index=1)
    top_n_loops = st.slider("Top N loops", 5, 100, 20, 5)

    st.divider()
    export_format = st.selectbox("Export format", ["wav", "mp3"], index=0)

st.write(
    "Upload en lydfil eller indsæt en URL. Focused mode giver et strammere udvalg af samples med score og anti-overlap."
)

with st.expander("Recovery / Load previous run", expanded=False):
    recent = _find_recent_runs(OUTPUT_ROOT, limit=20)
    labels = [f"{p.name}  ({datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})" for p in recent]
    if labels:
        choice = st.selectbox("Vælg tidligere output-mappe", labels, index=0)
        idx = labels.index(choice)
        chosen_dir = recent[idx]
        if st.button("Load selected run"):
            loaded = _build_job_from_output_dir(chosen_dir)
            if loaded:
                st.session_state.job = loaded
                st.success(f"Loaded run: {chosen_dir.name}")
                st.rerun()
            else:
                st.error("Kunne ikke indlæse valgt run.")
    else:
        st.caption("Ingen tidligere runs fundet endnu.")

uploaded = st.file_uploader("Audio file", type=["mp3", "wav", "m4a", "flac", "aac", "ogg"], disabled=(input_mode != "Upload fil"))
url_input = st.text_input("YouTube / online URL", "", disabled=(input_mode != "YouTube/URL"))
run = st.button("Run", type="primary")

def run_pipeline() -> JobOutputs:
    want_transcribe = mode in ["All", "Transcribe", "Clips (Speech)"]
    want_clips = mode in ["All", "Clips (Speech)"]
    want_loops = mode in ["All", "Loops (Music)"]
    needs_transcribe = want_transcribe or want_clips

    total_steps = 3 + int(needs_transcribe) + int(want_clips) + int(want_loops)
    if input_mode == "YouTube/URL":
        total_steps += 1

    started_at = perf_counter()
    progress = st.progress(0.0, text="  0% | Starter pipeline...")
    step_state = {"done": 0}

    def tick(label: str) -> None:
        step_state["done"] += 1
        frac = min(max(step_state["done"] / max(total_steps, 1), 0.0), 1.0)
        elapsed = max(0.0, perf_counter() - started_at)
        eta = (elapsed / frac - elapsed) if frac > 1e-6 else 0.0
        progress.progress(
            frac,
            text=f"{int(frac * 100):3d}% | {label} | elapsed {_format_clock(elapsed)} | ETA {_format_clock(eta)}",
        )

    if input_mode == "Upload fil":
        if not uploaded:
            st.error("Upload en fil først.")
            st.stop()
        job_id = f"{Path(uploaded.name).stem}"
        out_dir = OUTPUT_ROOT / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        src_path = out_dir / uploaded.name
        save_uploaded_file(uploaded, src_path)
    else:
        if not url_input.strip():
            st.error("Indtast en URL først.")
            st.stop()
        job_id = datetime.now().strftime("url_%Y%m%d_%H%M%S")
        out_dir = OUTPUT_ROOT / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        st.subheader("0) Download fra URL")
        with st.spinner("Downloader lyd..."):
            try:
                src_path, yt_meta = download_audio(url_input.strip(), out_dir / "Downloads")
                tick("Download færdig")
            except DownloadError as e:
                st.error("Download fra URL mislykkedes.")
                if getattr(e, "error_code", None):
                    st.write(f"Fejlkategori: {e.error_code.value}")
                if getattr(e, "hint", None):
                    st.write(e.hint)
                if getattr(e, "log_file", None):
                    st.write(f"Se download-log: {e.log_file}")
                st.stop()

    tick("Input klargjort")
    st.caption("Input probe (ffmpeg):")
    st.code(probe_audio(src_path), language="text")

    wav_speech = out_dir / "input_speech.wav"
    wav_music = out_dir / "input_music.wav"

    to_wav(src_path, wav_speech, sr=16000, mono=True)
    to_wav(src_path, wav_music, sr=44100, mono=False)
    tick("Audio konverteret")

    job = JobOutputs(
        job_id=job_id,
        out_dir=out_dir,
        src_path=src_path,
        wav_speech=wav_speech,
        wav_music=wav_music,
    )

    transcript: Optional[Dict[str, Any]] = None
    transcript_step_done = False

    def _mark_transcript_outputs() -> None:
        job.transcript_txt = out_dir / "transcript.txt"
        job.transcript_srt = out_dir / "transcript.srt"
        job.transcript_json = out_dir / "transcript.json"
        wx_words = out_dir / "transcript_words.json"
        job.transcript_words_json = wx_words if wx_words.exists() else None

    # 1) Transcribe
    if want_transcribe:
        st.subheader("1) Transcribe")
        with st.spinner("Transcribing..."):
            transcript = transcribe_audio(
                wav_path=wav_speech,
                out_dir=out_dir,
                backend="whisperx" if backend.startswith("Precise") else "faster-whisper",
                model_size=model_size,
                language=None if language == "auto" else language,
                beam_size=beam_size,
            )
        st.success(f"Done. Language={transcript.get('language')} (p={transcript.get('language_probability')})")
        st.write(f"Segments: {len(transcript['segments'])}")
        _mark_transcript_outputs()
        tick("Transskription færdig")
        transcript_step_done = True

    # 2) Clips
    if want_clips:
        if transcript is None:
            # If user selected Clips-only, we still need a transcript.
            st.subheader("1) Transcribe (auto, required for clips)")
            with st.spinner("Transcribing..."):
                transcript = transcribe_audio(
                    wav_path=wav_speech,
                    out_dir=out_dir,
                    backend="whisperx" if backend.startswith("Precise") else "faster-whisper",
                    model_size=model_size,
                    language=None if language == "auto" else language,
                    beam_size=beam_size,
                )
            st.success("Transcription done.")
            _mark_transcript_outputs()
            if not transcript_step_done:
                tick("Transskription færdig")
                transcript_step_done = True

        st.subheader("2) Export clips (speech)")
        clips_dir = out_dir / "clips"
        clips_dir.mkdir(exist_ok=True)
        clip_segments = list(transcript.get("segments", []))
        if sample_mode == "Focused":
            target_duration, word_weight = _profile_config(focus_profile)
            selected, ranked = _select_focused_segments(
                segments=clip_segments,
                max_items=int(max_clip_samples),
                min_gap=float(min_gap_focus),
                target_duration=float(target_duration),
                word_weight=float(word_weight),
            )
            clip_segments = selected
            focus_report = {
                "mode": sample_mode,
                "profile": focus_profile,
                "max_clip_samples": int(max_clip_samples),
                "min_gap_focus": float(min_gap_focus),
                "selected_count": len(selected),
                "total_count": len(transcript.get("segments", [])),
                "selected": selected,
                "ranked_top_30": ranked[:30],
            }
            focus_report_path = out_dir / "clip_focus_report.json"
            focus_report_path.write_text(json.dumps(focus_report, ensure_ascii=False, indent=2), encoding="utf-8")
            job.focus_report_json = focus_report_path
            st.info(f"Focused selection: {len(selected)} af {len(transcript.get('segments', []))} segmenter.")
        else:
            st.info(f"All segments mode: {len(clip_segments)} segmenter eksporteres.")
        with st.spinner("Exporting clips..."):
            export_clips_from_segments(
                wav_path=wav_speech,
                segments=clip_segments,
                out_dir=clips_dir,
                min_clip_sec=min_clip_sec,
                pad_sec=pad_sec,
                fade_ms=fade_ms,
                export_format=export_format,
            )
        job.clips_dir = clips_dir
        st.success("Clips exported.")
        tick("Klip eksporteret")

    # 3) Loops
    if want_loops:
        st.subheader("3) Export loops (music)")
        loops_dir = out_dir / "loops"
        loops_dir.mkdir(exist_ok=True)

        hop = 1 if loop_hop == "every beat" else 4  # 4 beats = 1 bar
        loops_top_n = int(top_n_loops)
        if sample_mode == "Focused":
            loops_top_n = min(loops_top_n, max(8, int(max_clip_samples)))
        with st.spinner("Analyzing beats + exporting loops..."):
            export_bar_loops(
                wav_music=wav_music,
                out_dir=loops_dir,
                bars=bars,
                snap_to_beats=prefer_beats,
                hop_beats=hop,
                fade_ms=fade_ms,
                export_format=export_format,
                top_n=loops_top_n,
            )
        job.loops_dir = loops_dir
        st.success("Loops exported.")
        tick("Loops eksporteret")

    # ZIP
    zip_path = out_dir / f"{job_id}_mini_splitter.zip"
    _zip_folder_to_file(out_dir, zip_path)
    job.zip_path = zip_path
    tick("Bundle klar")
    return job

# --- Run ---
if run:
    try:
        job = run_pipeline()
        st.session_state.job = job
    except Exception as e:
        st.error(f"Run failed: {type(e).__name__}: {e}")
        st.exception(e)

job: Optional[JobOutputs] = st.session_state.job

# --- Results UI (always shown if we have a job) ---
if job:
    st.divider()
    st.header("Results")

    # MAIN ZIP download (big)
    st.subheader("Download ALL")
    if job.zip_path and Path(job.zip_path).exists():
        st.markdown('<div class="mainzip">', unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download ALL outputs as ZIP",
            data=read_bytes(job.zip_path),
            file_name=job.zip_path.name,
            mime="application/zip",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("ZIP ikke fundet for dette run. Brug Recovery for at indlæse et fuldt run.")
    st.caption(f"Output folder: {job.out_dir.resolve()}")

    tab_trans, tab_clips, tab_loops = st.tabs(["Transcript", "Clips", "Loops"])

    # ---- Transcript tab ----
    with tab_trans:
        st.subheader("Transcript preview")
        if job.transcript_txt and job.transcript_txt.exists():
            txt = job.transcript_txt.read_text(encoding="utf-8", errors="replace")
            st.text_area("Text", txt, height=260)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button(
                    "Download transcript.txt",
                    data=read_bytes(job.transcript_txt),
                    file_name=job.transcript_txt.name,
                    mime="text/plain",
                    use_container_width=True,
                )
            with c2:
                if job.transcript_srt and job.transcript_srt.exists():
                    st.download_button(
                        "Download transcript.srt",
                        data=read_bytes(job.transcript_srt),
                        file_name=job.transcript_srt.name,
                        mime="text/plain",
                        use_container_width=True,
                    )
            with c3:
                if job.transcript_json and job.transcript_json.exists():
                    st.download_button(
                        "Download transcript.json",
                        data=read_bytes(job.transcript_json),
                        file_name=job.transcript_json.name,
                        mime="application/json",
                        use_container_width=True,
                    )

            if job.transcript_words_json and job.transcript_words_json.exists():
                st.info("WhisperX word-level timings er tilgængelige.")
                st.download_button(
                    "Download transcript_words.json (WhisperX)",
                    data=read_bytes(job.transcript_words_json),
                    file_name=job.transcript_words_json.name,
                    mime="application/json",
                    use_container_width=True,
                )
        else:
            st.warning("Ingen transcript fundet i dette run (kør Transcribe eller All).")

    # ---- Clips tab ----
    with tab_clips:
        st.subheader("Clip preview + individual download")
        if job.focus_report_json and job.focus_report_json.exists():
            st.download_button(
                "Download clip_focus_report.json",
                data=read_bytes(job.focus_report_json),
                file_name=job.focus_report_json.name,
                mime="application/json",
                use_container_width=True,
            )
        if job.clips_dir and job.clips_dir.exists():
            files = _list_audio_files(job.clips_dir)
            if not files:
                st.warning("Ingen klip fundet.")
            else:
                options = {_safe_rel(p, job.clips_dir): p for p in files}
                choice = st.selectbox("Vælg et klip", list(options.keys()))
                p = options[choice]
                st.audio(read_bytes(p))
                st.download_button(
                    "Download selected clip",
                    data=read_bytes(p),
                    file_name=p.name,
                    mime="audio/wav" if p.suffix.lower() == ".wav" else "audio/mpeg",
                    use_container_width=True,
                )
                st.caption(f"{len(files)} clips i alt.")
        else:
            st.info("Kør Clips (Speech) eller All for at få klip.")

    # ---- Loops tab ----
    with tab_loops:
        st.subheader("Loop preview + individual download")
        if job.loops_dir and job.loops_dir.exists():
            files = _list_audio_files(job.loops_dir)
            report = job.loops_dir / "loops_report.txt"
            if report.exists():
                st.caption("Loop report:")
                st.code(report.read_text(encoding="utf-8", errors="replace"), language="text")

            if not files:
                st.warning("Ingen loops fundet.")
            else:
                options = {_safe_rel(p, job.loops_dir): p for p in files}
                choice = st.selectbox("Vælg et loop", list(options.keys()))
                p = options[choice]
                st.audio(read_bytes(p))
                st.download_button(
                    "Download selected loop",
                    data=read_bytes(p),
                    file_name=p.name,
                    mime="audio/wav" if p.suffix.lower() == ".wav" else "audio/mpeg",
                    use_container_width=True,
                )
                st.caption(f"{len(files)} loops i alt.")
        else:
            st.info("Kør Loops (Music) eller All for at få loops.")
else:
    st.info("Kør et run først eller brug 'Recovery / Load previous run' for at få preview og eksport-knapper.")
