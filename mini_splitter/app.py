# app.py
import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import sys

import streamlit as st

from audio_utils import ensure_ffmpeg, save_uploaded_file, probe_audio, to_wav, read_bytes
from transcribe import transcribe_audio
from clipper import export_clips_from_segments
from loops import export_bar_loops

try:
    from downloaders import download_audio, DownloadError
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from downloaders import download_audio, DownloadError

OUTPUT_ROOT = Path("output")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Mini Splitter+", layout="wide")
st.title("Mini Splitter+ (Transcribe + Clips + Loops)")

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

st.write("Vælg input: upload en lydfil eller indsæt en YouTube/online URL. Efter ‘Run’ får du previews, individuelle downloads og én samlet zip.")

uploaded = st.file_uploader("Audio file", type=["mp3", "wav", "m4a", "flac", "aac", "ogg"], disabled=(input_mode != "Upload fil"))
url_input = st.text_input("YouTube / online URL", "", disabled=(input_mode != "YouTube/URL"))
run = st.button("Run", type="primary")

def run_pipeline() -> JobOutputs:
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
            except DownloadError as e:
                st.error("Download fra URL mislykkedes.")
                if getattr(e, "error_code", None):
                    st.write(f"Fejlkategori: {e.error_code.value}")
                if getattr(e, "hint", None):
                    st.write(e.hint)
                if getattr(e, "log_file", None):
                    st.write(f"Se download-log: {e.log_file}")
                st.stop()

    st.caption("Input probe (ffmpeg):")
    st.code(probe_audio(src_path), language="text")

    wav_speech = out_dir / "input_speech.wav"
    wav_music = out_dir / "input_music.wav"

    to_wav(src_path, wav_speech, sr=16000, mono=True)
    to_wav(src_path, wav_music, sr=44100, mono=False)

    job = JobOutputs(
        job_id=job_id,
        out_dir=out_dir,
        src_path=src_path,
        wav_speech=wav_speech,
        wav_music=wav_music,
    )

    want_transcribe = mode in ["All", "Transcribe", "Clips (Speech)"]
    want_clips = mode in ["All", "Clips (Speech)"]
    want_loops = mode in ["All", "Loops (Music)"]

    transcript: Optional[Dict[str, Any]] = None

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

        # Set known output paths (created by transcribe_audio)
        job.transcript_txt = out_dir / "transcript.txt"
        job.transcript_srt = out_dir / "transcript.srt"
        job.transcript_json = out_dir / "transcript.json"
        wx_words = out_dir / "transcript_words.json"
        job.transcript_words_json = wx_words if wx_words.exists() else None

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

            job.transcript_txt = out_dir / "transcript.txt"
            job.transcript_srt = out_dir / "transcript.srt"
            job.transcript_json = out_dir / "transcript.json"
            wx_words = out_dir / "transcript_words.json"
            job.transcript_words_json = wx_words if wx_words.exists() else None

        st.subheader("2) Export clips (speech)")
        clips_dir = out_dir / "clips"
        clips_dir.mkdir(exist_ok=True)
        with st.spinner("Exporting clips..."):
            export_clips_from_segments(
                wav_path=wav_speech,
                segments=transcript["segments"],
                out_dir=clips_dir,
                min_clip_sec=min_clip_sec,
                pad_sec=pad_sec,
                fade_ms=fade_ms,
                export_format=export_format,
            )
        job.clips_dir = clips_dir
        st.success("Clips exported.")

    # 3) Loops
    if want_loops:
        st.subheader("3) Export loops (music)")
        loops_dir = out_dir / "loops"
        loops_dir.mkdir(exist_ok=True)

        hop = 1 if loop_hop == "every beat" else 4  # 4 beats = 1 bar
        with st.spinner("Analyzing beats + exporting loops..."):
            export_bar_loops(
                wav_music=wav_music,
                out_dir=loops_dir,
                bars=bars,
                snap_to_beats=prefer_beats,
                hop_beats=hop,
                fade_ms=fade_ms,
                export_format=export_format,
                top_n=top_n_loops,
            )
        job.loops_dir = loops_dir
        st.success("Loops exported.")

    # ZIP
    zip_path = out_dir / f"{job_id}_mini_splitter.zip"
    _zip_folder_to_file(out_dir, zip_path)
    job.zip_path = zip_path
    return job

# --- Run ---
if run:
    job = run_pipeline()
    st.session_state.job = job

job: Optional[JobOutputs] = st.session_state.job

# --- Results UI (always shown if we have a job) ---
if job:
    st.divider()
    st.header("Results")

    # MAIN ZIP download (big)
    st.subheader("Download ALL")
    st.markdown('<div class="mainzip">', unsafe_allow_html=True)
    st.download_button(
        "⬇️ Download ALL outputs as ZIP",
        data=read_bytes(job.zip_path),
        file_name=job.zip_path.name,
        mime="application/zip",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
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
        if job.clips_dir and job.clips_dir.exists():
            files = _list_audio_files(job.clips_dir)
            if not files:
                st.warning("Ingen klip fundet.")
            else:
                options = { _safe_rel(p, job.clips_dir): p for p in files }
                choice = st.selectbox("Vælg et klip", list(options.keys()))
                p = options[choice]
                st.audio(read_bytes(p))
                st.download_button(
                    "⬇️ Download selected clip",
                    data=read_bytes(p),
                    file_name=p.name,
                    mime="audio/wav" if p.suffix.lower() == ".wav" else "audio/mpeg",
                    use_container_width=True,
                )

                st.caption(f"{len(files)} clips i alt.")
        else:
            st.info("Kør ‘Clips (Speech)’ eller ‘All’ for at få klip.")

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
                options = { _safe_rel(p, job.loops_dir): p for p in files }
                choice = st.selectbox("Vælg et loop", list(options.keys()))
                p = options[choice]
                st.audio(read_bytes(p))
                st.download_button(
                    "⬇️ Download selected loop",
                    data=read_bytes(p),
                    file_name=p.name,
                    mime="audio/wav" if p.suffix.lower() == ".wav" else "audio/mpeg",
                    use_container_width=True,
                )

                st.caption(f"{len(files)} loops i alt.")
        else:
            st.info("Kør ‘Loops (Music)’ eller ‘All’ for at få loops.")
