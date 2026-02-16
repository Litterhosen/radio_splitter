# app.py
import os
import io
import zipfile
from pathlib import Path
import streamlit as st

from audio_utils import ensure_ffmpeg, save_uploaded_file, probe_audio, to_wav
from transcribe_fw import transcribe_faster_whisper
from clipper import export_clips_from_segments
from loops import export_bar_loops

OUTPUT_ROOT = Path("output")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Mini Splitter", layout="wide")
st.title("Mini Splitter (Transcribe + Clips + Loops)")

ensure_ffmpeg()

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Mode", ["All", "Transcribe", "Clips (Speech)", "Loops (Music)"])
    language = st.selectbox("Language", ["auto", "da", "en"])
    model_size = st.selectbox("Whisper model", ["tiny", "base", "small"], index=2)
    beam_size = st.slider("Beam size", 1, 10, 5)

    st.divider()
    st.subheader("Clips")
    min_clip_sec = st.slider("Min clip length (sec)", 0.5, 10.0, 2.0, 0.5)
    pad_sec = st.slider("Pad around clip (sec)", 0.0, 2.0, 0.25, 0.05)
    fade_ms = st.slider("Fade (ms)", 0, 300, 30, 10)

    st.divider()
    st.subheader("Loops")
    bars = st.slider("Bars (4/4)", 4, 16, 8, 1)
    prefer_beats = st.checkbox("Snap to detected beats", value=True)
    loop_hop = st.selectbox("Loop hop", ["every beat", "every bar"], index=1)

    st.divider()
    export_format = st.selectbox("Export format", ["wav", "mp3"], index=0)

st.write("Upload audio (mp3/wav/m4a/flac). Output kommer som én zip.")

uploaded = st.file_uploader("Audio file", type=["mp3", "wav", "m4a", "flac", "aac", "ogg"])
run = st.button("Run")

def zip_folder(folder: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(folder))
    buf.seek(0)
    return buf.read()

if run:
    if not uploaded:
        st.error("Upload en fil først.")
        st.stop()

    job_id = f"{Path(uploaded.name).stem}"
    out_dir = OUTPUT_ROOT / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    src_path = out_dir / uploaded.name
    save_uploaded_file(uploaded, src_path)

    info = probe_audio(src_path)
    st.info(f"Input: {info}")

    # We keep two WAVs:
    # - speech wav: 16k mono
    # - music wav: 44.1k stereo
    wav_speech = out_dir / "input_speech.wav"
    wav_music = out_dir / "input_music.wav"
    to_wav(src_path, wav_speech, sr=16000, mono=True)
    to_wav(src_path, wav_music, sr=44100, mono=False)

    transcript = None

    # 1) TRANSCRIBE
    if mode in ["All", "Transcribe", "Clips (Speech)"]:
        st.subheader("1) Transcribe")
        with st.spinner("Transcribing..."):
            transcript = transcribe_faster_whisper(
                wav_speech,
                model_size=model_size,
                language=None if language == "auto" else language,
                beam_size=beam_size,
                out_dir=out_dir,
            )
        st.success("Transcription done.")
        st.write(f"Segments: {len(transcript['segments'])}")

    # 2) CLIPS FROM SEGMENTS
    if mode in ["All", "Clips (Speech)"]:
        if transcript is None:
            st.warning("Ingen transcript fundet (kør Transcribe eller All).")
        else:
            st.subheader("2) Export clips (speech)")
            clips_dir = out_dir / "clips"
            clips_dir.mkdir(exist_ok=True)
            with st.spinner("Exporting clips..."):
                export_clips_from_segments(
                    wav_speech,
                    transcript["segments"],
                    clips_dir,
                    min_clip_sec=min_clip_sec,
                    pad_sec=pad_sec,
                    fade_ms=fade_ms,
                    export_format=export_format,
                )
            st.success("Clips exported.")

    # 3) LOOPS
    if mode in ["All", "Loops (Music)"]:
        st.subheader("3) Export loops (music)")
        loops_dir = out_dir / "loops"
        loops_dir.mkdir(exist_ok=True)

        hop = 1 if loop_hop == "every beat" else 4  # 4 beats = 1 bar
        with st.spinner("Analyzing beats + exporting loops..."):
            export_bar_loops(
                wav_music,
                loops_dir,
                bars=bars,
                snap_to_beats=prefer_beats,
                hop_beats=hop,
                fade_ms=fade_ms,
                export_format=export_format,
            )
        st.success("Loops exported.")

    st.subheader("Download")
    zbytes = zip_folder(out_dir)
    st.download_button(
        "Download ZIP",
        data=zbytes,
        file_name=f"{job_id}_mini_splitter.zip",
        mime="application/zip",
    )

    st.caption(f"Output folder: {out_dir.resolve()}")
