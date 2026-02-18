# audio_utils.py
import shutil
import subprocess
from pathlib import Path

import streamlit as st

def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        st.error("ffmpeg ikke fundet. Installér fx: winget install Gyan.FFmpeg")
        raise RuntimeError("ffmpeg not found")

def save_uploaded_file(uploaded, path: Path):
    path.write_bytes(uploaded.getbuffer())

def probe_audio(path: Path) -> str:
    cmd = ["ffmpeg", "-hide_banner", "-i", str(path)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    # ffmpeg prints info to stderr
    lines = p.stderr.splitlines()
    return "\n".join(lines[-14:]) if lines else "OK"

def to_wav(src: Path, dst: Path, sr: int, mono: bool):
    ch = "1" if mono else "2"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ar", str(sr),
        "-ac", ch,
        "-vn",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def read_bytes(path: Path) -> bytes:
    return Path(path).read_bytes()
