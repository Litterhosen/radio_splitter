# audio_utils.py
import shutil
import subprocess
from pathlib import Path

def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found. Installér ffmpeg (fx: winget install Gyan.FFmpeg) og prøv igen."
        )

def save_uploaded_file(uploaded, path: Path):
    path.write_bytes(uploaded.getbuffer())

def probe_audio(path: Path) -> str:
    # Minimal probe via ffmpeg
    cmd = ["ffmpeg", "-hide_banner", "-i", str(path)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    # ffmpeg writes probe to stderr
    return p.stderr.splitlines()[-10:] and "\n".join(p.stderr.splitlines()[-10:]) or "OK"

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
