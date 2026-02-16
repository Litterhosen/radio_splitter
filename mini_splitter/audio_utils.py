import shutil
import subprocess
from pathlib import Path


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found. Installér ffmpeg (fx: winget install Gyan.FFmpeg) og prøv igen."
        )


def save_uploaded_file(uploaded, path: Path) -> None:
    path.write_bytes(uploaded.getbuffer())


def probe_audio(path: Path) -> str:
    cmd = ["ffmpeg", "-hide_banner", "-i", str(path)]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    lines = completed.stderr.splitlines()
    return "\n".join(lines[-10:]) if lines else "OK"


def to_wav(src: Path, dst: Path, sr: int, mono: bool) -> None:
    channels = "1" if mono else "2"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ar",
        str(sr),
        "-ac",
        channels,
        "-vn",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
