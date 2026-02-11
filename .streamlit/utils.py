import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import streamlit as st


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def hhmmss_ms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = (int(seconds) // 3600)
    return f"{h:02d}-{m:02d}-{s:02d}-{ms:03d}"


def safe_slug(text: str, max_len: int = 48) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9æøå _-]+", "", t, flags=re.IGNORECASE)
    t = t.replace(" ", "_")
    t = re.sub(r"_+", "_", t).strip("_")
    return t[:max_len]


def _ffmpeg_from_imageio() -> Optional[str]:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def find_ffmpeg() -> str:
    """
    FFMPEG discovery priority:
    1) st.secrets["FFMPEG_PATH"] (Streamlit)
    2) env var FFMPEG_PATH (via subprocess inherits env)
    3) "ffmpeg" in PATH
    4) imageio-ffmpeg bundled binary
    """
    try:
        sp = st.secrets.get("FFMPEG_PATH", "")
        if sp:
            return str(sp)
    except Exception:
        pass

    # PATH check
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        return "ffmpeg"
    except Exception:
        pass

    img = _ffmpeg_from_imageio()
    if img:
        return img

    raise RuntimeError(
        "FFmpeg ikke fundet. Installér ffmpeg (Windows) eller brug packages.txt på Streamlit Cloud."
    )


def find_ffprobe() -> str:
    # If ffmpeg is bundled, ffprobe sits next to it
    ffmpeg = find_ffmpeg()
    if ffmpeg == "ffmpeg":
        return "ffprobe"
    p = Path(ffmpeg)
    cand = p.with_name("ffprobe.exe" if p.name.lower().endswith(".exe") else "ffprobe")
    if cand.exists():
        return str(cand)
    return "ffprobe"


@dataclass
class CmdResult:
    cmd: List[str]
    rc: int
    out: str
    err: str


def run_cmd(cmd: List[str], check: bool = True) -> CmdResult:
    p = subprocess.run(cmd, capture_output=True, text=True)
    res = CmdResult(cmd=cmd, rc=p.returncode, out=p.stdout or "", err=p.stderr or "")
    if check and res.rc != 0:
        raise RuntimeError(f"Command failed ({res.rc}): {' '.join(cmd)}\n{res.err}")
    return res
