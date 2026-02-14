import re
import subprocess
import hashlib
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
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    res = CmdResult(cmd=cmd, rc=p.returncode, out=p.stdout or "", err=p.stderr or "")
    if check and res.rc != 0:
        raise RuntimeError(f"Command failed ({res.rc}): {' '.join(cmd)}\n{res.err}")
    return res


def safe_dirname(name: str) -> str:
    """
    Create an OS-safe directory name by stripping unsafe characters.
    Removes: [](){}|<>:;'"?*\\/
    """
    unsafe_chars = r'[\[\]\(\)\{\}\|<>:;\'"?*\\/]'
    safe = re.sub(unsafe_chars, '', name)
    safe = re.sub(r'\s+', '_', safe)
    safe = re.sub(r'_+', '_', safe).strip('_')
    return safe or "output"


def clip_uid(source: str, start: float, end: float, idx: int) -> str:
    """
    Generate a deterministic 6-character UID from clip metadata.
    Uses MD5 hash of source, start, end, and index.
    """
    data = f"{source}:{start}:{end}:{idx}"
    hash_obj = hashlib.md5(data.encode('utf-8'))
    return hash_obj.hexdigest()[:6]


def estimate_bars_from_duration(dur_sec: float, bpm: float, beats_per_bar: int = 4) -> int:
    """
    Estimate number of bars from duration, BPM, and beats per bar.
    Returns at least 1 bar if duration > 0.
    """
    if dur_sec <= 0 or bpm <= 0:
        return 1
    
    beats_per_second = bpm / 60.0
    total_beats = dur_sec * beats_per_second
    bars = max(1, int(round(total_beats / beats_per_bar)))
    return bars
