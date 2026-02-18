import re
import os
import subprocess
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict


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
    1) env var FFMPEG_PATH
    2) "ffmpeg" in PATH
    3) imageio-ffmpeg bundled binary
    """
    ffmpeg_env = os.environ.get("FFMPEG_PATH", "").strip()
    if ffmpeg_env:
        return ffmpeg_env

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


def snap_bars_to_valid(raw_bars: float, prefer_bars: int = 2, tolerance: float = 0.25) -> int:
    """
    Snap raw bars estimate to valid DAW-friendly values [1, 2, 4, 8, 16].
    
    Args:
        raw_bars: Raw calculated bars (can be fractional)
        prefer_bars: Preferred bar count (user preference)
        tolerance: Tolerance for snapping to prefer_bars (fraction of prefer_bars)
    
    Returns:
        Snapped bar count from [1, 2, 4, 8, 16]
    """
    valid_bars = [1, 2, 4, 8, 16]
    
    # If raw_bars is close to prefer_bars, use prefer_bars
    if prefer_bars in valid_bars:
        if abs(raw_bars - prefer_bars) <= (prefer_bars * tolerance):
            return prefer_bars
    
    # Otherwise, find closest valid value
    closest = min(valid_bars, key=lambda x: abs(x - raw_bars))
    return closest


def mmss(seconds: float) -> str:
    """
    Convert seconds to MM-SS format for filename timestamps.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        String in format MM-SS (e.g., "03-45")
    """
    seconds = max(0.0, float(seconds))
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}-{s:02d}"


def extract_id3_metadata(file_path: Path) -> Dict[str, Optional[str]]:
    """
    Extract artist and title from audio file ID3 tags using mutagen.
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Dictionary with 'artist' and 'title' keys (None if not found)
    """
    try:
        from mutagen import File
        audio = File(file_path, easy=True)
        if audio is None:
            return {"artist": None, "title": None}
        
        # Try to get artist (various tag formats)
        artist = None
        if hasattr(audio, 'tags') and audio.tags:
            artist = audio.tags.get('artist', [None])[0] if isinstance(audio.tags.get('artist'), list) else audio.tags.get('artist')
            if not artist:
                artist = audio.tags.get('TPE1', [None])[0] if isinstance(audio.tags.get('TPE1'), list) else audio.tags.get('TPE1')
        
        # Try to get title
        title = None
        if hasattr(audio, 'tags') and audio.tags:
            title = audio.tags.get('title', [None])[0] if isinstance(audio.tags.get('title'), list) else audio.tags.get('title')
            if not title:
                title = audio.tags.get('TIT2', [None])[0] if isinstance(audio.tags.get('TIT2'), list) else audio.tags.get('TIT2')
        
        return {
            "artist": str(artist).strip() if artist else None,
            "title": str(title).strip() if title else None
        }
    except Exception:
        return {"artist": None, "title": None}


def extract_track_metadata(file_path: Path, youtube_info: Optional[Dict] = None) -> Tuple[str, str, str]:
    """
    Extract track artist, title, and safe ID from file.
    
    Priority:
    1. ID3 tags from file
    2. YouTube info (uploader/title)
    3. Filename fallback
    
    Args:
        file_path: Path to audio file
        youtube_info: Optional dict with YouTube metadata (uploader, title)
    
    Returns:
        Tuple of (artist, title, track_id) - all OS-safe strings
    """
    # Max lengths for OS safety
    MAX_ARTIST_LEN = 32
    MAX_TITLE_LEN = 48
    
    artist = None
    title = None
    
    # Try ID3 tags first
    if file_path.exists():
        id3_data = extract_id3_metadata(file_path)
        artist = id3_data.get("artist")
        title = id3_data.get("title")
    
    # Try YouTube info if ID3 failed
    if (not artist or not title) and youtube_info:
        if not artist and youtube_info.get("uploader"):
            artist = youtube_info["uploader"]
        if not title and youtube_info.get("title"):
            title = youtube_info["title"]
    
    # Fallback to filename
    if not title:
        title = file_path.stem
    
    # Use default if still missing
    if not artist:
        artist = "UnknownArtist"
    if not title:
        title = "UnknownTitle"
    
    # Make OS-safe
    artist_safe = safe_slug(artist, max_len=MAX_ARTIST_LEN)
    title_safe = safe_slug(title, max_len=MAX_TITLE_LEN)
    
    # Create track_id from artist + title
    track_id = safe_slug(f"{artist}_{title}", max_len=60)
    
    return artist_safe, title_safe, track_id


def verify_audio_file(file_path: Path) -> Tuple[bool, str]:
    """
    Verify audio file is valid using ffprobe.
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"
    
    if file_path.stat().st_size == 0:
        return False, f"File is empty (0 bytes): {file_path}"
    
    try:
        ffprobe = find_ffprobe()
        cmd = [
            ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(file_path)
        ]
        result = run_cmd(cmd, check=False)
        
        if result.rc != 0:
            return False, f"ffprobe failed: {result.err}"
        
        # Parse JSON output
        data = json.loads(result.out)
        duration = float(data.get("format", {}).get("duration", 0))
        
        if duration <= 0:
            return False, f"Invalid duration: {duration}s"
        
        return True, ""
    
    except Exception as e:
        return False, f"Verification error: {str(e)}"
