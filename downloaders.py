from pathlib import Path

from utils import ensure_dir, find_ffmpeg
from yt_dlp import YoutubeDL


def download_audio(url: str, out_dir: Path) -> Path:
    out_dir = ensure_dir(out_dir)
    ffmpeg = find_ffmpeg()

    ydl_opts = {
        "outtmpl": str(out_dir / "%(title)s.%(ext)s"),
        "format": "bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": ffmpeg if ffmpeg != "ffmpeg" else None,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"},
        ],
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "download")
        final = out_dir / f"{title}.mp3"
        if final.exists():
            return final

        req = ydl.prepare_filename(info)
        p = Path(req).with_suffix(".mp3")
        if p.exists():
            return p

    raise RuntimeError("Download lykkedes ikke (ingen mp3 fundet).")
