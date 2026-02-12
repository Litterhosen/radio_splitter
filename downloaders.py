from pathlib import Path
import yt_dlp

from utils import ensure_dir, find_ffmpeg


def download_audio(url: str, out_dir: Path) -> Path:
    out_dir = ensure_dir(out_dir)
    ffmpeg = find_ffmpeg()

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(title)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "geo_bypass": True,
        "nocheckcertificate": True,
        "extractor_retries": 3,
        "retries": 3,
        "http_headers": {
            "User-Agent": "Mozilla/5.0",
        },
        # Forsøg at bruge forskellige clients – hjælper ofte på "not available on this app"
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"]
            }
        },
        "ffmpeg_location": ffmpeg if ffmpeg != "ffmpeg" else None,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"},
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
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
