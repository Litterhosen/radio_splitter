from pathlib import Path
import yt_dlp


def download_audio(url, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(title)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    # postprocessor changes extension to .mp3
    out = Path(filename).with_suffix(".mp3")
    if out.exists():
        return out
    # fallback to original filename if postprocessor didn't run
    return Path(filename)
