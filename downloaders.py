from pathlib import Path
import yt_dlp


class DownloadError(Exception):
    pass


def download_audio(url, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(title)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
            }
        },
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
        return Path(filename)
    except Exception:
        raise DownloadError(
            "Kunne ikke hente direkte (server blokeret). "
            "Brug Cobalt.tools til at downloade filen, og upload den i Fane 1."
        )
