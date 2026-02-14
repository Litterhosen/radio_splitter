from pathlib import Path
import yt_dlp
import logging
from datetime import datetime


class DownloadError(Exception):
    """Custom exception for download failures."""
    def __init__(self, message, log_file=None):
        super().__init__(message)
        self.log_file = log_file


class YTDLPLogger:
    """Custom logger to capture yt-dlp output."""
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lines = []
    
    def debug(self, msg):
        self.lines.append(f"[DEBUG] {msg}")
        self._write()
    
    def info(self, msg):
        self.lines.append(f"[INFO] {msg}")
        self._write()
    
    def warning(self, msg):
        self.lines.append(f"[WARNING] {msg}")
        self._write()
    
    def error(self, msg):
        self.lines.append(f"[ERROR] {msg}")
        self._write()
    
    def _write(self):
        """Write all lines to log file."""
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.lines))


def download_audio(url, out_dir):
    """
    Download audio from URL with fallback strategies and error logging.
    
    Args:
        url: URL to download from
        out_dir: Output directory Path
    
    Returns:
        Path to downloaded file
    
    Raises:
        DownloadError: If download fails with log file reference
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = out_dir / f"download_log_{timestamp}.txt"
    logger = YTDLPLogger(log_file)
    
    # Format strategies to try in order
    format_strategies = [
        "bestaudio/best",
        "bestaudio[ext=m4a]/bestaudio",
        "worstaudio",
    ]
    
    last_error = None
    
    for strategy in format_strategies:
        try:
            logger.info(f"Attempting download with format: {strategy}")
            
            ydl_opts = {
                "format": strategy,
                "outtmpl": str(out_dir / "%(title)s.%(ext)s"),
                "noplaylist": True,
                "quiet": False,
                "no_warnings": False,
                "logger": logger,
                "restrictfilenames": True,
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }],
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                p = Path(filename)
                
                # Check for various possible extensions after post-processing
                possible_exts = [".mp3", ".m4a", ".opus", ".webm", ".wav"]
                for ext in possible_exts:
                    candidate = p.with_suffix(ext)
                    if candidate.exists():
                        logger.info(f"Download successful: {candidate.name}")
                        return candidate
                
                # If original file exists
                if p.exists():
                    logger.info(f"Download successful: {p.name}")
                    return p
                
                raise FileNotFoundError(f"Downloaded file not found: {filename}")
        
        except Exception as e:
            last_error = str(e)
            logger.error(f"Strategy '{strategy}' failed: {last_error}")
            continue
    
    # All strategies failed
    error_msg = f"Download failed after trying all format strategies. Last error: {last_error}"
    logger.error(error_msg)
    raise DownloadError(error_msg, log_file=str(log_file))
