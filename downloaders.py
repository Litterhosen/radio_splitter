from pathlib import Path
import yt_dlp
import logging
import shutil
import subprocess
from datetime import datetime
from typing import Tuple, Optional, Dict


class DownloadError(Exception):
    """Custom exception for download failures."""
    def __init__(self, message, log_file=None, last_error=None, url=None, hint=None):
        super().__init__(message)
        self.log_file = log_file
        self.last_error = last_error
        self.url = url
        self.hint = hint


class YTDLPLogger:
    """Custom logger to capture yt-dlp output."""
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Open file in append mode for efficient logging
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
    
    def debug(self, msg):
        self._write_line(f"[DEBUG] {msg}")
    
    def info(self, msg):
        self._write_line(f"[INFO] {msg}")
    
    def warning(self, msg):
        self._write_line(f"[WARNING] {msg}")
    
    def error(self, msg):
        self._write_line(f"[ERROR] {msg}")
    
    def _write_line(self, line):
        """Write a single line to the log file and flush."""
        self.log_file.write(line + '\n')
        self.log_file.flush()
    
    def close(self):
        """Close the log file."""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()


def check_js_runtime() -> Optional[str]:
    """
    Check if a JavaScript runtime is available.
    
    Returns:
        Path to node executable if found, None otherwise
    """
    node_path = shutil.which("node")
    if node_path:
        return node_path
    return None


def download_audio(url, out_dir) -> Tuple[Path, Dict]:
    """
    Download audio from URL with fallback strategies, error logging, and verification.
    
    Args:
        url: URL to download from
        out_dir: Output directory Path
    
    Returns:
        Tuple of (Path to downloaded file, YouTube metadata dict)
    
    Raises:
        DownloadError: If download fails with log file reference and last error
    """
    from utils import verify_audio_file
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = out_dir / f"download_log_{timestamp}.txt"
    logger = YTDLPLogger(log_file)
    
    # Check for JS runtime
    js_runtime = check_js_runtime()
    if js_runtime:
        logger.info(f"JavaScript runtime detected: {js_runtime}")
    else:
        logger.warning("No JavaScript runtime found. Some videos may fail to download.")
        logger.warning("Consider installing Node.js for better compatibility.")
    
    # Format strategies to try in order
    format_strategies = [
        "bestaudio/best",
        "bestaudio[ext=m4a]/bestaudio",
        "worstaudio",
    ]
    
    last_error = None
    youtube_info = {}
    
    for strategy in format_strategies:
        try:
            logger.info(f"Attempting download with format: {strategy}")
            logger.info(f"URL: {url}")
            
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
            
            # Add JS runtime if available
            if js_runtime:
                ydl_opts["extractor_args"] = {
                    "youtube": {
                        "player_client": ["android", "web"]
                    }
                }
            
            # Try extractor fallback strategies
            logger.info(f"Extractor strategy: player_client=['android', 'web']" if js_runtime else "Extractor strategy: default")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                # Store YouTube metadata
                if info:
                    youtube_info = {
                        "title": info.get("title", ""),
                        "uploader": info.get("uploader", ""),
                        "id": info.get("id", ""),
                        "duration": info.get("duration", 0),
                    }
                    logger.info(f"YouTube metadata: title={youtube_info['title']}, uploader={youtube_info['uploader']}")
                
                filename = ydl.prepare_filename(info)
                p = Path(filename)
                
                # Check for various possible extensions after post-processing
                possible_exts = [".mp3", ".m4a", ".opus", ".webm", ".wav"]
                found_file = None
                
                for ext in possible_exts:
                    candidate = p.with_suffix(ext)
                    if candidate.exists():
                        found_file = candidate
                        break
                
                # If original file exists
                if not found_file and p.exists():
                    found_file = p
                
                if not found_file:
                    raise FileNotFoundError(f"Downloaded file not found. Expected: {filename}")
                
                # VERIFY THE FILE
                logger.info(f"Verifying downloaded file: {found_file.name}")
                is_valid, error_msg = verify_audio_file(found_file)
                
                if not is_valid:
                    raise RuntimeError(f"File verification failed: {error_msg}")
                
                # Success!
                logger.info(f"Download successful and verified: {found_file.name}")
                logger.info(f"File size: {found_file.stat().st_size} bytes")
                logger.close()
                return found_file, youtube_info
        
        except Exception as e:
            last_error = str(e)
            logger.error(f"Strategy '{strategy}' failed: {last_error}")
            continue
    
    # All strategies failed
    error_msg = f"Download failed after trying all format strategies."
    logger.error(error_msg)
    logger.error(f"Last error: {last_error}")
    logger.error("")
    logger.error("TROUBLESHOOTING:")
    logger.error("1. Update yt-dlp: pip install --upgrade yt-dlp")
    
    # Generate helpful hints based on error
    hint = ""
    if not js_runtime:
        logger.error("2. Install Node.js for JavaScript runtime support")
        hint = "Install Node.js (https://nodejs.org/) for better video extraction compatibility"
    
    if "Video unavailable" in str(last_error):
        logger.error("3. Video may be geo-blocked or require login")
        hint += " | Video may be geo-blocked or age-restricted"
    
    if "Sign in" in str(last_error) or "login" in str(last_error).lower():
        logger.error("4. Try using cookies file for authenticated content")
        hint += " | Video may require authentication - consider using cookies"
    
    logger.error("5. Check if URL is accessible in your browser")
    logger.close()
    
    raise DownloadError(
        error_msg, 
        log_file=str(log_file), 
        last_error=last_error,
        url=url,
        hint=hint if hint else "Check the log file for details"
    )
