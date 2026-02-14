from pathlib import Path
import yt_dlp
import logging
import shutil
import subprocess
from datetime import datetime
from typing import Tuple, Optional, Dict
from enum import Enum


class ErrorClassification(Enum):
    """Error classification for YouTube download failures."""
    ERR_JS_RUNTIME_MISSING = "ERR_JS_RUNTIME_MISSING"
    ERR_VIDEO_UNAVAILABLE = "ERR_VIDEO_UNAVAILABLE"
    ERR_GEO_BLOCK = "ERR_GEO_BLOCK"
    ERR_LOGIN_REQUIRED = "ERR_LOGIN_REQUIRED"
    ERR_NETWORK = "ERR_NETWORK"
    ERR_UNKNOWN = "ERR_UNKNOWN"


class DownloadError(Exception):
    """Custom exception for download failures."""
    def __init__(self, message, log_file=None, last_error=None, url=None, hint=None, error_code=None):
        super().__init__(message)
        self.log_file = log_file
        self.last_error = last_error
        self.url = url
        self.hint = hint
        self.error_code = error_code  # Optional[ErrorClassification] enum value


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


def classify_error(error_text: str, has_js_runtime: bool) -> Tuple[ErrorClassification, str, str]:
    """
    Classify download error and provide helpful guidance.
    
    Args:
        error_text: The error message text
        has_js_runtime: Whether JS runtime is available
    
    Returns:
        Tuple of (ErrorClassification, hint, next_steps)
    """
    error_lower = error_text.lower()
    
    # Check for specific error patterns
    if not has_js_runtime and ("player" in error_lower or "signature" in error_lower):
        return (
            ErrorClassification.ERR_JS_RUNTIME_MISSING,
            "JavaScript runtime (Node.js) is required for this video",
            "1. Install Node.js from https://nodejs.org/\n"
            "2. Restart the application\n"
            "3. Try the download again"
        )
    
    if "video unavailable" in error_lower or "removed" in error_lower or "deleted" in error_lower:
        return (
            ErrorClassification.ERR_VIDEO_UNAVAILABLE,
            "Video is unavailable, removed, or deleted",
            "1. Verify the URL is correct\n"
            "2. Check if the video exists in your browser\n"
            "3. Try a different video URL"
        )
    
    if "geo" in error_lower or "not available in your country" in error_lower or "region" in error_lower:
        return (
            ErrorClassification.ERR_GEO_BLOCK,
            "Video is geo-blocked or region-restricted",
            "1. This video cannot be accessed from your location\n"
            "2. Consider using a VPN (if permitted)\n"
            "3. Try a different video available in your region"
        )
    
    if "sign in" in error_lower or "login" in error_lower or "authenticate" in error_lower or "age" in error_lower:
        return (
            ErrorClassification.ERR_LOGIN_REQUIRED,
            "Video requires authentication or age verification",
            "1. Video may be age-restricted or private\n"
            "2. Try using cookies file with --cookies option\n"
            "3. Ensure you have access to this content"
        )
    
    if "network" in error_lower or "connection" in error_lower or "timeout" in error_lower or "http" in error_lower:
        return (
            ErrorClassification.ERR_NETWORK,
            "Network or connection issue",
            "1. Check your internet connection\n"
            "2. Verify the URL is accessible in your browser\n"
            "3. Try again in a few moments"
        )
    
    # Default unknown error
    return (
        ErrorClassification.ERR_UNKNOWN,
        "Unknown download error - see log for details",
        "1. Update yt-dlp: pip install --upgrade yt-dlp\n"
        "2. Check the detailed log file\n"
        "3. Verify the URL works in your browser"
    )


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
            extractor_info = "player_client=['android', 'web']" if js_runtime else "default"
            logger.info(f"Extractor strategy: {extractor_info}")
            
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
    
    # All strategies failed - classify the error
    error_msg = f"Download failed after trying all format strategies."
    logger.error(error_msg)
    logger.error(f"Last error: {last_error}")
    logger.error("")
    
    # Classify error and get structured guidance
    error_code, hint, next_steps = classify_error(str(last_error), js_runtime is not None)
    
    logger.error(f"ERROR CLASSIFICATION: {error_code.value}")
    logger.error("")
    logger.error("NEXT STEPS:")
    for line in next_steps.split('\n'):
        logger.error(line)
    
    logger.close()
    
    raise DownloadError(
        error_msg, 
        log_file=str(log_file), 
        last_error=last_error,
        url=url,
        hint=hint,
        error_code=error_code
    )
