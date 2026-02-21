from pathlib import Path
import yt_dlp
import logging
import shutil
import subprocess
import os
from datetime import datetime
from typing import Tuple, Optional, Dict, List
from enum import Enum


class ErrorClassification(Enum):
    """Error classification for YouTube download failures."""
    ERR_JS_RUNTIME_MISSING = "ERR_JS_RUNTIME_MISSING"
    ERR_VIDEO_UNAVAILABLE = "ERR_VIDEO_UNAVAILABLE"
    ERR_GEO_BLOCK = "ERR_GEO_BLOCK"
    ERR_AGE_RESTRICTED = "ERR_AGE_RESTRICTED"
    ERR_LOGIN_REQUIRED = "ERR_LOGIN_REQUIRED"
    ERR_NETWORK = "ERR_NETWORK"
    ERR_UNKNOWN = "ERR_UNKNOWN"


class DownloadError(Exception):
    """Custom exception for download failures."""
    def __init__(
        self,
        message,
        log_file=None,
        last_error=None,
        url=None,
        hint=None,
        error_code=None,
        next_steps=None,
    ):
        super().__init__(message)
        self.log_file = log_file
        self.last_error = last_error
        self.url = url
        self.hint = hint
        self.error_code = error_code  # Optional[ErrorClassification] enum value
        self.next_steps = next_steps


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
        Path to preferred executable if found, None otherwise.
    """
    runtimes = detect_js_runtimes()
    if "node" in runtimes:
        return runtimes["node"]
    if runtimes:
        return next(iter(runtimes.values()))
    return None


def detect_js_runtimes() -> Dict[str, str]:
    """Return available JS runtimes as {runtime_name: executable_path}."""
    runtimes: Dict[str, str] = {}
    for runtime in ("node", "deno", "bun"):
        exe = shutil.which(runtime)
        if exe:
            runtimes[runtime] = exe
    return runtimes


def _parse_cookies_from_browser(raw_value: str) -> Optional[Tuple[str, ...]]:
    """
    Parse YTDLP_COOKIES_FROM_BROWSER env var.
    Examples:
      chrome
      chrome:Default
      firefox:myprofile:keyring:container
    """
    raw = str(raw_value or "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(":") if p.strip()]
    if not parts:
        return None
    return tuple(parts[:4])


def classify_error(error_text: str, has_js_runtime: bool, log_text: str = "") -> Tuple[ErrorClassification, str, str]:
    """
    Classify download error and provide helpful guidance.
    
    Args:
        error_text: The error message text
        has_js_runtime: Whether JS runtime is available
        log_text: Optional full log text for better pattern matching
    
    Returns:
        Tuple of (ErrorClassification, hint, next_steps)
    """
    error_lower = str(error_text or "").lower()
    combined_text = f"{error_lower}\n{str(log_text or '').lower()}"
    
    # Check for specific error patterns
    if (
        "not available in your country" in combined_text
        or "geo" in combined_text
        or "region" in combined_text
        or "not available in your region" in combined_text
    ):
        return (
            ErrorClassification.ERR_GEO_BLOCK,
            "Video is geo-blocked or region-restricted",
            "1. This video cannot be accessed from your location\n"
            "2. Consider using a VPN (if permitted)\n"
            "3. Try a different video available in your region"
        )

    if "age" in combined_text or "sign in to confirm" in combined_text:
        return (
            ErrorClassification.ERR_AGE_RESTRICTED,
            "Video is age-restricted",
            "1. Video requires age verification\n"
            "2. Try authenticated download (cookies)\n"
            "3. Ensure your account has access to this content"
        )

    if "sign in" in combined_text or "login" in combined_text or "authenticate" in combined_text:
        return (
            ErrorClassification.ERR_LOGIN_REQUIRED,
            "Video requires authentication",
            "1. Video may be private or members-only\n"
            "2. Try authenticated download (cookies)\n"
            "3. Ensure you have access to this content"
        )

    if (
        "video unavailable" in combined_text
        or "this video is not available" in combined_text
        or "private video" in combined_text
        or "removed" in combined_text
        or "deleted" in combined_text
    ):
        return (
            ErrorClassification.ERR_VIDEO_UNAVAILABLE,
            "Video is unavailable, private, removed, or blocked for your account/region",
            "1. Verify the URL is correct\n"
            "2. Open the URL in your browser while signed in and confirm availability\n"
            "3. If it works only when signed in, configure cookies for yt-dlp\n"
            "4. Try a different video URL"
        )

    if (
        (not has_js_runtime and ("player" in combined_text or "signature" in combined_text))
        or "no supported javascript runtime" in combined_text
        or "javascript runtime" in combined_text
    ):
        return (
            ErrorClassification.ERR_JS_RUNTIME_MISSING,
            "JavaScript runtime support is limited for this video extraction path",
            "1. Local/dev: install Node.js from https://nodejs.org/\n"
            "2. Set JS runtime explicitly for yt-dlp if needed\n"
            "3. On hosted runtimes, try another video URL or authenticated cookies"
        )
    
    if "network" in combined_text or "connection" in combined_text or "timeout" in combined_text or "http" in combined_text:
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
    from rs_utils import verify_audio_file
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = out_dir / f"download_log_{timestamp}.txt"
    logger = YTDLPLogger(log_file)
    
    # Check JS runtimes
    js_runtimes = detect_js_runtimes()
    if js_runtimes:
        rt_text = ", ".join(f"{k}={v}" for k, v in js_runtimes.items())
        logger.info(f"JavaScript runtimes detected: {rt_text}")
    else:
        logger.warning("No JavaScript runtime found. Some videos may fail to download.")
        logger.warning("Consider installing Node.js for better compatibility.")
    
    # Format strategies to try in order
    format_strategies = [
        "bestaudio/best",
        "bestaudio[protocol!=m3u8]/bestaudio/best",
        "bestaudio[ext=m4a]/bestaudio",
        "worstaudio",
    ]

    # Extractor argument profiles to improve resilience for YouTube changes.
    extractor_profiles: List[Tuple[str, Optional[Dict]]] = [
        ("android+web", {"youtube": {"player_client": ["android", "web"]}}),
        ("ios+web", {"youtube": {"player_client": ["ios", "web"]}}),
        ("tv+web", {"youtube": {"player_client": ["tv", "web"]}}),
        ("default", None),
    ]
    
    last_error = None
    youtube_info = {}
    
    # Optional authentication inputs (env-based, no secrets committed in code).
    cookie_file = str(os.getenv("YTDLP_COOKIES_FILE", "") or "").strip()
    cookie_browser_raw = str(os.getenv("YTDLP_COOKIES_FROM_BROWSER", "") or "").strip()
    cookie_browser = _parse_cookies_from_browser(cookie_browser_raw)
    if cookie_file:
        if Path(cookie_file).exists():
            logger.info(f"Using cookies file from YTDLP_COOKIES_FILE: {cookie_file}")
        else:
            logger.warning(f"YTDLP_COOKIES_FILE is set but file does not exist: {cookie_file}")

    for profile_name, extractor_args in extractor_profiles:
        for strategy in format_strategies:
            try:
                logger.info(f"Attempting download with format: {strategy}")
                logger.info(f"URL: {url}")
                logger.info(f"Extractor strategy: {profile_name}")

                ydl_opts = {
                    "format": strategy,
                    "outtmpl": str(out_dir / "%(title)s.%(ext)s"),
                    "noplaylist": True,
                    "quiet": False,
                    "no_warnings": False,
                    "logger": logger,
                    "restrictfilenames": True,
                    "retries": 3,
                    "fragment_retries": 3,
                    "extractor_retries": 2,
                    "socket_timeout": 20,
                    "geo_bypass": True,
                    "postprocessors": [{
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }],
                }

                if extractor_args:
                    ydl_opts["extractor_args"] = extractor_args

                if js_runtimes:
                    ydl_opts["js_runtimes"] = dict(js_runtimes)

                if cookie_file and Path(cookie_file).exists():
                    ydl_opts["cookiefile"] = cookie_file
                elif cookie_browser:
                    ydl_opts["cookiesfrombrowser"] = cookie_browser
                    logger.info(f"Using cookies from browser: {cookie_browser}")

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
                logger.error(f"Strategy '{strategy}' with profile '{profile_name}' failed: {last_error}")
                continue
    
    # All strategies failed - classify the error
    error_msg = f"Download failed after trying all format strategies."
    logger.error(error_msg)
    logger.error(f"Last error: {last_error}")
    logger.error("")
    
    # Classify error and get structured guidance
    try:
        log_text = log_file.read_text(encoding="utf-8", errors="replace")
    except Exception:
        log_text = ""
    error_code, hint, next_steps = classify_error(
        str(last_error),
        bool(js_runtimes),
        log_text=log_text,
    )
    
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
        error_code=error_code,
        next_steps=next_steps,
    )
