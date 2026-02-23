from pathlib import Path
import yt_dlp
import logging
import shutil
import subprocess
import os
import re
import base64
from datetime import datetime
from typing import Tuple, Optional, Dict, List, Any
from enum import Enum


class ErrorClassification(Enum):
    """Error classification for YouTube download failures."""
    ERR_JS_RUNTIME_MISSING = "ERR_JS_RUNTIME_MISSING"
    ERR_VIDEO_UNAVAILABLE = "ERR_VIDEO_UNAVAILABLE"
    ERR_GEO_BLOCK = "ERR_GEO_BLOCK"
    ERR_AGE_RESTRICTED = "ERR_AGE_RESTRICTED"
    ERR_LOGIN_REQUIRED = "ERR_LOGIN_REQUIRED"
    ERR_COOKIE_CONFIG = "ERR_COOKIE_CONFIG"
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
        node_path = runtimes["node"].get("path", "")
        return str(node_path) if node_path else None
    if runtimes:
        first_cfg = next(iter(runtimes.values()))
        first_path = first_cfg.get("path", "")
        return str(first_path) if first_path else None
    return None


def detect_js_runtimes() -> Dict[str, Dict[str, Any]]:
    """
    Return available JS runtimes in yt-dlp expected format:
    {runtime_name: {config}} where config may contain "path".
    """
    runtimes: Dict[str, Dict[str, Any]] = {}
    for runtime in ("node", "deno", "bun"):
        exe = shutil.which(runtime)
        if exe:
            runtimes[runtime] = {"path": exe}
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


def _write_inline_cookies_if_configured(out_dir: Path, logger: YTDLPLogger) -> Optional[str]:
    """
    Support inline cookie injection from env vars for hosted deployments.
    Env options:
      - YTDLP_COOKIES_CONTENT (raw Netscape cookie text)
      - YTDLP_COOKIES_B64 (base64-encoded Netscape cookie text)
    """
    raw_text = str(os.getenv("YTDLP_COOKIES_CONTENT", "") or "").strip()
    raw_b64 = str(os.getenv("YTDLP_COOKIES_B64", "") or "").strip()
    cookie_text = raw_text

    if not cookie_text and raw_b64:
        try:
            cookie_text = base64.b64decode(raw_b64).decode("utf-8", errors="replace").strip()
        except Exception as e:
            logger.warning(
                "YTDLP_COOKIES_B64 is set but could not be decoded: "
                f"{type(e).__name__}: {str(e)[:120]}"
            )
            cookie_text = ""

    if not cookie_text:
        return None

    cookie_path = out_dir / "_cookies_runtime.txt"
    cookie_path.write_text(cookie_text + "\n", encoding="utf-8")
    logger.info("Using inline cookies from environment (YTDLP_COOKIES_CONTENT/YTDLP_COOKIES_B64).")
    return str(cookie_path)


def _resolve_auth_inputs(
    out_dir: Path,
    logger: YTDLPLogger,
) -> Tuple[str, str, Optional[Tuple[str, ...]], str, Optional[str]]:
    """
    Resolve cookie/auth-related options from environment.

    Returns:
        Tuple of (cookie_file, cookie_file_source, cookie_browser, geo_bypass_country, inline_cookie_file)
    """
    cookie_file = str(os.getenv("YTDLP_COOKIES_FILE", "") or "").strip()
    cookie_file_source = "YTDLP_COOKIES_FILE"
    inline_cookie_file = _write_inline_cookies_if_configured(out_dir, logger)
    if not cookie_file and inline_cookie_file:
        cookie_file = inline_cookie_file
        cookie_file_source = "inline env cookies"

    cookie_browser_raw = str(os.getenv("YTDLP_COOKIES_FROM_BROWSER", "") or "").strip()
    cookie_browser = _parse_cookies_from_browser(cookie_browser_raw)
    geo_bypass_country = str(os.getenv("YTDLP_GEO_BYPASS_COUNTRY", "") or "").strip().upper()

    if cookie_file:
        if Path(cookie_file).exists():
            logger.info(f"Using cookies file from {cookie_file_source}: {cookie_file}")
        else:
            logger.warning(f"Cookies file is configured but does not exist: {cookie_file}")

    if cookie_browser:
        logger.info(f"Using cookies from browser config: {cookie_browser}")

    if geo_bypass_country:
        logger.info(f"Using geo bypass country: {geo_bypass_country}")

    raw_b64 = str(os.getenv("YTDLP_COOKIES_B64", "") or "").strip()
    raw_text = str(os.getenv("YTDLP_COOKIES_CONTENT", "") or "").strip()
    if raw_b64 and not raw_text and not inline_cookie_file:
        logger.error(
            "Invalid YTDLP_COOKIES_B64 configuration. "
            "Use plain base64 of a Netscape cookies.txt file (no line breaks, no extra quoting)."
        )

    return cookie_file, cookie_file_source, cookie_browser, geo_bypass_country, inline_cookie_file


def _summarize_cookie_file(cookie_path: Path) -> Dict[str, Any]:
    """
    Parse Netscape cookie file and return non-sensitive health summary.
    """
    now_ts = int(datetime.now().timestamp())
    summary: Dict[str, Any] = {
        "exists": cookie_path.exists(),
        "path": str(cookie_path),
        "total_cookie_rows": 0,
        "parsed_cookie_rows": 0,
        "parse_errors": 0,
        "youtube_google_rows": 0,
        "youtube_google_active_rows": 0,
    }

    if not cookie_path.exists():
        return summary

    try:
        lines = cookie_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        summary["read_error"] = True
        return summary

    for line in lines:
        row = line.strip()
        if not row or row.startswith("#"):
            continue
        summary["total_cookie_rows"] += 1
        parts = row.split("\t")
        if len(parts) < 7:
            summary["parse_errors"] += 1
            continue

        summary["parsed_cookie_rows"] += 1
        domain = parts[0].lower()
        expires_raw = parts[4].strip()
        is_relevant = (
            "youtube.com" in domain
            or "google.com" in domain
            or "googlevideo.com" in domain
            or "ytimg.com" in domain
        )
        if not is_relevant:
            continue

        summary["youtube_google_rows"] += 1

        try:
            expires_ts = int(float(expires_raw))
        except Exception:
            # Unknown expiry format - treat as potentially active
            expires_ts = now_ts

        # Netscape format uses 0 for session cookies.
        if expires_ts == 0 or expires_ts >= now_ts:
            summary["youtube_google_active_rows"] += 1

    return summary


def check_cookie_health(
    out_dir: Path,
    probe_url: str = "https://www.youtube.com/feed/history",
) -> Dict[str, Any]:
    """
    Validate yt-dlp cookie setup and run an authenticated probe.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = out_dir / f"cookie_health_{timestamp}.txt"
    logger = YTDLPLogger(log_file)

    result: Dict[str, Any] = {
        "ok": False,
        "summary": "Cookie health check did not complete.",
        "log_file": str(log_file),
        "probe_url": probe_url,
        "auth_source": "none",
        "configured": False,
    }

    try:
        js_runtimes = detect_js_runtimes()
        result["js_runtimes"] = sorted(js_runtimes.keys())
        if js_runtimes:
            rt_text = ", ".join(f"{k}={cfg.get('path', '')}" for k, cfg in js_runtimes.items())
            logger.info(f"JavaScript runtimes detected: {rt_text}")
        else:
            logger.warning("No JavaScript runtime found.")

        cookie_file, cookie_file_source, cookie_browser, geo_bypass_country, inline_cookie_file = _resolve_auth_inputs(
            out_dir,
            logger,
        )
        result["geo_bypass_country"] = geo_bypass_country

        if cookie_file and Path(cookie_file).exists():
            result["configured"] = True
            result["auth_source"] = cookie_file_source
            result["cookie_file"] = cookie_file
            cookie_summary = _summarize_cookie_file(Path(cookie_file))
            result["cookie_file_summary"] = cookie_summary

            if cookie_summary.get("youtube_google_rows", 0) == 0:
                result["summary"] = "Cookies file has no YouTube/Google cookies."
                return result

            if cookie_summary.get("youtube_google_active_rows", 0) == 0:
                result["summary"] = "Cookies file has no active YouTube/Google cookies (possibly expired)."
                return result
        elif cookie_browser:
            result["configured"] = True
            result["auth_source"] = "YTDLP_COOKIES_FROM_BROWSER"
            result["cookies_from_browser"] = ":".join(cookie_browser)
        else:
            raw_b64 = str(os.getenv("YTDLP_COOKIES_B64", "") or "").strip()
            if raw_b64 and not inline_cookie_file:
                result["summary"] = (
                    "YTDLP_COOKIES_B64 is set but invalid (cannot decode). Regenerate base64 from cookies.txt."
                )
            else:
                result["summary"] = (
                    "No cookie configuration found. Set YTDLP_COOKIES_B64 or YTDLP_COOKIES_FILE."
                )
            return result

        ydl_opts = {
            "skip_download": True,
            "simulate": True,
            "noplaylist": True,
            "quiet": False,
            "no_warnings": False,
            "logger": logger,
            "retries": 1,
            "extractor_retries": 1,
            "socket_timeout": 20,
            "geo_bypass": True,
        }
        if js_runtimes:
            ydl_opts["js_runtimes"] = dict(js_runtimes)
        if geo_bypass_country:
            ydl_opts["geo_bypass_country"] = geo_bypass_country
        if cookie_file and Path(cookie_file).exists():
            ydl_opts["cookiefile"] = cookie_file
        elif cookie_browser:
            ydl_opts["cookiesfrombrowser"] = cookie_browser

        logger.info(f"Running authenticated probe URL: {probe_url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(probe_url, download=False)

        probe_title = ""
        if isinstance(info, dict):
            probe_title = str(info.get("title", "") or "").strip()
        result["probe_title"] = probe_title
        result["ok"] = True
        result["summary"] = "Cookie check passed and authenticated probe succeeded."
        return result

    except Exception as e:
        err_text = str(e)
        result["probe_error"] = err_text
        error_code, hint, next_steps = classify_error(err_text, bool(result.get("js_runtimes")))
        result["error_code"] = error_code.value
        result["hint"] = hint
        result["next_steps"] = next_steps
        result["summary"] = f"Probe failed: {hint}"
        logger.error(f"Cookie health probe failed: {err_text}")
        return result
    finally:
        logger.close()


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

    if "ytdlp_cookies_b64 is set but could not be decoded" in combined_text:
        return (
            ErrorClassification.ERR_COOKIE_CONFIG,
            "Cookie configuration is invalid (YTDLP_COOKIES_B64 cannot be decoded)",
            "1. Regenerate base64 from a valid Netscape cookies.txt file\n"
            "2. Paste full base64 string into YTDLP_COOKIES_B64\n"
            "3. Keep only one cookie method enabled (B64 or file)\n"
            "4. Re-run cookie health check"
        )

    # Check for specific error patterns
    if (
        "not available in your country" in combined_text
        or "not available in your region" in combined_text
        or "geo-restricted" in combined_text
        or "geo restricted" in combined_text
        or "geoblocked" in combined_text
    ):
        return (
            ErrorClassification.ERR_GEO_BLOCK,
            "Video is geo-blocked or region-restricted",
            "1. This video cannot be accessed from your location\n"
            "2. Consider using a VPN (if permitted)\n"
            "3. Try a different video available in your region"
        )

    if (
        re.search(r"\bage[- ]?restricted\b", combined_text)
        or "confirm your age" in combined_text
        or "verify your age" in combined_text
        or "age verification" in combined_text
    ):
        return (
            ErrorClassification.ERR_AGE_RESTRICTED,
            "Video is age-restricted",
            "1. Video requires age verification\n"
            "2. Try authenticated download (cookies)\n"
            "3. Ensure your account has access to this content"
        )

    if (
        "sign in to confirm you" in combined_text
        or "login required" in combined_text
        or "authentication required" in combined_text
        or "members-only" in combined_text
        or "private video" in combined_text
        or "authenticate" in combined_text
    ):
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
            "Video is unavailable from this runtime/location (or requires auth)",
            "1. Verify the URL is correct\n"
            "2. Open the URL in your browser while signed in and confirm availability\n"
            "3. If it works only when signed in, configure cookies for yt-dlp\n"
            "4. Optional: set YTDLP_GEO_BYPASS_COUNTRY (e.g. US) and retry\n"
            "5. Try a different video URL"
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
        rt_text = ", ".join(f"{k}={cfg.get('path', '')}" for k, cfg in js_runtimes.items())
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
        ("web_safari", {"youtube": {"player_client": ["web_safari"]}}),
        ("android_vr", {"youtube": {"player_client": ["android_vr"]}}),
        ("default", {"youtube": {"player_client": ["default"]}}),
    ]
    
    last_error = None
    youtube_info = {}
    
    # Optional authentication inputs (env-based, no secrets committed in code).
    cookie_file, _, cookie_browser, geo_bypass_country, _ = _resolve_auth_inputs(out_dir, logger)

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

                if geo_bypass_country:
                    ydl_opts["geo_bypass_country"] = geo_bypass_country
                    logger.info(f"Using geo bypass country: {geo_bypass_country}")

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
