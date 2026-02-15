"""
Test YouTube error classification.
Validates the classify_error function and optional live download test.
"""
from pathlib import Path
from downloaders import classify_error, ErrorClassification, check_js_runtime


def run_test(youtube_url: str, output_dir: Path):
    """
    Test YouTube error classification.
    
    Args:
        youtube_url: Optional YouTube URL for live test (empty string to skip)
        output_dir: Directory to write results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "youtube_error_classification.log"
    
    print("\n=== Testing YouTube Error Classification ===")
    
    log_lines = []
    all_passed = True
    
    # Check for JS runtime
    js_runtime = check_js_runtime()
    has_js = js_runtime is not None
    
    log_lines.append(f"JavaScript runtime: {'Found' if has_js else 'Not found'}")
    if has_js:
        log_lines.append(f"  Path: {js_runtime}")
    print(f"JavaScript runtime: {'Found' if has_js else 'Not found'}")
    
    # Test error classification with simulated errors
    test_cases = [
        (
            "ERROR: No supported JavaScript runtime found",
            ErrorClassification.ERR_JS_RUNTIME_MISSING,
            "JS runtime missing error"
        ),
        (
            "ERROR: Video unavailable",
            ErrorClassification.ERR_VIDEO_UNAVAILABLE,
            "Video unavailable error"
        ),
        (
            "ERROR: network timeout occurred",
            ErrorClassification.ERR_NETWORK,
            "Network error"
        ),
        (
            "Sign in to confirm you're not a bot and verify your age",
            ErrorClassification.ERR_AGE_RESTRICTED,
            "Age-restricted error"
        ),
        (
            "ERROR: Some completely unknown error",
            ErrorClassification.ERR_UNKNOWN,
            "Unknown error"
        ),
    ]
    
    log_lines.append("\n=== Simulated Error Classification Tests ===")
    print("\nSimulated error tests:")
    
    for error_text, expected_code, description in test_cases:
        result_code, hint, next_steps = classify_error(error_text, has_js)
        
        if result_code == expected_code:
            log_lines.append(f"✓ PASS: {description}")
            log_lines.append(f"  Error text: {error_text[:60]}...")
            log_lines.append(f"  Expected: {expected_code.value}")
            log_lines.append(f"  Got: {result_code.value}")
            log_lines.append(f"  Hint: {hint}")
            print(f"  ✓ PASS: {description} -> {result_code.value}")
        else:
            log_lines.append(f"✗ FAIL: {description}")
            log_lines.append(f"  Error text: {error_text[:60]}...")
            log_lines.append(f"  Expected: {expected_code.value}")
            log_lines.append(f"  Got: {result_code.value}")
            log_lines.append(f"  Hint: {hint}")
            print(f"  ✗ FAIL: {description} (expected {expected_code.value}, got {result_code.value})")
            all_passed = False
    
    # Optional live download test
    if youtube_url and youtube_url.strip():
        log_lines.append("\n=== Live YouTube Download Test ===")
        log_lines.append(f"URL: {youtube_url}")
        print(f"\nLive download test: {youtube_url}")
        
        try:
            from downloaders import download_audio
            from tempfile import TemporaryDirectory
            
            with TemporaryDirectory() as tmpdir:
                file_path, metadata = download_audio(youtube_url, tmpdir)
                log_lines.append(f"✓ SUCCESS: Download completed")
                log_lines.append(f"  File: {file_path.name}")
                log_lines.append(f"  Title: {metadata.get('title', 'N/A')}")
                log_lines.append(f"  Uploader: {metadata.get('uploader', 'N/A')}")
                print(f"  ✓ SUCCESS: Downloaded {file_path.name}")
        except Exception as e:
            from downloaders import DownloadError
            if isinstance(e, DownloadError) and e.error_code:
                log_lines.append(f"✗ FAILED: Download error")
                log_lines.append(f"  Error code: {e.error_code.value}")
                log_lines.append(f"  Hint: {e.hint}")
                log_lines.append(f"  Message: {str(e)[:100]}...")
                print(f"  ✗ FAILED: {e.error_code.value}")
            else:
                log_lines.append(f"✗ FAILED: {str(e)[:100]}...")
                print(f"  ✗ FAILED: {str(e)[:60]}...")
    else:
        log_lines.append("\n=== Live YouTube Download Test ===")
        log_lines.append("SKIPPED: No YouTube URL configured")
        print("\nLive download test: SKIPPED (no URL configured)")
    
    # Write log
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f"\nLog written to: {log_path}")
    
    return all_passed


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python test_youtube_errors.py <output_dir> [youtube_url]")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    youtube_url = sys.argv[2] if len(sys.argv) > 2 else ""
    
    passed = run_test(youtube_url, output_dir)
    sys.exit(0 if passed else 1)
