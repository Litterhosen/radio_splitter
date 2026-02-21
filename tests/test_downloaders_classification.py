from unittest.mock import patch

from downloaders import (
    ErrorClassification,
    check_js_runtime,
    classify_error,
    detect_js_runtimes,
)


def test_detect_js_runtimes_and_check_prefers_node():
    with patch("downloaders.shutil.which") as mock_which:
        mock_which.side_effect = lambda name: {
            "node": "C:/bin/node.exe",
            "deno": "C:/bin/deno.exe",
            "bun": None,
        }.get(name)
        runtimes = detect_js_runtimes()
        assert runtimes["node"].endswith("node.exe")
        assert runtimes["deno"].endswith("deno.exe")
        assert check_js_runtime().endswith("node.exe")


def test_classify_video_unavailable_wins_over_js_warning():
    error_text = "ERROR: [youtube] abc123: Video unavailable. This video is not available"
    log_text = "No supported JavaScript runtime could be found"
    code, hint, steps = classify_error(error_text, has_js_runtime=False, log_text=log_text)
    assert code == ErrorClassification.ERR_VIDEO_UNAVAILABLE
    assert "unavailable" in hint.lower()
    assert "Verify the URL" in steps


def test_classify_js_runtime_missing_when_signature_error_without_runtime():
    code, hint, steps = classify_error(
        "player signature extraction failed",
        has_js_runtime=False,
        log_text="",
    )
    assert code == ErrorClassification.ERR_JS_RUNTIME_MISSING
    assert "runtime" in hint.lower()
    assert "Node.js" in steps
