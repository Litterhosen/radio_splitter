from config import DEFAULTS


def test_broadcast_defaults_enable_fail_soft_export():
    assert DEFAULTS["export_without_transcript"] is True


def test_broadcast_quick_scan_window_has_reasonable_default():
    assert 60.0 <= float(DEFAULTS["quick_scan_window_sec"]) <= 90.0
