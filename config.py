"""
Configuration constants and defaults for the Sample Machine.
"""
import os
from pathlib import Path
from typing import Dict, List, Any

# Version number
VERSION: str = "1.1.16"

# Runtime/output directories (default to local AppData to avoid cloud-sync I/O overhead)
RUNTIME_ROOT: Path = Path(
    os.getenv(
        "RADIO_SPLITTER_RUNTIME_ROOT",
        str(Path.home() / "AppData" / "Local" / "radio_splitter2"),
    )
).resolve()
OUTPUT_ROOT: Path = Path(
    os.getenv("RADIO_SPLITTER_OUTPUT_ROOT", str(RUNTIME_ROOT / "output"))
).resolve()

# Anti-overlap and filter thresholds
OVERLAP_THRESHOLD: float = 0.30  # 30% overlap threshold for duplicate detection
MIN_DURATION_SECONDS: float = 4.0  # Hard floor for exported clips/loops
MIN_CLIP_DURATION_SECONDS: float = 4.0  # Minimum duration after refinement (shorter clips rejected)
DECAY_TAIL_DURATION: float = 0.75  # Extra audio tail for loops (seconds)
MAX_SLUG_LENGTH: int = 24  # Maximum characters for slug in filename

# Filename length constraints
MAX_FILENAME_LENGTH: int = 140  # Maximum total filename length (OS compatibility)
MAX_STEM_LENGTH: int = 130  # Reserve 10 chars for extension (_tail.mp3)

# Mode options for the application
MODE_OPTIONS: List[str] = [
    "🎵 Song Hunter (Loops)",
    "📻 Broadcast Hunter (Mix)",
]

# Bilingual theme keywords (DA + EN)
THEMES: Dict[str, List[str]] = {
    "THEME:TIME": ["tid", "evighed", "nu", "time", "eternity", "now"],
    "THEME:MEMORY": ["huske", "glemme", "remember", "forget", "back"],
    "THEME:DREAM": ["drøm", "natten", "dream", "night", "sleep"],
    "THEME:EXISTENTIAL": ["livet", "verden", "cirkel", "life", "world", "circle"],
    "THEME:META": ["radio", "musik", "stemme", "lyd", "music", "voice", "sound"],
}

# Default settings for the application
DEFAULTS: Dict[str, Any] = {
    "mode": MODE_OPTIONS[0],
    "model_size": "small",
    "whisper_language_ui": "Auto",
    "device": "cpu",
    "compute_type": "int8",
    "noise_db": -28.0,
    "silence_threshold_mode": "Auto (noise floor + margin)",
    "silence_margin_db": 10.0,
    "silence_quick_test_enabled": True,
    "silence_quick_test_seconds": 120.0,
    "silence_quick_test_retries": 3,
    "min_silence_s": 0.4,
    "pad_s": 0.15,
    "min_segment_s": 4.0,
    "max_segment_s": 45.0,
    "merge_gap_s": 0.35,
    "broadcast_chunk_s": 600.0,
    "broadcast_profile": "Balanced",
    "broadcast_split_method": "VAD-first (recommended)",
    "export_without_transcript": True,
    "quick_scan_window_sec": 75.0,
    "quick_scan_probe_segments": 8,
    "fixed_len": 8.0,
    "hook_len_range_min": 4.0,
    "hook_len_range_max": 15.0,
    "prefer_len": 8.0,
    "hook_hop": 1.0,
    "hook_topn": 12,
    "hook_gap": 2.0,
    "chorus_len_range_min": 30.0,
    "chorus_len_range_max": 45.0,
    "chorus_hop": 2.0,
    "chorus_topn": 4,
    "chorus_gap": 8.0,
    "loops_per_chorus": 10,
    "beat_refine": True,
    "beats_per_bar": 4,
    "prefer_bars": 2,
    "prefer_bars_ui": "2 bars",
    "try_both_bars": True,
    "use_slug": True,
    "slug_words": 6,
    "export_format": "mp3 (192k)",
    "downloaded_files": [],
}
