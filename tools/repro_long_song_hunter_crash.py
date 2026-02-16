"""Deterministic reproduction for long-file Song Hunter crash path.

This script exercises the same code path used by Streamlit app.py:
app.py -> hook_finder.find_hooks(...) -> librosa.load(...)

It monkeypatches librosa.load to raise MemoryError, emulating long-duration
allocation failure observed in constrained runtime environments.
"""
from pathlib import Path
from unittest.mock import patch
import traceback
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hook_finder import find_hooks


def main() -> int:
    out_dir = Path("output") / "repro_long_song_hunter"
    out_dir.mkdir(parents=True, exist_ok=True)
    crash_log = out_dir / "crash.log"

    try:
        with patch("hook_finder.librosa.load", side_effect=MemoryError("Unable to allocate array for long audio buffer")):
            find_hooks(
                "synthetic_60min.wav",
                hook_len_range=(2.0, 8.0),
                prefer_len=4.0,
                hop_s=1.0,
                topn=3,
                min_gap_s=0.5,
                prefer_bars=2,
                beats_per_bar=4,
            )
    except Exception as e:
        payload = (
            "repro=long_song_hunter_memory\n"
            f"error={type(e).__name__}: {e}\n\n"
            f"traceback:\n{traceback.format_exc()}"
        )
        crash_log.write_text(payload, encoding="utf-8")
        print(f"Wrote crash log: {crash_log}")
        print(payload)
        return 0

    print("Repro did not fail; expected MemoryError path was not triggered")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
