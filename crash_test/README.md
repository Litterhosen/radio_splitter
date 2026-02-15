# Crash Test Harness

A comprehensive testing framework for radio_splitter v1.1.9 that validates core functionality through synthetic and real audio tests.

## Quick Start

Run the complete test suite with a single command:

```bash
# Using Python module
python -m crash_test.run_all

# Or using Make
make crash-test
```

All results will be written to `crash_test/results/<timestamp>/`

## Configuration

Edit `crash_test/config.yml` to provide real audio files (optional):

```yaml
# Path to a real MP3/WAV song (3-5 min ideal)
sample_song_path: ""

# Path to a 30+ min broadcast/podcast file
long_broadcast_path: ""

# Optional YouTube URL to test download (leave blank to skip live test)
youtube_test_url: ""
```

If no real files are provided, the harness automatically generates synthetic audio.

## Test Modules

### 1. test_env.py
Captures environment information:
- Python version
- ffmpeg/ffprobe versions
- yt-dlp, streamlit, librosa, numpy versions
- OS platform
- radio_splitter version

**Output**: `env.txt`

### 2. test_song_hunter_bars.py
Tests `find_hooks()` with different `prefer_bars` values (1, 2, 4, 8, 16):
- Validates that hook durations match expected bar-based calculations
- Checks global BPM estimation
- Verifies bar-based windowing logic

**Output**: `song_hunter_bars.csv` with all detected hooks

### 3. test_broadcast_longfile.py
Tests `detect_broadcast_segments()` on a 32-minute broadcast:
- Validates segment detection
- Checks chunking behavior for long files
- Measures processing time
- Verifies duration constraints

**Output**: 
- `broadcast_segments.csv` with all segments
- `broadcast_longfile.log` with metrics

### 4. test_youtube_errors.py
Tests YouTube error classification:
- Validates `classify_error()` function with simulated errors
- Tests detection of JS runtime issues, network errors, age restrictions, etc.
- Optional live YouTube download test if URL configured

**Output**: `youtube_error_classification.log`

### 5. test_stress_ui.py
Stress tests with high hook counts:
- Runs `find_hooks()` with `topn=250`, `hop_s=0.5`
- Measures peak memory usage with `tracemalloc`
- Measures processing time
- Validates performance under load

**Output**: `stress_ui_hooks.log`

### 6. test_transcript_presence.py
Documents expected behavior:
- Scans for transcript files after broadcast processing
- Documents that transcript files are not generated in broadcast mode
- Lists all artifacts produced

**Output**: `transcript_full_presence_check.log`

## Synthetic Audio

If real audio files are not provided, the harness generates:

- **synth_song_120bpm.wav** (3 min): Song with clear 120 BPM beat pattern
- **synth_broadcast_32min.wav** (32 min): Broadcast with alternating speech-like bursts and silence
- **synth_short_2s.wav** (2 sec): Short clip for edge cases
- **synth_corrupt.dat** (1 KB): Invalid audio file for error testing

## Report

After all tests complete, a comprehensive report is generated:

**CRASH_TEST_REPORT.md** includes:
- Environment information
- Test matrix with PASS/FAIL status
- Findings and observations
- Complete list of artifacts
- Summary of results

Example report structure:
```markdown
# CRASH TEST REPORT
**Radio Splitter v1.1.9**

**Date:** 2026-02-15 23:10:00 UTC
**Commit:** cc2bd67
**Test Run ID:** 2026-02-15_2310

## Environment
...

## Test Matrix
| Test Module | Status | Notes |
...

## Findings
...

## Artifacts Produced
...
```

## Requirements

All dependencies are in `requirements.txt`:
- numpy
- soundfile
- librosa
- streamlit
- yt-dlp
- etc.

System requirements:
- ffmpeg and ffprobe (for audio processing)
- Python 3.8+

## Directory Structure

```
crash_test/
├── __init__.py
├── __main__.py          # Entry point for python -m
├── config.yml           # User configuration
├── synth_audio.py       # Synthetic audio generator
├── test_env.py          # Environment capture
├── test_song_hunter_bars.py
├── test_broadcast_longfile.py
├── test_youtube_errors.py
├── test_stress_ui.py
├── test_transcript_presence.py
├── generate_report.py   # Report generator
├── run_all.py          # Main orchestrator
└── results/            # Test outputs (timestamped)
    └── 2026-02-15_2310/
        ├── CRASH_TEST_REPORT.md
        ├── env.txt
        ├── song_hunter_bars.csv
        ├── broadcast_segments.csv
        ├── *.log
        └── synth/
            └── *.wav
```

## Example Usage

### Basic run (with synthetic audio):
```bash
python -m crash_test.run_all
```

### With real audio files:
1. Edit `crash_test/config.yml`:
```yaml
sample_song_path: "/path/to/my_song.mp3"
long_broadcast_path: "/path/to/broadcast.mp3"
youtube_test_url: "https://youtube.com/watch?v=..."
```

2. Run tests:
```bash
make crash-test
```

### Review results:
```bash
cd crash_test/results/2026-02-15_2310/
cat CRASH_TEST_REPORT.md
```

## Continuous Integration

The crash test harness is designed for CI/CD integration:

```yaml
# Example CI job
- name: Run Crash Tests
  run: |
    pip install -r requirements.txt
    make crash-test
    
- name: Upload Results
  uses: actions/upload-artifact@v3
  with:
    name: crash-test-results
    path: crash_test/results/
```

## Exit Codes

- `0`: All tests completed (some may have documented expected failures)
- `1`: Critical error prevented test execution

Note: The harness continues running even if individual tests fail, to collect maximum diagnostic information.
