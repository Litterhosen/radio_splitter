"""
Main runner for crash test harness.
Orchestrates all test modules and generates the final report.
"""
import sys
import yaml
from pathlib import Path
from datetime import datetime


# Timestamp format for run_id
RUN_ID_FORMAT = "%Y-%m-%d_%H%M"


def load_config():
    """Load crash test configuration from config.yml."""
    config_path = Path(__file__).parent / "config.yml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config or {}
    except Exception as e:
        print(f"Warning: Could not load config.yml: {e}")
        return {}


def main():
    """Main entry point for crash test harness."""
    print("=" * 80)
    print("RADIO SPLITTER v1.1.9 - CRASH TEST HARNESS")
    print("=" * 80)
    
    # Create run_id from timestamp
    run_id = datetime.now().strftime(RUN_ID_FORMAT)
    
    # Create results directory
    results_base = Path(__file__).parent / "results"
    results_dir = results_base / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nRun ID: {run_id}")
    print(f"Results directory: {results_dir}")
    
    # Load configuration
    print("\nLoading configuration...")
    config = load_config()
    
    sample_song_path = config.get('sample_song_path', '').strip()
    long_broadcast_path = config.get('long_broadcast_path', '').strip()
    youtube_test_url = config.get('youtube_test_url', '').strip()
    
    # Determine which files to use
    use_real_song = bool(sample_song_path and Path(sample_song_path).exists())
    use_real_broadcast = bool(long_broadcast_path and Path(long_broadcast_path).exists())
    
    if use_real_song:
        print(f"Using real song file: {sample_song_path}")
        song_file = Path(sample_song_path)
    else:
        print("No real song file configured - will generate synthetic audio")
        song_file = None
    
    if use_real_broadcast:
        print(f"Using real broadcast file: {long_broadcast_path}")
        broadcast_file = Path(long_broadcast_path)
    else:
        print("No real broadcast file configured - will generate synthetic audio")
        broadcast_file = None
    
    if youtube_test_url:
        print(f"YouTube test URL configured: {youtube_test_url}")
    else:
        print("No YouTube test URL configured - will skip live download test")
    
    # Generate synthetic audio if needed
    synth_dir = results_dir / "synth"
    if not use_real_song or not use_real_broadcast:
        print("\n--- Generating Synthetic Audio ---")
        from crash_test import synth_audio
        synth_audio.generate_all_synthetic(synth_dir)
        
        if not use_real_song:
            song_file = synth_dir / "synth_song_120bpm.wav"
        if not use_real_broadcast:
            broadcast_file = synth_dir / "synth_broadcast_32min.wav"
    
    # Verify files exist
    if not song_file or not song_file.exists():
        print(f"\nERROR: Song file not found: {song_file}")
        return 1
    
    if not broadcast_file or not broadcast_file.exists():
        print(f"\nERROR: Broadcast file not found: {broadcast_file}")
        return 1
    
    # Track test results
    test_results = {}
    
    # Run test modules
    print("\n" + "=" * 80)
    print("RUNNING TEST MODULES")
    print("=" * 80)
    
    # Test 1: Environment
    print("\n[1/6] Capturing environment information...")
    try:
        from crash_test import test_env
        test_results['test_env'] = test_env.run_test(results_dir)
        print("✓ Environment capture completed")
    except Exception as e:
        print(f"✗ Environment capture failed: {e}")
        test_results['test_env'] = False
    
    # Test 2: Song Hunter Bars
    print("\n[2/6] Testing song hunter with different bar values...")
    try:
        from crash_test import test_song_hunter_bars
        test_results['test_song_hunter_bars'] = test_song_hunter_bars.run_test(
            song_file, results_dir
        )
        print("✓ Song hunter bars test completed")
    except Exception as e:
        print(f"✗ Song hunter bars test failed: {e}")
        test_results['test_song_hunter_bars'] = False
    
    # Test 3: Broadcast Longfile
    print("\n[3/6] Testing broadcast splitter with long file...")
    try:
        from crash_test import test_broadcast_longfile
        test_results['test_broadcast_longfile'] = test_broadcast_longfile.run_test(
            broadcast_file, results_dir
        )
        print("✓ Broadcast longfile test completed")
    except Exception as e:
        print(f"✗ Broadcast longfile test failed: {e}")
        test_results['test_broadcast_longfile'] = False
    
    # Test 4: YouTube Errors
    print("\n[4/6] Testing YouTube error classification...")
    try:
        from crash_test import test_youtube_errors
        test_results['test_youtube_errors'] = test_youtube_errors.run_test(
            youtube_test_url, results_dir
        )
        print("✓ YouTube errors test completed")
    except Exception as e:
        print(f"✗ YouTube errors test failed: {e}")
        test_results['test_youtube_errors'] = False
    
    # Test 5: Stress UI
    print("\n[5/6] Testing stress/UI performance...")
    try:
        from crash_test import test_stress_ui
        test_results['test_stress_ui'] = test_stress_ui.run_test(
            song_file, results_dir
        )
        print("✓ Stress UI test completed")
    except Exception as e:
        print(f"✗ Stress UI test failed: {e}")
        test_results['test_stress_ui'] = False
    
    # Test 6: Transcript Presence
    print("\n[6/6] Checking transcript file presence...")
    try:
        from crash_test import test_transcript_presence
        test_results['test_transcript_presence'] = test_transcript_presence.run_test(
            results_dir
        )
        print("✓ Transcript presence check completed")
    except Exception as e:
        print(f"✗ Transcript presence check failed: {e}")
        test_results['test_transcript_presence'] = False
    
    # Generate report
    print("\n" + "=" * 80)
    print("GENERATING REPORT")
    print("=" * 80)
    
    try:
        from crash_test import generate_report
        generate_report.generate_report(results_dir)
        print("\n✓ Report generation completed")
    except Exception as e:
        print(f"\n✗ Report generation failed: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("CRASH TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    print(f"\nTests passed: {passed}/{total}")
    print(f"\nAll artifacts written to: {results_dir}")
    print(f"\nView the full report at: {results_dir / 'CRASH_TEST_REPORT.md'}")
    
    # List key files
    print("\nKey artifacts:")
    key_files = [
        "CRASH_TEST_REPORT.md",
        "env.txt",
        "song_hunter_bars.csv",
        "broadcast_segments.csv",
        "broadcast_longfile.log",
        "youtube_error_classification.log",
        "stress_ui_hooks.log",
        "transcript_full_presence_check.log",
    ]
    
    for filename in key_files:
        file_path = results_dir / filename
        if file_path.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (not found)")
    
    print("\n" + "=" * 80)
    print("CRASH TEST COMPLETE")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
