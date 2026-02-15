"""
Test song_hunter with different prefer_bars values.
Validates that hook durations match expected bar-based calculations.
"""
import csv
from pathlib import Path
from hook_finder import find_hooks


# Tolerance for duration matching (seconds)
DURATION_TOLERANCE_SEC = 0.1


def run_test(song_path: Path, output_dir: Path):
    """
    Test find_hooks with different prefer_bars values.
    
    Args:
        song_path: Path to audio file
        output_dir: Directory to write results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "song_hunter_bars.csv"
    
    print("\n=== Testing Song Hunter with Different Bar Values ===")
    print(f"Song file: {song_path.name}")
    
    results = []
    all_passed = True
    
    # Test with different prefer_bars values
    for prefer_bars in [1, 2, 4, 8, 16]:
        print(f"\nTesting prefer_bars={prefer_bars}...")
        
        try:
            hooks, global_bpm, global_confidence = find_hooks(
                str(song_path),
                prefer_bars=prefer_bars,
                beats_per_bar=4,
                hook_len_range=(2.0, 120.0),  # Wide range to allow any bar duration
                hop_s=1.0,
                topn=12,
                min_gap_s=2.0,
            )
            
            # Calculate expected window duration
            bar_duration = (60.0 / global_bpm) * 4  # 4 beats per bar
            expected_win_sec = prefer_bars * bar_duration
            
            print(f"  Global BPM: {global_bpm} (confidence: {global_confidence:.3f})")
            print(f"  Expected window: {expected_win_sec:.2f}s ({prefer_bars} bars)")
            print(f"  Found {len(hooks)} hooks")
            
            # Check each hook
            test_passed = True
            for i, hook in enumerate(hooks):
                actual_dur_sec = hook.end - hook.start
                
                # Allow tolerance or equal to track duration
                duration_ok = abs(actual_dur_sec - expected_win_sec) <= DURATION_TOLERANCE_SEC
                
                if not duration_ok:
                    print(f"  ⚠ Hook {i}: duration {actual_dur_sec:.2f}s != expected {expected_win_sec:.2f}s")
                    test_passed = False
                
                results.append({
                    'prefer_bars': prefer_bars,
                    'hook_idx': i,
                    'global_bpm': global_bpm,
                    'global_confidence': f"{global_confidence:.3f}",
                    'expected_win_sec': f"{expected_win_sec:.3f}",
                    'actual_dur_sec': f"{actual_dur_sec:.3f}",
                    'start': f"{hook.start:.3f}",
                    'end': f"{hook.end:.3f}",
                    'score': f"{hook.score:.4f}",
                    'bpm_source': hook.bpm_source,
                    'bpm_confidence': f"{hook.bpm_confidence:.3f}",
                })
            
            if test_passed:
                print(f"  ✓ PASS: All hooks match expected duration (±0.1s)")
            else:
                print(f"  ✗ FAIL: Some hooks have incorrect duration")
                all_passed = False
                
        except Exception as e:
            print(f"  ✗ FAIL: Exception occurred: {e}")
            all_passed = False
            results.append({
                'prefer_bars': prefer_bars,
                'hook_idx': -1,
                'global_bpm': 0,
                'global_confidence': "0.000",
                'expected_win_sec': "0.000",
                'actual_dur_sec': "0.000",
                'start': "0.000",
                'end': "0.000",
                'score': "0.0000",
                'bpm_source': "ERROR",
                'bpm_confidence': "0.000",
            })
    
    # Write CSV
    if results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'prefer_bars', 'hook_idx', 'global_bpm', 'global_confidence',
                'expected_win_sec', 'actual_dur_sec', 'start', 'end', 'score',
                'bpm_source', 'bpm_confidence'
            ])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to: {csv_path}")
    
    return all_passed


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python test_song_hunter_bars.py <song_path> <output_dir>")
        sys.exit(1)
    
    song_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    passed = run_test(song_path, output_dir)
    sys.exit(0 if passed else 1)
