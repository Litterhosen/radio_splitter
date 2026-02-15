"""
Test stress/UI performance with many hooks.
Measures memory and processing time with high topn value.
"""
import time
import tracemalloc
from pathlib import Path
from hook_finder import find_hooks


def run_test(song_path: Path, output_dir: Path):
    """
    Stress test find_hooks with parameters that produce many results.
    
    Args:
        song_path: Path to audio file
        output_dir: Directory to write results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "stress_ui_hooks.log"
    
    print("\n=== Testing Stress/UI Performance ===")
    print(f"Song file: {song_path.name}")
    
    log_lines = []
    all_passed = True
    
    try:
        # Start memory tracking
        tracemalloc.start()
        
        # Start timer
        start_time = time.time()
        
        # Run find_hooks with stress parameters
        hooks, global_bpm, global_confidence = find_hooks(
            str(song_path),
            hook_len_range=(2.0, 4.0),
            prefer_len=3.0,
            hop_s=0.5,
            topn=250,
            min_gap_s=0.5,
        )
        
        # Stop timer
        processing_time = time.time() - start_time
        
        # Get peak memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / (1024 * 1024)
        
        log_lines.append(f"Song file: {song_path.name}")
        log_lines.append(f"Hooks returned: {len(hooks)}")
        log_lines.append(f"Processing time: {processing_time:.2f}s")
        log_lines.append(f"Peak memory: {peak_mb:.2f} MB")
        log_lines.append(f"Global BPM: {global_bpm}")
        log_lines.append(f"Global confidence: {global_confidence:.3f}")
        
        print(f"  Hooks returned: {len(hooks)}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Peak memory: {peak_mb:.2f} MB")
        
        # Log first 5 and last 5 hooks
        log_lines.append("\n=== First 5 Hooks ===")
        for i, hook in enumerate(hooks[:5]):
            log_lines.append(
                f"Hook {i}: {hook.start:.2f}s - {hook.end:.2f}s "
                f"(dur: {hook.end - hook.start:.2f}s, score: {hook.score:.4f})"
            )
        
        if len(hooks) > 10:
            log_lines.append("\n=== Last 5 Hooks ===")
            for i, hook in enumerate(hooks[-5:], start=len(hooks) - 5):
                log_lines.append(
                    f"Hook {i}: {hook.start:.2f}s - {hook.end:.2f}s "
                    f"(dur: {hook.end - hook.start:.2f}s, score: {hook.score:.4f})"
                )
        
        # Assertions
        log_lines.append("\n=== Assertions ===")
        
        if len(hooks) > 0:
            log_lines.append("✓ PASS: Processing completed without crash")
            log_lines.append("✓ PASS: hooks returned > 0")
            print("  ✓ PASS: Processing completed without crash")
            print("  ✓ PASS: hooks returned > 0")
        else:
            log_lines.append("✗ FAIL: No hooks returned")
            print("  ✗ FAIL: No hooks returned")
            all_passed = False
        
    except Exception as e:
        log_lines.append(f"\n✗ EXCEPTION: {e}")
        print(f"  ✗ EXCEPTION: {e}")
        all_passed = False
        
        # Stop memory tracking if still active
        if tracemalloc.is_tracing():
            tracemalloc.stop()
    
    # Write log
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f"\nLog written to: {log_path}")
    
    return all_passed


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python test_stress_ui.py <song_path> <output_dir>")
        sys.exit(1)
    
    song_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    passed = run_test(song_path, output_dir)
    sys.exit(0 if passed else 1)
