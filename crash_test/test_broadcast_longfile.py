"""
Test broadcast_splitter with a long file.
Validates segment detection and chunking behavior.
"""
import csv
import time
from pathlib import Path
from broadcast_splitter import detect_broadcast_segments


def run_test(broadcast_path: Path, output_dir: Path):
    """
    Test detect_broadcast_segments on a long broadcast file.
    
    Args:
        broadcast_path: Path to broadcast audio file
        output_dir: Directory to write results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "broadcast_segments.csv"
    log_path = output_dir / "broadcast_longfile.log"
    
    print("\n=== Testing Broadcast Splitter with Long File ===")
    print(f"Broadcast file: {broadcast_path.name}")
    
    log_lines = []
    all_passed = True
    
    try:
        # Run broadcast detection
        start_time = time.time()
        segments, method_used, chunking_enabled = detect_broadcast_segments(
            broadcast_path,
            min_segment_sec=1.5,
            max_segment_sec=45.0,
            merge_gap_sec=0.35,
            chunk_sec=600.0,
            prefer_method="vad",
        )
        processing_time = time.time() - start_time
        
        log_lines.append(f"Broadcast file: {broadcast_path.name}")
        log_lines.append(f"Processing time: {processing_time:.2f}s")
        log_lines.append(f"Method used: {method_used}")
        log_lines.append(f"Chunking enabled: {chunking_enabled}")
        log_lines.append(f"Total segments: {len(segments)}")
        
        print(f"  Method: {method_used}")
        print(f"  Chunking: {chunking_enabled}")
        print(f"  Segments: {len(segments)}")
        print(f"  Processing time: {processing_time:.2f}s")
        
        # Calculate segment statistics
        if segments:
            durations = [end - start for start, end in segments]
            min_dur = min(durations)
            max_dur = max(durations)
            mean_dur = sum(durations) / len(durations)
            
            log_lines.append(f"Min segment duration: {min_dur:.2f}s")
            log_lines.append(f"Max segment duration: {max_dur:.2f}s")
            log_lines.append(f"Mean segment duration: {mean_dur:.2f}s")
            
            print(f"  Duration range: {min_dur:.2f}s - {max_dur:.2f}s (mean: {mean_dur:.2f}s)")
            
            # Write segments CSV
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['segment_idx', 'start_sec', 'end_sec', 'duration_sec'])
                for idx, (start, end) in enumerate(segments):
                    writer.writerow([idx, f"{start:.3f}", f"{end:.3f}", f"{end - start:.3f}"])
            
            print(f"  Segments written to: {csv_path.name}")
        
        # Assertions
        log_lines.append("\n=== Assertions ===")
        
        # Check segment count
        if len(segments) > 0:
            log_lines.append("✓ PASS: segment_count > 0")
            print("  ✓ PASS: segment_count > 0")
        else:
            log_lines.append("✗ FAIL: segment_count == 0")
            print("  ✗ FAIL: segment_count == 0")
            all_passed = False
        
        # Check duration constraints
        if segments:
            all_valid = all(1.5 <= (end - start) <= 45.0 for start, end in segments)
            if all_valid:
                log_lines.append("✓ PASS: All durations within min_segment_sec and max_segment_sec")
                print("  ✓ PASS: All durations within constraints")
            else:
                log_lines.append("✗ FAIL: Some durations outside min/max constraints")
                print("  ✗ FAIL: Some durations outside constraints")
                all_passed = False
        
        # Check chunking for long files (>=30 min = 1800s)
        # Our test file is 32 min = 1920s, so chunking should be enabled
        from audio_split import get_duration_seconds
        duration = get_duration_seconds(broadcast_path)
        
        if duration >= 1800:  # 30 minutes
            if chunking_enabled:
                log_lines.append(f"✓ PASS: chunking_enabled == True for {duration/60:.1f}min file")
                print(f"  ✓ PASS: chunking_enabled == True for {duration/60:.1f}min file")
            else:
                log_lines.append(f"✗ FAIL: chunking_enabled == False for {duration/60:.1f}min file")
                print(f"  ✗ FAIL: chunking_enabled == False for long file")
                all_passed = False
        
        # Check for transcript files (expected to FAIL based on static analysis)
        log_lines.append("\n=== Transcript Presence Check ===")
        transcript_txt = output_dir / "transcript_full.txt"
        transcript_json = output_dir / "transcript_full.json"
        
        if transcript_txt.exists():
            log_lines.append(f"✓ FOUND: {transcript_txt.name}")
            print(f"  ✓ FOUND: {transcript_txt.name}")
        else:
            log_lines.append(f"✗ NOT FOUND: {transcript_txt.name} (EXPECTED - this is a known gap)")
            print(f"  ✗ NOT FOUND: {transcript_txt.name} (expected)")
        
        if transcript_json.exists():
            log_lines.append(f"✓ FOUND: {transcript_json.name}")
            print(f"  ✓ FOUND: {transcript_json.name}")
        else:
            log_lines.append(f"✗ NOT FOUND: {transcript_json.name} (EXPECTED - this is a known gap)")
            print(f"  ✗ NOT FOUND: {transcript_json.name} (expected)")
        
    except Exception as e:
        log_lines.append(f"\n✗ EXCEPTION: {e}")
        print(f"  ✗ EXCEPTION: {e}")
        all_passed = False
    
    # Write log
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f"\nLog written to: {log_path}")
    
    return all_passed


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python test_broadcast_longfile.py <broadcast_path> <output_dir>")
        sys.exit(1)
    
    broadcast_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    passed = run_test(broadcast_path, output_dir)
    sys.exit(0 if passed else 1)
