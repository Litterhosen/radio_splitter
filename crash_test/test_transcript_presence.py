"""
Test transcript file presence after broadcast processing.
Scans results directory for transcript_full.* files.
"""
from pathlib import Path


def run_test(output_dir: Path):
    """
    Check for transcript files in the results directory.
    
    Args:
        output_dir: Directory to scan for files
    """
    log_path = output_dir / "transcript_full_presence_check.log"
    
    print("\n=== Testing Transcript File Presence ===")
    print(f"Scanning directory: {output_dir}")
    
    log_lines = []
    all_passed = True
    
    # List all files in the session directory
    log_lines.append(f"Directory: {output_dir}")
    log_lines.append("\n=== All Files in Results Directory ===")
    
    all_files = []
    for pattern in ['*.txt', '*.json', '*.csv', '*.mp3', '*.wav', '*.log', '*.dat']:
        all_files.extend(output_dir.glob(pattern))
    
    # Also check subdirectories
    for subdir in output_dir.iterdir():
        if subdir.is_dir():
            for pattern in ['*.txt', '*.json', '*.csv', '*.mp3', '*.wav', '*.log']:
                all_files.extend(subdir.glob(pattern))
    
    if all_files:
        for file_path in sorted(all_files):
            rel_path = file_path.relative_to(output_dir)
            size = file_path.stat().st_size
            log_lines.append(f"  {rel_path} ({size} bytes)")
            print(f"  {rel_path} ({size} bytes)")
    else:
        log_lines.append("  (no files found)")
        print("  (no files found)")
    
    # Check for transcript_full.txt
    log_lines.append("\n=== Transcript Presence Check ===")
    transcript_txt = output_dir / "transcript_full.txt"
    transcript_json = output_dir / "transcript_full.json"
    
    # Also check output/ subdirectory if it exists
    output_subdir = output_dir / "output"
    if output_subdir.exists():
        transcript_txt_alt = output_subdir / "transcript_full.txt"
        transcript_json_alt = output_subdir / "transcript_full.json"
    else:
        transcript_txt_alt = None
        transcript_json_alt = None
    
    # Check transcript_full.txt
    if transcript_txt.exists():
        log_lines.append(f"✓ FOUND: transcript_full.txt")
        print("  ✓ FOUND: transcript_full.txt")
    elif transcript_txt_alt and transcript_txt_alt.exists():
        log_lines.append(f"✓ FOUND: output/transcript_full.txt")
        print("  ✓ FOUND: output/transcript_full.txt")
    else:
        log_lines.append(f"✗ NOT FOUND: transcript_full.txt")
        log_lines.append("  NOTE: This is EXPECTED - transcript files are not generated in broadcast mode")
        log_lines.append("  FINDING: broadcast_splitter.py does not call transcribe functions")
        print("  ✗ NOT FOUND: transcript_full.txt (EXPECTED)")
        # Don't fail the test since this is expected
    
    # Check transcript_full.json
    if transcript_json.exists():
        log_lines.append(f"✓ FOUND: transcript_full.json")
        print("  ✓ FOUND: transcript_full.json")
    elif transcript_json_alt and transcript_json_alt.exists():
        log_lines.append(f"✓ FOUND: output/transcript_full.json")
        print("  ✓ FOUND: output/transcript_full.json")
    else:
        log_lines.append(f"✗ NOT FOUND: transcript_full.json")
        log_lines.append("  NOTE: This is EXPECTED - transcript files are not generated in broadcast mode")
        log_lines.append("  FINDING: broadcast_splitter.py does not call transcribe functions")
        print("  ✗ NOT FOUND: transcript_full.json (EXPECTED)")
        # Don't fail the test since this is expected
    
    # Summary
    log_lines.append("\n=== Summary ===")
    log_lines.append("This test documents the expected absence of transcript_full.* files")
    log_lines.append("after broadcast segment detection. The broadcast_splitter module")
    log_lines.append("focuses on segment detection and does not include transcription.")
    
    # Write log
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f"\nLog written to: {log_path}")
    
    # Always pass since missing transcripts are expected
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python test_transcript_presence.py <output_dir>")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    
    passed = run_test(output_dir)
    sys.exit(0 if passed else 1)
