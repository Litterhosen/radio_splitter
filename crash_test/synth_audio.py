"""
Synthetic audio generator for crash tests.
Generates test audio files using numpy and soundfile.
"""
import numpy as np
import soundfile as sf
from pathlib import Path


def generate_song_120bpm(output_path: Path, duration_sec: float = 180.0):
    """
    Generate a 3-minute synthetic song with a clear beat at 120 BPM.
    Uses sine wave pulses + noise to create a rhythmic pattern.
    
    Args:
        output_path: Path to write the WAV file
        duration_sec: Duration in seconds (default 180 = 3 minutes)
    """
    sr = 22050  # Sample rate
    bpm = 120
    beats_per_second = bpm / 60.0
    samples_per_beat = int(sr / beats_per_second)
    
    total_samples = int(duration_sec * sr)
    audio = np.zeros(total_samples, dtype=np.float32)
    
    # Generate beat pulses
    beat_idx = 0
    while beat_idx * samples_per_beat < total_samples:
        # Strong beat every 4 beats (downbeat)
        if beat_idx % 4 == 0:
            freq = 440.0  # A4
            amplitude = 0.5
        else:
            freq = 330.0  # E4
            amplitude = 0.3
        
        # Create a short sine pulse for the beat
        pulse_duration = 0.1  # 100ms
        pulse_samples = int(pulse_duration * sr)
        t = np.linspace(0, pulse_duration, pulse_samples)
        pulse = amplitude * np.sin(2 * np.pi * freq * t)
        
        # Apply envelope (fade in/out)
        envelope = np.hanning(pulse_samples)
        pulse = pulse * envelope
        
        # Place pulse at beat position
        start = beat_idx * samples_per_beat
        end = min(start + pulse_samples, total_samples)
        audio[start:end] = pulse[:end - start]
        
        beat_idx += 1
    
    # Add some background noise for texture
    noise = np.random.normal(0, 0.05, total_samples).astype(np.float32)
    audio = audio + noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    # Write to file
    sf.write(output_path, audio, sr)
    print(f"Generated synthetic song: {output_path.name} ({duration_sec:.0f}s, {bpm} BPM)")


def generate_broadcast_32min(output_path: Path, duration_sec: float = 1920.0):
    """
    Generate a 32-minute synthetic broadcast with alternating speech-like
    noise bursts and silence gaps.
    
    Args:
        output_path: Path to write the WAV file
        duration_sec: Duration in seconds (default 1920 = 32 minutes)
    """
    sr = 22050
    audio = []
    
    elapsed = 0.0
    np.random.seed(42)  # For reproducibility
    
    while elapsed < duration_sec:
        # Speech-like burst duration: 0.5 to 8 seconds
        burst_dur = np.random.uniform(0.5, 8.0)
        burst_samples = int(burst_dur * sr)
        
        # Generate speech-like noise with varying amplitude
        burst = np.random.normal(0, 0.3, burst_samples).astype(np.float32)
        
        # Add some low-frequency modulation to simulate speech rhythm
        modulation_freq = np.random.uniform(2, 8)  # 2-8 Hz
        t = np.linspace(0, burst_dur, burst_samples)
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * modulation_freq * t)
        burst = burst * modulation
        
        audio.append(burst)
        elapsed += burst_dur
        
        if elapsed >= duration_sec:
            break
        
        # Silence gap: 0.3 to 2 seconds
        gap_dur = np.random.uniform(0.3, 2.0)
        gap_samples = int(gap_dur * sr)
        gap = np.zeros(gap_samples, dtype=np.float32)
        audio.append(gap)
        elapsed += gap_dur
    
    # Concatenate all segments
    audio = np.concatenate(audio)
    
    # Trim to exact duration
    audio = audio[:int(duration_sec * sr)]
    
    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    
    # Write to file
    sf.write(output_path, audio, sr)
    print(f"Generated synthetic broadcast: {output_path.name} ({duration_sec / 60:.1f}min)")


def generate_short_clip(output_path: Path, duration_sec: float = 2.0):
    """
    Generate a 2-second short clip for edge case testing.
    
    Args:
        output_path: Path to write the WAV file
        duration_sec: Duration in seconds (default 2.0)
    """
    sr = 22050
    total_samples = int(duration_sec * sr)
    
    # Simple sine wave
    freq = 440.0  # A4
    t = np.linspace(0, duration_sec, total_samples)
    audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    
    # Write to file
    sf.write(output_path, audio, sr)
    print(f"Generated short clip: {output_path.name} ({duration_sec:.1f}s)")


def generate_corrupt_file(output_path: Path, size_bytes: int = 1024):
    """
    Generate a corrupt file with random bytes (not valid audio).
    
    Args:
        output_path: Path to write the file
        size_bytes: Size in bytes (default 1024)
    """
    random_bytes = np.random.bytes(size_bytes)
    output_path.write_bytes(random_bytes)
    print(f"Generated corrupt file: {output_path.name} ({size_bytes} bytes)")


def generate_all_synthetic(output_dir: Path):
    """
    Generate all synthetic audio files for crash tests.
    
    Args:
        output_dir: Directory to write synthetic files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Generating Synthetic Audio Files ===")
    
    generate_song_120bpm(output_dir / "synth_song_120bpm.wav", duration_sec=180.0)
    generate_broadcast_32min(output_dir / "synth_broadcast_32min.wav", duration_sec=1920.0)
    generate_short_clip(output_dir / "synth_short_2s.wav", duration_sec=2.0)
    generate_corrupt_file(output_dir / "synth_corrupt.dat", size_bytes=1024)
    
    print("=== Synthetic Audio Generation Complete ===\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path("crash_test/results/test_synth")
    
    generate_all_synthetic(output_dir)
