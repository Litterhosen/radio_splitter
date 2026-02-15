"""
Test environment capture.
Captures version information for all dependencies and tools.
"""
import sys
import platform
import subprocess
from pathlib import Path


def get_version_safe(module_name, attr_path="__version__"):
    """
    Safely get module version.
    
    Args:
        module_name: Name of module to import
        attr_path: Path to version attribute (e.g., "__version__" or "version.__version__")
    
    Returns:
        Version string or "N/A"
    """
    try:
        module = __import__(module_name)
        for attr in attr_path.split('.'):
            module = getattr(module, attr)
        return str(module)
    except Exception:
        return "N/A"


def get_command_version(command, version_flag="-version"):
    """
    Get version from command line tool.
    
    Args:
        command: Command to run
        version_flag: Flag to get version (default "-version")
    
    Returns:
        First line of output or "N/A"
    """
    try:
        result = subprocess.run(
            [command, version_flag],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Return first non-empty line
            for line in result.stdout.split('\n'):
                if line.strip():
                    return line.strip()
        return "N/A"
    except Exception:
        return "N/A"


def run_test(output_dir: Path):
    """
    Capture environment information.
    
    Args:
        output_dir: Directory to write env.txt
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    env_path = output_dir / "env.txt"
    
    print("\n=== Capturing Environment Information ===")
    
    lines = []
    
    # Python version
    python_version = sys.version
    lines.append(f"Python version: {python_version}")
    print(f"  Python: {python_version.split()[0]}")
    
    # FFmpeg version
    ffmpeg_version = get_command_version("ffmpeg", "-version")
    lines.append(f"ffmpeg version: {ffmpeg_version}")
    print(f"  ffmpeg: {ffmpeg_version[:50] if ffmpeg_version != 'N/A' else 'N/A'}")
    
    # FFprobe version
    ffprobe_version = get_command_version("ffprobe", "-version")
    lines.append(f"ffprobe version: {ffprobe_version}")
    print(f"  ffprobe: {ffprobe_version[:50] if ffprobe_version != 'N/A' else 'N/A'}")
    
    # yt-dlp version
    ytdlp_version = get_version_safe("yt_dlp.version", "__version__")
    lines.append(f"yt-dlp version: {ytdlp_version}")
    print(f"  yt-dlp: {ytdlp_version}")
    
    # streamlit version
    streamlit_version = get_version_safe("streamlit", "__version__")
    lines.append(f"streamlit version: {streamlit_version}")
    print(f"  streamlit: {streamlit_version}")
    
    # librosa version
    librosa_version = get_version_safe("librosa", "__version__")
    lines.append(f"librosa version: {librosa_version}")
    print(f"  librosa: {librosa_version}")
    
    # numpy version
    numpy_version = get_version_safe("numpy", "__version__")
    lines.append(f"numpy version: {numpy_version}")
    print(f"  numpy: {numpy_version}")
    
    # soundfile version
    soundfile_version = get_version_safe("soundfile", "__version__")
    lines.append(f"soundfile version: {soundfile_version}")
    print(f"  soundfile: {soundfile_version}")
    
    # OS platform
    os_platform = platform.platform()
    lines.append(f"OS platform: {os_platform}")
    print(f"  OS: {os_platform}")
    
    # radio_splitter version
    try:
        from config import VERSION
        lines.append(f"radio_splitter version: {VERSION}")
        print(f"  radio_splitter: {VERSION}")
    except Exception:
        lines.append("radio_splitter version: N/A")
        print("  radio_splitter: N/A")
    
    # Write to file
    with open(env_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\nEnvironment info written to: {env_path}")
    
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python test_env.py <output_dir>")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    
    passed = run_test(output_dir)
    sys.exit(0 if passed else 1)
