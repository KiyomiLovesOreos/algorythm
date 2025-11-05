"""
algorythm.audio_loader - Load audio from files

This module provides tools for loading audio from various file formats
(MP3, WAV, OGG, FLAC) and applying visualizations.
"""

import numpy as np
import wave
import struct
from pathlib import Path
from typing import Optional, Tuple
import warnings

# Check for pydub availability at module level
try:
    import pydub
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False


def load_audio_wav(filepath: str) -> Tuple[np.ndarray, int]:
    """
    Load audio from WAV file.
    
    Args:
        filepath: Path to WAV file
        
    Returns:
        Tuple of (audio_signal, sample_rate)
    """
    with wave.open(filepath, 'rb') as wav_file:
        # Get parameters
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        n_frames = wav_file.getnframes()
        sample_width = wav_file.getsampwidth()
        
        # Read raw data
        raw_data = wav_file.readframes(n_frames)
        
        # Convert to numpy array based on sample width
        if sample_width == 1:  # 8-bit
            signal = np.frombuffer(raw_data, dtype=np.uint8)
            signal = (signal - 128) / 128.0  # Convert to -1 to 1
        elif sample_width == 2:  # 16-bit
            signal = np.frombuffer(raw_data, dtype=np.int16)
            signal = signal / 32768.0  # Convert to -1 to 1
        elif sample_width == 3:  # 24-bit
            # 24-bit requires special handling
            signal = np.zeros(n_frames * n_channels, dtype=np.float64)
            for i in range(n_frames * n_channels):
                byte_data = raw_data[i*3:(i+1)*3]
                # Convert 3 bytes to int
                value = int.from_bytes(byte_data, byteorder='little', signed=True)
                signal[i] = value / 8388608.0  # 2^23
        elif sample_width == 4:  # 32-bit
            signal = np.frombuffer(raw_data, dtype=np.int32)
            signal = signal / 2147483648.0  # 2^31
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        # Convert stereo to mono if needed
        if n_channels == 2:
            signal = signal.reshape(-1, 2).mean(axis=1)
        elif n_channels > 2:
            signal = signal.reshape(-1, n_channels).mean(axis=1)
        
        return signal, sample_rate


def load_audio_pydub(filepath: str) -> Tuple[np.ndarray, int]:
    """
    Load audio using pydub (supports MP3, OGG, FLAC, etc).
    
    Args:
        filepath: Path to audio file
        
    Returns:
        Tuple of (audio_signal, sample_rate)
    """
    if not HAS_PYDUB:
        raise ImportError(
            "pydub is required to load MP3/OGG/FLAC files.\n"
            "Install it with: pip install pydub\n"
            "Or reinstall algorythm with: pip install --upgrade algorythm"
        )
    
    try:
        from pydub import AudioSegment
    except ImportError as e:
        raise ImportError(
            f"Failed to import pydub: {e}\n"
            "Make sure pydub is properly installed: pip install pydub"
        )
    
    # Check if ffmpeg is available (needed for MP3/OGG/FLAC)
    try:
        from pydub.utils import which
        if not which("ffmpeg") and not which("avconv"):
            raise RuntimeError(
                "ffmpeg or avconv is required for MP3/OGG/FLAC support.\n"
                "Install ffmpeg:\n"
                "  Ubuntu/Debian: sudo apt install ffmpeg\n"
                "  macOS: brew install ffmpeg\n"
                "  Windows: Download from https://ffmpeg.org/"
            )
    except Exception as e:
        warnings.warn(f"Could not verify ffmpeg availability: {e}")
    
    # Load audio
    try:
        audio = AudioSegment.from_file(filepath)
    except FileNotFoundError as e:
        if "ffmpeg" in str(e).lower() or "avconv" in str(e).lower():
            raise RuntimeError(
                "ffmpeg is not installed or not in PATH.\n"
                "MP3/OGG/FLAC support requires ffmpeg.\n"
                "Install it:\n"
                "  Ubuntu/Debian: sudo apt install ffmpeg\n"
                "  macOS: brew install ffmpeg\n"
                "  Windows: Download from https://ffmpeg.org/"
            )
        raise
    
    # Convert to mono if needed
    if audio.channels > 1:
        audio = audio.set_channels(1)
    
    # Get sample rate
    sample_rate = audio.frame_rate
    
    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples())
    
    # Normalize to -1 to 1 range
    if audio.sample_width == 1:  # 8-bit
        signal = (samples - 128) / 128.0
    elif audio.sample_width == 2:  # 16-bit
        signal = samples / 32768.0
    elif audio.sample_width == 4:  # 32-bit
        signal = samples / 2147483648.0
    else:
        signal = samples / float(2 ** (8 * audio.sample_width - 1))
    
    return signal, sample_rate


def load_audio(
    filepath: str,
    target_sample_rate: Optional[int] = None,
    duration: Optional[float] = None,
    offset: float = 0.0
) -> Tuple[np.ndarray, int]:
    """
    Load audio from file (auto-detects format).
    
    Supports WAV, MP3, OGG, FLAC, and other formats.
    
    Args:
        filepath: Path to audio file
        target_sample_rate: Resample to this rate (optional)
        duration: Load only this many seconds (optional)
        offset: Start loading from this offset in seconds
        
    Returns:
        Tuple of (audio_signal, sample_rate)
        
    Examples:
        >>> # Load entire file
        >>> signal, sr = load_audio('song.mp3')
        
        >>> # Load first 30 seconds
        >>> signal, sr = load_audio('song.mp3', duration=30.0)
        
        >>> # Load 10 seconds starting at 1 minute
        >>> signal, sr = load_audio('song.mp3', offset=60.0, duration=10.0)
        
        >>> # Load and resample to 44.1kHz
        >>> signal, sr = load_audio('song.mp3', target_sample_rate=44100)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")
    
    # Detect format and load
    extension = filepath.suffix.lower()
    
    if extension == '.wav':
        signal, sample_rate = load_audio_wav(str(filepath))
    else:
        # Try pydub for other formats
        signal, sample_rate = load_audio_pydub(str(filepath))
    
    # Apply offset
    if offset > 0:
        offset_samples = int(offset * sample_rate)
        signal = signal[offset_samples:]
    
    # Apply duration limit
    if duration is not None:
        duration_samples = int(duration * sample_rate)
        signal = signal[:duration_samples]
    
    # Resample if needed
    if target_sample_rate is not None and target_sample_rate != sample_rate:
        signal = resample_audio(signal, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    
    return signal, sample_rate


def resample_audio(
    signal: np.ndarray,
    original_rate: int,
    target_rate: int
) -> np.ndarray:
    """
    Resample audio to a different sample rate.
    
    Uses linear interpolation for simplicity.
    
    Args:
        signal: Input audio signal
        original_rate: Original sample rate
        target_rate: Target sample rate
        
    Returns:
        Resampled audio signal
    """
    if original_rate == target_rate:
        return signal
    
    # Calculate new length
    duration = len(signal) / original_rate
    new_length = int(duration * target_rate)
    
    # Create time arrays
    original_times = np.arange(len(signal)) / original_rate
    target_times = np.arange(new_length) / target_rate
    
    # Interpolate
    resampled = np.interp(target_times, original_times, signal)
    
    return resampled


def visualize_audio_file(
    input_file: str,
    output_file: str,
    visualizer,
    video_width: int = 1920,
    video_height: int = 1080,
    video_fps: int = 30,
    offset: float = 0.0,
    duration: Optional[float] = None
):
    """
    Load an audio file and create a visualization video.
    
    This is a convenience function that loads audio and exports
    with visualization in one step.
    
    Performance Tips:
        - Use 1280x720 (720p) instead of 1920x1080 for 2-3x speedup
        - Use 24 fps instead of 30 fps for 20% speedup
        - CircularVisualizer and WaveformVisualizer are fastest
        - Process shorter segments with duration parameter for testing
    
    Args:
        input_file: Path to input audio file (MP3, WAV, OGG, FLAC)
        output_file: Path to output video file (MP4)
        visualizer: Visualizer instance to use
        video_width: Video width in pixels (default 1920, try 1280 for speed)
        video_height: Video height in pixels (default 1080, try 720 for speed)
        video_fps: Video frames per second (default 30, try 24 for speed)
        offset: Start offset in seconds
        duration: Duration to process in seconds (None = entire file)
        
    Examples:
        >>> from algorythm.audio_loader import visualize_audio_file
        >>> from algorythm.visualization import CircularVisualizer
        >>> 
        >>> # Fast rendering with lower resolution
        >>> viz = CircularVisualizer(sample_rate=44100, num_bars=64)
        >>> visualize_audio_file(
        ...     'my_song.mp3',
        ...     'my_song_video.mp4',
        ...     visualizer=viz,
        ...     video_width=1280,  # 720p for speed
        ...     video_height=720,
        ...     video_fps=24
        ... )
    """
    from algorythm.export import Exporter
    
    print(f"Loading audio from: {input_file}")
    signal, sample_rate = load_audio(
        input_file,
        target_sample_rate=44100,  # Standard rate
        offset=offset,
        duration=duration
    )
    
    duration_sec = len(signal)/sample_rate
    print(f"✓ Loaded {duration_sec:.2f}s of audio at {sample_rate}Hz")
    
    # Warn if file is long and high resolution
    est_frames = int(duration_sec * video_fps)
    if est_frames > 3000 and (video_width > 1280 or video_height > 720):
        print(f"⚠ Warning: Rendering {est_frames} frames at {video_width}x{video_height}")
        print(f"  This may take several minutes. Consider using 1280x720 for faster rendering.")
    
    # Update visualizer sample rate if needed
    if hasattr(visualizer, 'sample_rate'):
        visualizer.sample_rate = sample_rate
    
    # Export with visualization
    print(f"Creating visualization video...")
    print(f"  Resolution: {video_width}x{video_height} @ {video_fps}fps")
    print(f"  Estimated frames: {est_frames}")
    exporter = Exporter()
    exporter.export(
        signal,
        output_file,
        sample_rate=sample_rate,
        visualizer=visualizer,
        video_width=video_width,
        video_height=video_height,
        video_fps=video_fps
    )
    
    print(f"✓ Video created: {output_file}")


# Convenience class
class AudioFile:
    """
    Represents a loaded audio file.
    
    Examples:
        >>> audio = AudioFile('song.mp3')
        >>> print(f"Duration: {audio.duration:.2f}s")
        >>> print(f"Sample rate: {audio.sample_rate}Hz")
        >>> 
        >>> # Create visualization
        >>> from algorythm.visualization import WaveformVisualizer
        >>> viz = WaveformVisualizer(sample_rate=audio.sample_rate)
        >>> audio.visualize('output.mp4', visualizer=viz)
    """
    
    def __init__(
        self,
        filepath: str,
        target_sample_rate: Optional[int] = None,
        duration: Optional[float] = None,
        offset: float = 0.0
    ):
        """
        Load an audio file.
        
        Args:
            filepath: Path to audio file
            target_sample_rate: Resample to this rate (optional)
            duration: Load only this many seconds (optional)
            offset: Start loading from this offset in seconds
        """
        self.filepath = filepath
        self.signal, self.sample_rate = load_audio(
            filepath,
            target_sample_rate=target_sample_rate,
            duration=duration,
            offset=offset
        )
    
    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        return len(self.signal) / self.sample_rate
    
    @property
    def num_samples(self) -> int:
        """Get number of samples."""
        return len(self.signal)
    
    def visualize(
        self,
        output_file: str,
        visualizer,
        video_width: int = 1920,
        video_height: int = 1080,
        video_fps: int = 30
    ):
        """
        Create a visualization video from this audio.
        
        Args:
            output_file: Path to output MP4 file
            visualizer: Visualizer instance to use
            video_width: Video width in pixels
            video_height: Video height in pixels
            video_fps: Video frames per second
        """
        from algorythm.export import Exporter
        
        # Update visualizer sample rate
        if hasattr(visualizer, 'sample_rate'):
            visualizer.sample_rate = self.sample_rate
        
        # Export
        exporter = Exporter()
        exporter.export(
            self.signal,
            output_file,
            sample_rate=self.sample_rate,
            visualizer=visualizer,
            video_width=video_width,
            video_height=video_height,
            video_fps=video_fps
        )
    
    def __repr__(self):
        return f"AudioFile('{self.filepath}', duration={self.duration:.2f}s, sample_rate={self.sample_rate}Hz)"
