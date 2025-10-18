"""
algorythm.visualization - Synchronized visualization tools

This module provides tools for generating visual output alongside audio,
including waveforms, spectrograms, and frequency scopes.
"""

from typing import Optional, Literal, Tuple
import numpy as np


class Visualizer:
    """
    Base class for audio visualization.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize a visualizer.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
    
    def generate(self, signal: np.ndarray) -> np.ndarray:
        """
        Generate visualization data from audio signal.
        
        Args:
            signal: Audio signal
            
        Returns:
            Visualization data
        """
        raise NotImplementedError("Subclasses must implement generate()")


class WaveformVisualizer(Visualizer):
    """
    Real-time waveform visualization.
    
    Displays amplitude over time.
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        window_size: int = 1024,
        downsample_factor: int = 1
    ):
        """
        Initialize a waveform visualizer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            window_size: Size of visualization window
            downsample_factor: Factor to downsample signal for display
        """
        super().__init__(sample_rate)
        self.window_size = window_size
        self.downsample_factor = downsample_factor
    
    def generate(self, signal: np.ndarray) -> np.ndarray:
        """
        Generate waveform visualization data.
        
        Args:
            signal: Audio signal
            
        Returns:
            Waveform data (amplitude values)
        """
        # Downsample if needed
        if self.downsample_factor > 1:
            signal = signal[::self.downsample_factor]
        
        # Split into windows for visualization frames
        num_windows = len(signal) // self.window_size
        if num_windows == 0:
            return signal
        
        # Reshape into windows
        trimmed_length = num_windows * self.window_size
        windowed = signal[:trimmed_length].reshape(num_windows, self.window_size)
        
        return windowed
    
    def to_image_data(
        self,
        signal: np.ndarray,
        height: int = 256,
        width: int = 1024
    ) -> np.ndarray:
        """
        Convert waveform to image data.
        
        Args:
            signal: Audio signal
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Image data as 2D array (height x width)
        """
        # Downsample to match width
        if len(signal) > width:
            indices = np.linspace(0, len(signal) - 1, width).astype(int)
            signal = signal[indices]
        elif len(signal) < width:
            signal = np.pad(signal, (0, width - len(signal)))
        
        # Create image data
        image = np.zeros((height, width))
        
        # Map signal to image coordinates
        center = height // 2
        for i, sample in enumerate(signal):
            y = int(center - sample * center)
            y = np.clip(y, 0, height - 1)
            image[y, i] = 1.0
        
        return image


class SpectrogramVisualizer(Visualizer):
    """
    Spectrogram visualization.
    
    Displays frequency content over time using Short-Time Fourier Transform (STFT).
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        window_size: int = 2048,
        hop_size: int = 512,
        window_type: Literal['hann', 'hamming', 'blackman'] = 'hann'
    ):
        """
        Initialize a spectrogram visualizer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            window_size: FFT window size
            hop_size: Number of samples between windows
            window_type: Type of window function
        """
        super().__init__(sample_rate)
        self.window_size = window_size
        self.hop_size = hop_size
        self.window_type = window_type
        
        # Create window function
        if window_type == 'hann':
            self.window = np.hanning(window_size)
        elif window_type == 'hamming':
            self.window = np.hamming(window_size)
        elif window_type == 'blackman':
            self.window = np.blackman(window_size)
        else:
            self.window = np.ones(window_size)
    
    def generate(self, signal: np.ndarray) -> np.ndarray:
        """
        Generate spectrogram data using STFT.
        
        Args:
            signal: Audio signal
            
        Returns:
            Spectrogram data (frequency x time)
        """
        # Calculate number of frames
        num_frames = (len(signal) - self.window_size) // self.hop_size + 1
        
        if num_frames <= 0:
            return np.array([[]])
        
        # Initialize spectrogram array
        spectrogram = np.zeros((self.window_size // 2 + 1, num_frames))
        
        # Compute STFT
        for frame_idx in range(num_frames):
            start = frame_idx * self.hop_size
            end = start + self.window_size
            
            if end > len(signal):
                break
            
            # Extract and window the frame
            frame = signal[start:end] * self.window
            
            # Compute FFT
            fft = np.fft.rfft(frame)
            
            # Convert to magnitude (dB scale)
            magnitude = np.abs(fft)
            magnitude_db = 20 * np.log10(magnitude + 1e-10)  # Avoid log(0)
            
            spectrogram[:, frame_idx] = magnitude_db
        
        return spectrogram
    
    def get_time_axis(self, num_frames: int) -> np.ndarray:
        """
        Get time axis for spectrogram.
        
        Args:
            num_frames: Number of frames in spectrogram
            
        Returns:
            Time values in seconds
        """
        return np.arange(num_frames) * self.hop_size / self.sample_rate
    
    def get_frequency_axis(self) -> np.ndarray:
        """
        Get frequency axis for spectrogram.
        
        Returns:
            Frequency values in Hz
        """
        return np.linspace(0, self.sample_rate / 2, self.window_size // 2 + 1)


class FrequencyScopeVisualizer(Visualizer):
    """
    Frequency scope / oscilloscope visualization.
    
    Displays current frequency spectrum.
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        fft_size: int = 2048,
        freq_range: Tuple[float, float] = (20.0, 20000.0)
    ):
        """
        Initialize a frequency scope visualizer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            fft_size: FFT size for frequency analysis
            freq_range: Frequency range to display (min_hz, max_hz)
        """
        super().__init__(sample_rate)
        self.fft_size = fft_size
        self.freq_range = freq_range
    
    def generate(self, signal: np.ndarray) -> np.ndarray:
        """
        Generate frequency spectrum data.
        
        Args:
            signal: Audio signal
            
        Returns:
            Frequency spectrum (magnitude vs frequency)
        """
        # Pad or trim to FFT size
        if len(signal) < self.fft_size:
            signal = np.pad(signal, (0, self.fft_size - len(signal)))
        else:
            signal = signal[:self.fft_size]
        
        # Apply window
        window = np.hanning(self.fft_size)
        windowed = signal * window
        
        # Compute FFT
        fft = np.fft.rfft(windowed)
        magnitude = np.abs(fft)
        
        # Convert to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        return magnitude_db
    
    def get_frequency_bins(self) -> np.ndarray:
        """
        Get frequency bin values.
        
        Returns:
            Frequency values in Hz for each bin
        """
        return np.linspace(0, self.sample_rate / 2, self.fft_size // 2 + 1)
    
    def filter_frequency_range(self, spectrum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter spectrum to specified frequency range.
        
        Args:
            spectrum: Full frequency spectrum
            
        Returns:
            Tuple of (filtered frequencies, filtered magnitudes)
        """
        freq_bins = self.get_frequency_bins()
        
        # Find indices within range
        min_freq, max_freq = self.freq_range
        mask = (freq_bins >= min_freq) & (freq_bins <= max_freq)
        
        return freq_bins[mask], spectrum[mask]


class VideoRenderer:
    """
    Renders synchronized video with audio visualization.
    """
    
    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        sample_rate: int = 44100
    ):
        """
        Initialize a video renderer.
        
        Args:
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            sample_rate: Audio sample rate in Hz
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.sample_rate = sample_rate
        self.samples_per_frame = sample_rate // fps
    
    def render_frames(
        self,
        signal: np.ndarray,
        visualizer: Visualizer,
        output_path: Optional[str] = None
    ) -> list:
        """
        Render video frames from audio signal.
        
        Args:
            signal: Audio signal
            visualizer: Visualizer to use for frame generation
            output_path: Optional path to save video (requires external video library)
            
        Returns:
            List of frame data
        """
        # Calculate number of frames
        num_frames = int(len(signal) / self.samples_per_frame)
        
        frames = []
        
        for frame_idx in range(num_frames):
            start = frame_idx * self.samples_per_frame
            end = start + self.samples_per_frame
            
            if end > len(signal):
                end = len(signal)
            
            # Extract audio chunk for this frame
            chunk = signal[start:end]
            
            # Generate visualization
            if isinstance(visualizer, WaveformVisualizer):
                frame_data = visualizer.to_image_data(chunk, self.height, self.width)
            elif isinstance(visualizer, SpectrogramVisualizer):
                spec_data = visualizer.generate(chunk)
                # Resize to fit frame
                frame_data = self._resize_spectrogram(spec_data, self.height, self.width)
            elif isinstance(visualizer, FrequencyScopeVisualizer):
                spectrum = visualizer.generate(chunk)
                frame_data = self._spectrum_to_image(spectrum, self.height, self.width)
            else:
                frame_data = np.zeros((self.height, self.width))
            
            frames.append(frame_data)
        
        # Save video if output path provided
        if output_path:
            self._save_video(frames, output_path)
        
        return frames
    
    def _resize_spectrogram(self, spec_data: np.ndarray, height: int, width: int) -> np.ndarray:
        """Resize spectrogram to fit frame dimensions."""
        if spec_data.size == 0:
            return np.zeros((height, width))
        
        # Simple nearest-neighbor resize
        h, w = spec_data.shape
        if h == 0 or w == 0:
            return np.zeros((height, width))
        
        row_indices = np.linspace(0, h - 1, height).astype(int)
        col_indices = np.linspace(0, w - 1, width).astype(int)
        
        resized = spec_data[row_indices][:, col_indices]
        
        # Normalize to 0-1 range
        resized = (resized - np.min(resized)) / (np.max(resized) - np.min(resized) + 1e-10)
        
        return resized
    
    def _spectrum_to_image(self, spectrum: np.ndarray, height: int, width: int) -> np.ndarray:
        """Convert frequency spectrum to image."""
        image = np.zeros((height, width))
        
        # Resize spectrum to width
        if len(spectrum) > width:
            indices = np.linspace(0, len(spectrum) - 1, width).astype(int)
            spectrum = spectrum[indices]
        
        # Normalize spectrum
        spec_min = np.min(spectrum)
        spec_max = np.max(spectrum)
        if spec_max > spec_min:
            spectrum = (spectrum - spec_min) / (spec_max - spec_min)
        
        # Draw bars
        for i, mag in enumerate(spectrum):
            if i >= width:
                break
            bar_height = int(mag * height)
            if bar_height > 0:
                image[-bar_height:, i] = 1.0
        
        return image
    
    def _save_video(self, frames: list, output_path: str) -> None:
        """
        Save frames as video.
        
        Note: This is a placeholder. Real implementation would require
        a video encoding library like opencv-python or moviepy.
        """
        print(f"Note: Video export to {output_path} requires additional dependencies.")
        print("Install opencv-python or moviepy for video export support.")
        # In a real implementation:
        # import cv2
        # out = cv2.VideoWriter(output_path, ...)
        # for frame in frames:
        #     out.write(frame)
        # out.release()
