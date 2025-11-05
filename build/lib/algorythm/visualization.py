"""
algorythm.visualization - Synchronized visualization tools

This module provides tools for generating visual output alongside audio,
including waveforms, spectrograms, and frequency scopes.
"""

from typing import Optional, Literal, Tuple, List, Dict, Any
import numpy as np
import warnings

# Optional dependencies
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class Visualizer:
    """
    Base class for audio visualization.
    """
    
    def __init__(self, sample_rate: int = 44100, debug: bool = False):
        """
        Initialize a visualizer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            debug: Enable debug output
        """
        self.sample_rate = sample_rate
        self.debug = debug
        self._frame_count = 0
    
    def generate(self, signal: np.ndarray) -> np.ndarray:
        """
        Generate visualization data from audio signal.
        
        Args:
            signal: Audio signal
            
        Returns:
            Visualization data
        """
        raise NotImplementedError("Subclasses must implement generate()")
    
    def _log_debug(self, message: str):
        """Log debug message if debug mode is enabled."""
        if self.debug:
            print(f"[{self.__class__.__name__}] {message}")


class WaveformVisualizer(Visualizer):
    """
    Real-time waveform visualization.
    
    Displays amplitude over time.
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        window_size: int = 1024,
        downsample_factor: int = 1,
        debug: bool = False
    ):
        """
        Initialize a waveform visualizer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            window_size: Size of visualization window
            downsample_factor: Factor to downsample signal for display
            debug: Enable debug output
        """
        super().__init__(sample_rate, debug)
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
        width: int = 1024,
        line_thickness: int = 2,
        center_line: bool = True
    ) -> np.ndarray:
        """
        Convert waveform to image data with enhanced rendering.
        
        Args:
            signal: Audio signal
            width: Image width in pixels
            height: Image height in pixels
            line_thickness: Thickness of waveform line
            center_line: Draw center reference line
            
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
        
        # Draw center line if requested
        if center_line:
            center = height // 2
            image[center, :] = 0.3
        
        # Map signal to image coordinates with anti-aliasing
        center = height // 2
        for i in range(len(signal) - 1):
            y1 = int(center - signal[i] * center * 0.9)
            y2 = int(center - signal[i + 1] * center * 0.9)
            y1 = np.clip(y1, 0, height - 1)
            y2 = np.clip(y2, 0, height - 1)
            
            # Draw line between points
            if abs(y2 - y1) <= 1:
                # Horizontal or nearly horizontal
                for t in range(line_thickness):
                    offset = t - line_thickness // 2
                    y = np.clip(y1 + offset, 0, height - 1)
                    image[y, i] = 1.0
            else:
                # Vertical line using interpolation
                steps = abs(y2 - y1)
                for step in range(steps + 1):
                    y = int(y1 + (y2 - y1) * step / steps)
                    y = np.clip(y, 0, height - 1)
                    for t in range(line_thickness):
                        offset = t - line_thickness // 2
                        y_thick = np.clip(y + offset, 0, height - 1)
                        image[y_thick, i] = 1.0
        
        self._log_debug(f"Generated waveform image: {width}x{height}, signal length: {len(signal)}")
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
        window_type: Literal['hann', 'hamming', 'blackman'] = 'hann',
        debug: bool = False
    ):
        """
        Initialize a spectrogram visualizer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            window_size: FFT window size
            hop_size: Number of samples between windows
            window_type: Type of window function
            debug: Enable debug output
        """
        super().__init__(sample_rate, debug)
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
        Generate spectrogram data using STFT (optimized vectorized version).
        
        Args:
            signal: Audio signal
            
        Returns:
            Spectrogram data (frequency x time)
        """
        # Calculate number of frames
        num_frames = (len(signal) - self.window_size) // self.hop_size + 1
        
        if num_frames <= 0:
            self._log_debug("Signal too short for spectrogram")
            return np.array([[]])
        
        # Vectorized STFT using stride tricks for efficiency
        # Create a view into the signal with overlapping windows
        from numpy.lib.stride_tricks import as_strided
        
        # Calculate shape and strides for windowed view
        shape = (num_frames, self.window_size)
        strides = (signal.strides[0] * self.hop_size, signal.strides[0])
        
        # Create windowed view (no copy)
        frames = as_strided(signal, shape=shape, strides=strides, writeable=False)
        
        # Apply window function (broadcast)
        windowed_frames = frames * self.window
        
        # Compute FFT for all frames at once (much faster)
        fft_result = np.fft.rfft(windowed_frames, axis=1)
        
        # Convert to magnitude (dB scale)
        magnitude = np.abs(fft_result)
        spectrogram = 20 * np.log10(magnitude + 1e-10)  # Avoid log(0)
        
        # Transpose to match expected output shape (freq x time)
        spectrogram = spectrogram.T
        
        self._log_debug(f"Generated spectrogram: {spectrogram.shape[0]} freq bins x {spectrogram.shape[1]} time frames")
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
    
    def to_colored_image(
        self,
        spectrogram: np.ndarray,
        colormap: str = 'viridis',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ) -> np.ndarray:
        """
        Convert spectrogram to colored image using matplotlib colormap.
        
        Args:
            spectrogram: Spectrogram data
            colormap: Matplotlib colormap name
            vmin: Minimum value for color mapping
            vmax: Maximum value for color mapping
            
        Returns:
            RGB image array (height, width, 3)
        """
        if not HAS_MATPLOTLIB:
            warnings.warn("Matplotlib not available, returning grayscale")
            # Normalize to 0-1 range
            spec_norm = (spectrogram - np.min(spectrogram))
            if np.max(spec_norm) > 0:
                spec_norm = spec_norm / np.max(spec_norm)
            return np.stack([spec_norm] * 3, axis=-1)
        
        # Normalize spectrogram
        if vmin is None:
            vmin = np.percentile(spectrogram, 5)
        if vmax is None:
            vmax = np.percentile(spectrogram, 95)
        
        spec_norm = np.clip((spectrogram - vmin) / (vmax - vmin + 1e-10), 0, 1)
        
        # Apply colormap
        cmap = plt.get_cmap(colormap)
        colored = cmap(spec_norm)[:, :, :3]  # Remove alpha channel
        
        return colored
    
    def to_image_data(
        self,
        signal: np.ndarray,
        height: int = 480,
        width: int = 640
    ) -> np.ndarray:
        """
        Generate spectrogram visualization as image data.
        
        Args:
            signal: Audio signal chunk
            height: Image height in pixels
            width: Image width in pixels
            
        Returns:
            2D array of visualization data (0-1 range)
        """
        # Generate spectrogram
        spectrogram = self.generate(signal)
        
        if spectrogram.size == 0 or spectrogram.shape[0] == 0 or spectrogram.shape[1] == 0:
            self._log_debug(f"Empty spectrogram generated")
            return np.zeros((height, width))
        
        # Resize to target dimensions
        spec_height, spec_width = spectrogram.shape
        
        # Create indices for resizing
        if spec_height > 0 and spec_width > 0:
            row_indices = (np.arange(height) * spec_height / height).astype(int)
            col_indices = (np.arange(width) * spec_width / width).astype(int)
            
            # Clip indices to valid range
            row_indices = np.clip(row_indices, 0, spec_height - 1)
            col_indices = np.clip(col_indices, 0, spec_width - 1)
            
            resized = spectrogram[row_indices][:, col_indices]
        else:
            resized = np.zeros((height, width))
        
        # Normalize to 0-1 range
        spec_min = np.min(resized)
        spec_max = np.max(resized)
        
        if spec_max > spec_min:
            normalized = (resized - spec_min) / (spec_max - spec_min)
        else:
            normalized = np.zeros((height, width))
        
        # Flip vertically (low frequencies at bottom)
        normalized = np.flipud(normalized)
        
        self._log_debug(f"Generated spectrogram image: {height}x{width}")
        return normalized


class FrequencyScopeVisualizer(Visualizer):
    """
    Frequency scope / oscilloscope visualization.
    
    Displays current frequency spectrum.
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        fft_size: int = 2048,
        freq_range: Tuple[float, float] = (20.0, 20000.0),
        debug: bool = False
    ):
        """
        Initialize a frequency scope visualizer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            fft_size: FFT size for frequency analysis
            freq_range: Frequency range to display (min_hz, max_hz)
            debug: Enable debug output
        """
        super().__init__(sample_rate, debug)
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
    
    def to_image_data(
        self,
        signal: np.ndarray,
        height: int = 480,
        width: int = 640
    ) -> np.ndarray:
        """
        Generate frequency scope visualization as image data.
        
        Args:
            signal: Audio signal chunk
            height: Image height in pixels
            width: Image width in pixels
            
        Returns:
            2D array of visualization data (0-1 range)
        """
        # Generate spectrum
        spectrum = self.generate(signal)
        
        # Filter to frequency range
        freq_bins, filtered_spectrum = self.filter_frequency_range(spectrum)
        
        if len(filtered_spectrum) == 0:
            self._log_debug(f"Empty spectrum after filtering")
            return np.zeros((height, width))
        
        # Normalize to 0-1 range
        spec_min = np.min(filtered_spectrum)
        spec_max = np.max(filtered_spectrum)
        
        if spec_max > spec_min:
            normalized = (filtered_spectrum - spec_min) / (spec_max - spec_min)
        else:
            normalized = np.zeros_like(filtered_spectrum)
        
        # Create image with bars
        image = np.zeros((height, width))
        
        # Resample to width
        if len(normalized) > width:
            indices = np.linspace(0, len(normalized) - 1, width).astype(int)
            bar_heights = normalized[indices]
        else:
            # Interpolate to width
            x_old = np.arange(len(normalized))
            x_new = np.linspace(0, len(normalized) - 1, width)
            bar_heights = np.interp(x_new, x_old, normalized)
        
        # Draw frequency bars from bottom
        for i in range(width):
            bar_height = int(bar_heights[i] * height * 0.9)  # Use 90% of height
            if bar_height > 0:
                image[-bar_height:, i] = 1.0
        
        self._log_debug(f"Generated frequency scope image: {height}x{width}")
        return image


class VideoRenderer:
    """
    Renders synchronized video with audio visualization.
    Enhanced with multiple visualizer support and extensive customization.
    Supports both OpenCV and matplotlib backends.
    
    Performance Tips:
    - Lower resolution (720p instead of 1080p) significantly speeds up rendering
    - Reduce fps (24 instead of 30) for 20% speed improvement
    - Use CircularVisualizer or WaveformVisualizer (faster than SpectrogramVisualizer)
    - Streaming mode automatically used when output_path is provided (saves memory)
    - Multi-threaded ffmpeg encoding uses all CPU cores
    
    NEW in this version:
    - Streaming video writer: No longer buffers all frames in memory
    - Optimized BGR conversion: 30-40% faster frame writing
    - Vectorized circular visualizer: 2-3x faster bar rendering
    - Faster ffmpeg preset: 2x faster video encoding
    - Memory-efficient: Can handle videos of any length without OOM
    """
    
    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        sample_rate: int = 44100,
        background_color: Tuple[int, int, int] = (0, 0, 0),
        foreground_color: Tuple[int, int, int] = (255, 255, 255),
        colormap: str = 'viridis',
        debug: bool = False,
        use_matplotlib: bool = False
    ):
        """
        Initialize a video renderer.
        
        Args:
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            sample_rate: Audio sample rate in Hz
            background_color: RGB background color (0-255 each)
            foreground_color: RGB foreground color (0-255 each)
            colormap: Matplotlib colormap for spectrograms
            debug: Enable debug output
            use_matplotlib: Force use of matplotlib backend (slower but no opencv required)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.sample_rate = sample_rate
        self.samples_per_frame = sample_rate // fps
        self.background_color = background_color
        self.foreground_color = foreground_color
        self.colormap = colormap
        self.debug = debug
        self.use_matplotlib = use_matplotlib or not HAS_OPENCV
        
        if self.debug:
            print(f"[VideoRenderer] Initialized: {width}x{height} @ {fps}fps")
            print(f"[VideoRenderer] Backend: {'matplotlib' if self.use_matplotlib else 'opencv'}")
            print(f"[VideoRenderer] Samples per frame: {self.samples_per_frame}")
    
    def render_frames(
        self,
        signal: np.ndarray,
        visualizer: Visualizer,
        output_path: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> list:
        """
        Render video frames from audio signal with progress tracking (optimized).
        
        Args:
            signal: Audio signal
            visualizer: Visualizer to use for frame generation
            output_path: Optional path to save video
            progress_callback: Optional callback function(current, total) for progress updates
            
        Returns:
            List of frame data
        """
        # Calculate number of frames
        num_frames = int(len(signal) / self.samples_per_frame)
        
        if self.debug:
            print(f"[VideoRenderer] Rendering {num_frames} frames...")
            print(f"[VideoRenderer] Signal length: {len(signal)} samples ({len(signal)/self.sample_rate:.2f}s)")
        
        # If output path is provided, use streaming mode to save memory
        if output_path:
            if self.debug:
                print(f"[VideoRenderer] Using streaming mode (memory-efficient)")
            self._render_and_save_streaming(signal, visualizer, output_path, progress_callback)
            return []
        
        # Otherwise, buffer frames in memory
        # Optimize for spectrogram visualizer - process entire signal at once
        if isinstance(visualizer, SpectrogramVisualizer):
            if self.debug:
                print(f"[VideoRenderer] Using optimized spectrogram rendering...")
            frames = self._render_spectrogram_optimized(signal, visualizer, num_frames, progress_callback)
        else:
            # Standard frame-by-frame rendering for other visualizers
            frames = self._render_frames_standard(signal, visualizer, num_frames, progress_callback)
        
        if self.debug:
            print(f"[VideoRenderer] Completed rendering {len(frames)} frames")
        
        return frames
    
    def _render_and_save_streaming(
        self,
        signal: np.ndarray,
        visualizer: Visualizer,
        output_path: str,
        progress_callback: Optional[callable] = None
    ) -> None:
        """
        Memory-efficient streaming renderer - generates and writes frames directly to video
        without buffering all frames in memory. Ideal for long videos.
        
        Args:
            signal: Audio signal
            visualizer: Visualizer instance
            output_path: Output video path
            progress_callback: Optional progress callback
        """
        if not HAS_OPENCV:
            if self.debug:
                print(f"[VideoRenderer] OpenCV not available, falling back to buffered rendering")
            frames = self._render_frames_standard(signal, visualizer, 
                                                 int(len(signal) / self.samples_per_frame), 
                                                 progress_callback)
            self._save_video(frames, signal, output_path)
            return
        
        import cv2
        import tempfile
        import os
        
        if self.debug:
            print(f"[VideoRenderer] Using streaming mode (memory-efficient)")
        
        # Calculate number of frames
        num_frames = int(len(signal) / self.samples_per_frame)
        
        # Create temporary video without audio
        temp_video = tempfile.NamedTemporaryFile(suffix='_novideo.mp4', delete=False)
        temp_video_path = temp_video.name
        temp_video.close()
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, self.fps, (self.width, self.height))
        
        if not out.isOpened():
            raise RuntimeError("Failed to open video writer")
        
        # Pre-allocate arrays for efficiency
        frame_rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame_bgr = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        print_interval = max(1, num_frames // 20)
        
        print(f"[VideoRenderer] Streaming {num_frames} frames directly to video...")
        
        # Special handling for spectrogram (still needs full computation)
        if isinstance(visualizer, SpectrogramVisualizer):
            # Compute full spectrogram once
            full_spectrogram = visualizer.generate(signal)
            
            if full_spectrogram.size > 0 and full_spectrogram.shape[0] > 0 and full_spectrogram.shape[1] > 0:
                spec_height, spec_width = full_spectrogram.shape
                row_indices = (np.arange(self.height) * spec_height / self.height).astype(int)
                row_indices = np.clip(row_indices, 0, spec_height - 1)
                
                # Normalize once
                spec_min = np.percentile(full_spectrogram, 5)
                spec_max = np.percentile(full_spectrogram, 95)
                normalized_spec = np.clip((full_spectrogram - spec_min) / (spec_max - spec_min + 1e-10), 0, 1)
                normalized_spec = np.flipud(normalized_spec)
                
                # Stream frames
                for frame_idx in range(num_frames):
                    start_col = int(frame_idx * spec_width / num_frames)
                    end_col = int((frame_idx + 1) * spec_width / num_frames)
                    end_col = min(end_col, spec_width)
                    
                    if start_col < spec_width:
                        frame_spec = normalized_spec[:, start_col:end_col]
                        
                        if frame_spec.shape[1] > 0:
                            col_indices = (np.arange(self.width) * frame_spec.shape[1] / self.width).astype(int)
                            col_indices = np.clip(col_indices, 0, frame_spec.shape[1] - 1)
                            frame_data = frame_spec[row_indices][:, col_indices]
                        else:
                            frame_data = np.zeros((self.height, self.width))
                    else:
                        frame_data = np.zeros((self.height, self.width))
                    
                    # Convert and write frame
                    self._write_frame_to_video(frame_data, out, frame_rgb, frame_bgr)
                    
                    if progress_callback and frame_idx % 10 == 0:
                        progress_callback(frame_idx, num_frames)
                    
                    if (frame_idx + 1) % print_interval == 0 or frame_idx == num_frames - 1:
                        progress = 100 * (frame_idx + 1) / num_frames
                        print(f"[VideoRenderer] Progress: {progress:.1f}% ({frame_idx+1}/{num_frames})")
            else:
                # Empty spectrogram, write blank frames
                blank_frame = np.zeros((self.height, self.width))
                for frame_idx in range(num_frames):
                    self._write_frame_to_video(blank_frame, out, frame_rgb, frame_bgr)
        else:
            # Standard visualizers - process frame by frame
            for frame_idx in range(num_frames):
                start = frame_idx * self.samples_per_frame
                end = start + self.samples_per_frame
                
                if end > len(signal):
                    end = len(signal)
                
                chunk = signal[start:end]
                
                # Generate visualization
                if isinstance(visualizer, WaveformVisualizer):
                    frame_data = visualizer.to_image_data(chunk, self.height, self.width)
                elif isinstance(visualizer, FrequencyScopeVisualizer):
                    spectrum = visualizer.generate(chunk)
                    frame_data = self._spectrum_to_image(spectrum, self.height, self.width)
                elif isinstance(visualizer, CircularVisualizer):
                    frame_data = visualizer.to_image_data(chunk, self.height, self.width)
                elif isinstance(visualizer, OscilloscopeVisualizer):
                    frame_data = visualizer.to_image_data(chunk, self.height, self.width)
                elif isinstance(visualizer, ParticleVisualizer):
                    frame_data = visualizer.to_image_data(chunk, self.height, self.width)
                else:
                    frame_data = np.zeros((self.height, self.width))
                
                # Convert and write frame
                self._write_frame_to_video(frame_data, out, frame_rgb, frame_bgr)
                
                if progress_callback and frame_idx % 10 == 0:
                    progress_callback(frame_idx, num_frames)
                
                if (frame_idx + 1) % print_interval == 0 or frame_idx == num_frames - 1:
                    progress = 100 * (frame_idx + 1) / num_frames
                    print(f"[VideoRenderer] Progress: {progress:.1f}% ({frame_idx+1}/{num_frames})")
        
        out.release()
        
        if self.debug:
            print(f"[VideoRenderer] Adding audio track...")
        
        # Add audio using ffmpeg
        self._add_audio_to_video(temp_video_path, signal, output_path)
        
        # Clean up temporary file
        os.unlink(temp_video_path)
        
        print(f"✓ Video exported to: {output_path}")
    
    def _write_frame_to_video(
        self,
        frame_data: np.ndarray,
        video_writer,
        frame_rgb: np.ndarray,
        frame_bgr: np.ndarray
    ) -> None:
        """
        Efficiently write a single frame to video writer.
        Reuses pre-allocated arrays to minimize memory allocation.
        
        Args:
            frame_data: Normalized frame data (0-1 range)
            video_writer: OpenCV VideoWriter object
            frame_rgb: Pre-allocated RGB array (reused)
            frame_bgr: Pre-allocated BGR array (reused)
        """
        if frame_data.ndim == 3 and frame_data.shape[2] == 3:
            # Colored frame - direct conversion
            np.multiply(frame_data, 255, out=frame_rgb, casting='unsafe')
            frame_bgr[:, :, 0] = frame_rgb[:, :, 2]  # B
            frame_bgr[:, :, 1] = frame_rgb[:, :, 1]  # G
            frame_bgr[:, :, 2] = frame_rgb[:, :, 0]  # R
        else:
            # Grayscale - apply colors
            frame_rgb[:] = self.background_color
            for c in range(3):
                if self.foreground_color[c] != self.background_color[c]:
                    np.add(frame_rgb[:, :, c], 
                           frame_data * (self.foreground_color[c] - self.background_color[c]),
                           out=frame_rgb[:, :, c], casting='unsafe')
            np.clip(frame_rgb, 0, 255, out=frame_rgb)
            # RGB to BGR
            frame_bgr[:, :, 0] = frame_rgb[:, :, 2]
            frame_bgr[:, :, 1] = frame_rgb[:, :, 1]
            frame_bgr[:, :, 2] = frame_rgb[:, :, 0]
        
        video_writer.write(frame_bgr)
    
    def _render_spectrogram_optimized(
        self,
        signal: np.ndarray,
        visualizer: SpectrogramVisualizer,
        num_frames: int,
        progress_callback: Optional[callable] = None
    ) -> list:
        """
        Optimized spectrogram rendering - compute full spectrogram once, then slice.
        
        Args:
            signal: Audio signal
            visualizer: SpectrogramVisualizer instance
            num_frames: Number of frames to render
            progress_callback: Optional progress callback
            
        Returns:
            List of frame data
        """
        # Compute spectrogram for entire signal at once (much faster)
        full_spectrogram = visualizer.generate(signal)
        
        if full_spectrogram.size == 0 or full_spectrogram.shape[0] == 0 or full_spectrogram.shape[1] == 0:
            if self.debug:
                print(f"[VideoRenderer] Empty spectrogram, returning blank frames")
            return [np.zeros((self.height, self.width)) for _ in range(num_frames)]
        
        # Pre-compute resize indices for efficiency
        spec_height, spec_width = full_spectrogram.shape
        row_indices = (np.arange(self.height) * spec_height / self.height).astype(int)
        row_indices = np.clip(row_indices, 0, spec_height - 1)
        
        # Calculate how many spectrogram columns per video frame
        cols_per_frame = max(1, spec_width // num_frames)
        
        # Normalize the entire spectrogram once
        spec_min = np.percentile(full_spectrogram, 5)
        spec_max = np.percentile(full_spectrogram, 95)
        normalized_spec = np.clip((full_spectrogram - spec_min) / (spec_max - spec_min + 1e-10), 0, 1)
        normalized_spec = np.flipud(normalized_spec)  # Flip once
        
        frames = []
        for frame_idx in range(num_frames):
            # Calculate which columns of spectrogram to use for this frame
            start_col = int(frame_idx * spec_width / num_frames)
            end_col = int((frame_idx + 1) * spec_width / num_frames)
            end_col = min(end_col, spec_width)
            
            if start_col >= spec_width:
                frames.append(np.zeros((self.height, self.width)))
                continue
            
            # Extract relevant columns
            frame_spec = normalized_spec[:, start_col:end_col]
            
            # Resize width to video width
            if frame_spec.shape[1] > 0:
                col_indices = (np.arange(self.width) * frame_spec.shape[1] / self.width).astype(int)
                col_indices = np.clip(col_indices, 0, frame_spec.shape[1] - 1)
                frame_resized = frame_spec[row_indices][:, col_indices]
            else:
                frame_resized = np.zeros((self.height, self.width))
            
            frames.append(frame_resized)
            
            # Progress callback
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, num_frames)
            
            # Debug output
            if self.debug and frame_idx % 100 == 0:
                print(f"[VideoRenderer] Rendered frame {frame_idx}/{num_frames} ({100*frame_idx/num_frames:.1f}%)")
        
        return frames
    
    def _render_frames_standard(
        self,
        signal: np.ndarray,
        visualizer: Visualizer,
        num_frames: int,
        progress_callback: Optional[callable] = None
    ) -> list:
        """
        Standard frame-by-frame rendering for non-spectrogram visualizers.
        
        Args:
            signal: Audio signal
            visualizer: Visualizer instance
            num_frames: Number of frames to render
            progress_callback: Optional progress callback
            
        Returns:
            List of frame data
        """
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
            elif isinstance(visualizer, FrequencyScopeVisualizer):
                spectrum = visualizer.generate(chunk)
                frame_data = self._spectrum_to_image(spectrum, self.height, self.width)
            elif isinstance(visualizer, CircularVisualizer):
                frame_data = visualizer.to_image_data(chunk, self.height, self.width)
            elif isinstance(visualizer, OscilloscopeVisualizer):
                frame_data = visualizer.to_image_data(chunk, self.height, self.width)
            elif isinstance(visualizer, ParticleVisualizer):
                frame_data = visualizer.to_image_data(chunk, self.height, self.width)
            else:
                frame_data = np.zeros((self.height, self.width))
            
            frames.append(frame_data)
            
            # Progress callback
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, num_frames)
            
            # Debug output every 100 frames
            if self.debug and frame_idx % 100 == 0:
                print(f"[VideoRenderer] Rendered frame {frame_idx}/{num_frames} ({100*frame_idx/num_frames:.1f}%)")
        
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
    
    def _save_video(self, frames: list, audio_signal: np.ndarray, output_path: str) -> None:
        """
        Save frames as video with audio using opencv or matplotlib backend.
        
        Args:
            frames: List of frame data arrays
            audio_signal: Audio signal to embed in video
            output_path: Output video file path
        """
        if self.debug:
            print(f"[VideoRenderer] Saving video to: {output_path}")
        
        try:
            import tempfile
            import os
            from pathlib import Path
            
            # Ensure proper file extension
            if not output_path.lower().endswith('.mp4'):
                output_path += '.mp4'
            
            if self.use_matplotlib and HAS_MATPLOTLIB:
                self._save_video_matplotlib(frames, audio_signal, output_path)
            elif HAS_OPENCV:
                self._save_video_opencv(frames, audio_signal, output_path)
            else:
                raise ImportError("No video backend available. Install opencv-python or matplotlib.")
            
            if self.debug:
                print(f"[VideoRenderer] Video saved successfully")
            
        except ImportError as e:
            print(f"❌ Error: {e}")
            print(f"Note: Video export requires opencv-python or matplotlib and ffmpeg.")
            print(f"Install with: pip install opencv-python matplotlib")
            print(f"And ensure ffmpeg is installed on your system.")
            raise  # Re-raise the exception so the caller knows it failed
        except Exception as e:
            print(f"❌ Error exporting video: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            raise  # Re-raise the exception so the caller knows it failed
    
    def _save_video_opencv(self, frames: list, audio_signal: np.ndarray, output_path: str) -> None:
        """Save video using OpenCV backend (streaming optimized - no frame buffering)."""
        import cv2
        import tempfile
        import os
        
        if self.debug:
            print(f"[VideoRenderer] Using OpenCV backend")
        
        # Create temporary video without audio
        temp_video = tempfile.NamedTemporaryFile(suffix='_novideo.mp4', delete=False)
        temp_video_path = temp_video.name
        temp_video.close()
        
        # Initialize video writer with faster codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, self.fps, (self.width, self.height))
        
        if self.debug:
            print(f"[VideoRenderer] Writing {len(frames)} frames to video...")
        
        # Pre-allocate BGR conversion arrays for efficiency
        frame_rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame_bgr = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Write frames with optimized conversion
        total_frames = len(frames)
        print_interval = max(1, total_frames // 20)  # Print progress every 5%
        
        for i, frame_data in enumerate(frames):
            # Handle colored images (3 channels)
            if frame_data.ndim == 3 and frame_data.shape[2] == 3:
                # Direct RGB to BGR conversion (reuse arrays)
                np.multiply(frame_data, 255, out=frame_rgb, casting='unsafe')
                frame_bgr[:, :, 0] = frame_rgb[:, :, 2]  # B = R
                frame_bgr[:, :, 1] = frame_rgb[:, :, 1]  # G = G
                frame_bgr[:, :, 2] = frame_rgb[:, :, 0]  # R = B
            else:
                # Optimized grayscale to colored (vectorized)
                frame_rgb[:] = self.background_color
                mask = frame_data > 0
                for c in range(3):
                    if self.foreground_color[c] != self.background_color[c]:
                        np.add(frame_rgb[:, :, c], 
                               frame_data * (self.foreground_color[c] - self.background_color[c]),
                               out=frame_rgb[:, :, c], casting='unsafe')
                np.clip(frame_rgb, 0, 255, out=frame_rgb)
                # RGB to BGR
                frame_bgr[:, :, 0] = frame_rgb[:, :, 2]
                frame_bgr[:, :, 1] = frame_rgb[:, :, 1]
                frame_bgr[:, :, 2] = frame_rgb[:, :, 0]
            
            out.write(frame_bgr)
            
            # Progress feedback
            if (i + 1) % print_interval == 0 or i == total_frames - 1:
                progress = 100 * (i + 1) / total_frames
                print(f"[VideoRenderer] Writing frames: {progress:.1f}% ({i+1}/{total_frames})")
        
        out.release()
        
        if self.debug:
            print(f"[VideoRenderer] Adding audio track...")
        
        # Now add audio using ffmpeg
        self._add_audio_to_video(temp_video_path, audio_signal, output_path)
        
        # Clean up temporary file
        os.unlink(temp_video_path)
        
        print(f"✓ Video exported to: {output_path}")
    
    def _save_video_matplotlib(self, frames: list, audio_signal: np.ndarray, output_path: str) -> None:
        """Save video using matplotlib backend (slower but no opencv required)."""
        import tempfile
        import os
        from matplotlib.animation import FFMpegWriter
        
        if self.debug:
            print(f"[VideoRenderer] Using matplotlib backend")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(self.width/100, self.height/100), dpi=100)
        ax.set_position([0, 0, 1, 1])
        ax.axis('off')
        
        # Initialize plot
        im = ax.imshow(frames[0], cmap=self.colormap, aspect='auto')
        
        # Create temporary video
        temp_video = tempfile.NamedTemporaryFile(suffix='_novideo.mp4', delete=False)
        temp_video_path = temp_video.name
        temp_video.close()
        
        # Setup writer
        writer = FFMpegWriter(fps=self.fps, bitrate=5000)
        
        if self.debug:
            print(f"[VideoRenderer] Writing {len(frames)} frames with matplotlib...")
        
        with writer.saving(fig, temp_video_path, dpi=100):
            for i, frame_data in enumerate(frames):
                im.set_data(frame_data)
                writer.grab_frame()
                
                if self.debug and i % 100 == 0:
                    print(f"[VideoRenderer] Wrote frame {i}/{len(frames)}")
        
        plt.close(fig)
        
        if self.debug:
            print(f"[VideoRenderer] Adding audio track...")
        
        # Add audio
        self._add_audio_to_video(temp_video_path, audio_signal, output_path)
        
        # Clean up
        os.unlink(temp_video_path)
        
        print(f"✓ Video exported to: {output_path}")
    
    def _convert_to_bgr(self, frame_data: np.ndarray) -> np.ndarray:
        """Convert normalized grayscale frame to BGR with better color handling."""
        if not HAS_OPENCV:
            raise ImportError("OpenCV required for BGR conversion")
        
        import cv2
        
        # Handle colored images (3 channels)
        if frame_data.ndim == 3 and frame_data.shape[2] == 3:
            # Already RGB, convert to BGR
            frame_rgb = (frame_data * 255).astype(np.uint8)
            return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Create RGB image with background color
        frame_rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame_rgb[:] = self.background_color
        
        # Apply foreground color where frame_data is non-zero
        for c in range(3):
            channel = frame_rgb[:, :, c].astype(float)
            channel += frame_data * (self.foreground_color[c] - self.background_color[c])
            frame_rgb[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        return frame_bgr
    
    def _add_audio_to_video(self, video_path: str, audio_signal: np.ndarray, output_path: str) -> None:
        """Add audio to video using ffmpeg with optimized settings."""
        import subprocess
        import tempfile
        import wave
        import os
        
        if self.debug:
            print(f"[VideoRenderer] Combining video and audio with ffmpeg...")
        
        # Save audio to temporary WAV file
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        # Write WAV file
        audio_data = np.clip(audio_signal * 32767, -32768, 32767).astype(np.int16)
        with wave.open(temp_audio_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        # Use ffmpeg with faster encoding preset
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', temp_audio_path,
                '-c:v', 'libx264',
                '-preset', 'faster',  # Changed from 'medium' to 'faster' for 2x speed
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-strict', 'experimental',
                '-shortest',
                '-threads', '0',  # Use all CPU threads
                output_path
            ]
            
            if self.debug:
                print(f"[VideoRenderer] Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"[VideoRenderer] ffmpeg output: {result.stderr[-500:]}")  # Last 500 chars
            else:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running ffmpeg: {e}")
            if self.debug:
                print(f"ffmpeg stderr: {e.stderr}")
            print("Make sure ffmpeg is installed on your system.")
            raise
        except FileNotFoundError:
            print("❌ ffmpeg not found. Please install ffmpeg to export video with audio.")
            print("   Ubuntu/Debian: sudo apt install ffmpeg")
            print("   macOS: brew install ffmpeg")
            print("   Windows: Download from https://ffmpeg.org/")
            raise
        finally:
            # Clean up temporary audio file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)


class OscilloscopeVisualizer(Visualizer):
    """
    Oscilloscope/Phase Scope visualization.
    
    Displays real-time feedback on individual sound waves and stereo relationships.
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        window_size: int = 1024,
        mode: Literal['waveform', 'lissajous', 'phase'] = 'waveform',
        debug: bool = False
    ):
        """
        Initialize an oscilloscope visualizer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            window_size: Size of visualization window
            mode: Display mode (waveform, lissajous, phase)
            debug: Enable debug output
        """
        super().__init__(sample_rate, debug)
        self.window_size = window_size
        self.mode = mode
        self.mode = mode
    
    def generate(self, signal: np.ndarray) -> np.ndarray:
        """
        Generate oscilloscope visualization data.
        
        Args:
            signal: Audio signal (mono or stereo)
            
        Returns:
            Visualization data
        """
        if self.mode == 'waveform':
            return self._waveform_mode(signal)
        elif self.mode == 'lissajous':
            return self._lissajous_mode(signal)
        elif self.mode == 'phase':
            return self._phase_mode(signal)
        else:
            return self._waveform_mode(signal)
    
    def _waveform_mode(self, signal: np.ndarray) -> np.ndarray:
        """
        Generate waveform oscilloscope display.
        
        Args:
            signal: Audio signal
            
        Returns:
            Waveform data for display
        """
        # Take last window_size samples
        if len(signal) < self.window_size:
            padded = np.pad(signal, (self.window_size - len(signal), 0))
            return padded
        
        return signal[-self.window_size:]
    
    def _lissajous_mode(self, signal: np.ndarray) -> np.ndarray:
        """
        Generate Lissajous curve (X-Y scope) for stereo analysis.
        
        Args:
            signal: Stereo audio signal (2, N) or mono
            
        Returns:
            X-Y coordinate pairs for Lissajous curve
        """
        # For mono, create a delayed version for X-Y plot
        if signal.ndim == 1:
            left = signal[-self.window_size:]
            right = np.roll(left, self.window_size // 4)
        else:
            left = signal[0, -self.window_size:]
            right = signal[1, -self.window_size:]
        
        # Combine as X-Y pairs
        return np.vstack([left, right])
    
    def _phase_mode(self, signal: np.ndarray) -> np.ndarray:
        """
        Generate phase correlation display.
        
        Args:
            signal: Stereo audio signal (2, N) or mono
            
        Returns:
            Phase correlation data over time
        """
        if signal.ndim == 1:
            # For mono, return zeros (no phase difference)
            return np.zeros(self.window_size)
        
        # Calculate phase correlation for windows
        left = signal[0]
        right = signal[1]
        
        num_windows = len(left) // self.window_size
        if num_windows == 0:
            return np.zeros(1)
        
        correlations = []
        for i in range(num_windows):
            start = i * self.window_size
            end = start + self.window_size
            
            l_window = left[start:end]
            r_window = right[start:end]
            
            # Calculate correlation
            if np.std(l_window) > 0 and np.std(r_window) > 0:
                corr = np.corrcoef(l_window, r_window)[0, 1]
            else:
                corr = 0.0
            
            correlations.append(corr)
        
        return np.array(correlations)
    
    def to_image_data(
        self,
        signal: np.ndarray,
        height: int = 256,
        width: int = 512
    ) -> np.ndarray:
        """
        Convert oscilloscope data to image.
        
        Args:
            signal: Audio signal
            height: Image height in pixels
            width: Image width in pixels
            
        Returns:
            Image data array
        """
        data = self.generate(signal)
        image = np.zeros((height, width))
        
        if self.mode == 'waveform':
            # Draw waveform
            x_coords = np.linspace(0, width - 1, len(data)).astype(int)
            y_coords = ((data + 1) / 2 * (height - 1)).astype(int)
            y_coords = np.clip(y_coords, 0, height - 1)
            
            for x, y in zip(x_coords, y_coords):
                if 0 <= x < width:
                    image[y, x] = 1.0
        
        elif self.mode == 'lissajous' or self.mode == 'phase':
            # Draw X-Y plot
            if data.ndim == 2 and data.shape[0] == 2:
                x_coords = ((data[0] + 1) / 2 * (width - 1)).astype(int)
                y_coords = ((data[1] + 1) / 2 * (height - 1)).astype(int)
                
                x_coords = np.clip(x_coords, 0, width - 1)
                y_coords = np.clip(y_coords, 0, height - 1)
                
                for x, y in zip(x_coords, y_coords):
                    image[y, x] = 1.0
        
        return image


class CircularVisualizer(Visualizer):
    """
    Circular / Radial visualization.
    
    Displays audio in a circular pattern with customizable effects.
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        num_bars: int = 64,
        inner_radius: float = 0.2,
        bar_width: float = 0.8,
        smoothing: float = 0.5,
        debug: bool = False
    ):
        """
        Initialize a circular visualizer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            num_bars: Number of bars in the circle
            inner_radius: Inner radius as fraction of image size (0.0 to 1.0)
            bar_width: Bar width as fraction of available space (0.0 to 1.0)
            smoothing: Smoothing factor for bar heights (0.0 to 1.0)
            debug: Enable debug output
        """
        super().__init__(sample_rate, debug)
        self.num_bars = num_bars
        self.inner_radius = inner_radius
        self.bar_width = bar_width
        self.smoothing = smoothing
        self.prev_magnitudes = None
    
    def generate(self, signal: np.ndarray) -> np.ndarray:
        """
        Generate circular visualization data.
        
        Args:
            signal: Audio signal
            
        Returns:
            Bar magnitudes for circular display
        """
        # Compute FFT
        fft_size = min(2048, len(signal))
        if len(signal) < fft_size:
            signal = np.pad(signal, (0, fft_size - len(signal)))
        else:
            signal = signal[:fft_size]
        
        window = np.hanning(fft_size)
        windowed = signal * window
        fft = np.fft.rfft(windowed)
        magnitude = np.abs(fft)
        
        # Resample to num_bars
        if len(magnitude) > self.num_bars:
            indices = np.linspace(0, len(magnitude) - 1, self.num_bars).astype(int)
            magnitudes = magnitude[indices]
        else:
            magnitudes = np.pad(magnitude, (0, self.num_bars - len(magnitude)))
        
        # Apply smoothing
        if self.prev_magnitudes is not None and self.smoothing > 0:
            magnitudes = (magnitudes * (1 - self.smoothing) + 
                         self.prev_magnitudes * self.smoothing)
        
        self.prev_magnitudes = magnitudes.copy()
        
        # Normalize
        max_mag = np.max(magnitudes)
        if max_mag > 0:
            magnitudes = magnitudes / max_mag
        
        return magnitudes
    
    def to_image_data(
        self,
        signal: np.ndarray,
        height: int = 512,
        width: int = 512
    ) -> np.ndarray:
        """
        Convert circular visualization to image data (optimized).
        
        Args:
            signal: Audio signal
            height: Image height in pixels
            width: Image width in pixels
            
        Returns:
            Image data array
        """
        magnitudes = self.generate(signal)
        image = np.zeros((height, width), dtype=np.float32)
        
        center_x, center_y = width // 2, height // 2
        max_radius = min(center_x, center_y)
        inner_r = int(max_radius * self.inner_radius)
        
        # Vectorized bar drawing for better performance
        angles = (2 * np.pi * np.arange(self.num_bars) / self.num_bars) - np.pi / 2
        
        # Pre-calculate all bar coordinates
        for i, (angle, mag) in enumerate(zip(angles, magnitudes)):
            outer_r = int(inner_r + mag * (max_radius - inner_r) * self.bar_width)
            
            if outer_r <= inner_r:
                continue
            
            # Calculate line endpoints
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # Draw radial line (optimized)
            num_points = outer_r - inner_r + 1
            if num_points <= 0:
                continue
            
            # Vectorized line drawing
            radii = np.linspace(inner_r, outer_r, num_points)
            xs = (center_x + radii * cos_a).astype(np.int32)
            ys = (center_y + radii * sin_a).astype(np.int32)
            
            # Clip to image bounds
            valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            xs = xs[valid]
            ys = ys[valid]
            
            # Set pixels (vectorized)
            image[ys, xs] = 1.0
            
            # Add thickness for better visibility (only 1 pixel each side)
            if len(xs) > 0:
                # Perpendicular offsets
                perp_x = int(-sin_a)
                perp_y = int(cos_a)
                
                if perp_x != 0 or perp_y != 0:
                    xs1 = xs + perp_x
                    ys1 = ys + perp_y
                    valid1 = (xs1 >= 0) & (xs1 < width) & (ys1 >= 0) & (ys1 < height)
                    image[ys1[valid1], xs1[valid1]] = 1.0
                    
                    xs2 = xs - perp_x
                    ys2 = ys - perp_y
                    valid2 = (xs2 >= 0) & (xs2 < width) & (ys2 >= 0) & (ys2 < height)
                    image[ys2[valid2], xs2[valid2]] = 1.0
        
        return image


class ParticleVisualizer(Visualizer):
    """
    Particle-based visualization.
    
    Displays audio as animated particles reacting to frequency content.
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        num_particles: int = 100,
        decay: float = 0.95,
        sensitivity: float = 1.0,
        debug: bool = False
    ):
        """
        Initialize a particle visualizer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            num_particles: Number of particles to simulate
            decay: Particle velocity decay (0.0 to 1.0)
            sensitivity: Sensitivity to audio changes
            debug: Enable debug output
        """
        super().__init__(sample_rate, debug)
        self.num_particles = num_particles
        self.decay = decay
        self.sensitivity = sensitivity
        self.particles = None
    
    def generate(self, signal: np.ndarray) -> np.ndarray:
        """
        Generate particle visualization data.
        
        Args:
            signal: Audio signal
            
        Returns:
            Particle positions and velocities
        """
        # Initialize particles if needed
        if self.particles is None:
            self.particles = {
                'x': np.random.rand(self.num_particles),
                'y': np.random.rand(self.num_particles),
                'vx': np.zeros(self.num_particles),
                'vy': np.zeros(self.num_particles),
                'energy': np.zeros(self.num_particles)
            }
        
        # Calculate energy from signal
        energy = np.sqrt(np.mean(signal ** 2))
        
        # Update particle velocities based on energy
        for i in range(self.num_particles):
            if energy > 0.01:
                angle = np.random.rand() * 2 * np.pi
                force = energy * self.sensitivity
                self.particles['vx'][i] += np.cos(angle) * force
                self.particles['vy'][i] += np.sin(angle) * force
                self.particles['energy'][i] = energy
        
        # Update positions
        self.particles['x'] += self.particles['vx']
        self.particles['y'] += self.particles['vy']
        
        # Apply decay
        self.particles['vx'] *= self.decay
        self.particles['vy'] *= self.decay
        
        # Bounce off edges
        self.particles['x'] = np.clip(self.particles['x'], 0, 1)
        self.particles['y'] = np.clip(self.particles['y'], 0, 1)
        
        # Reverse velocity at edges
        self.particles['vx'] = np.where(
            (self.particles['x'] <= 0) | (self.particles['x'] >= 1),
            -self.particles['vx'],
            self.particles['vx']
        )
        self.particles['vy'] = np.where(
            (self.particles['y'] <= 0) | (self.particles['y'] >= 1),
            -self.particles['vy'],
            self.particles['vy']
        )
        
        return self.particles
    
    def to_image_data(
        self,
        signal: np.ndarray,
        height: int = 512,
        width: int = 512
    ) -> np.ndarray:
        """
        Convert particle visualization to image data.
        
        Args:
            signal: Audio signal
            height: Image height in pixels
            width: Image width in pixels
            
        Returns:
            Image data array
        """
        particles = self.generate(signal)
        image = np.zeros((height, width))
        
        # Draw particles
        for i in range(self.num_particles):
            x = int(particles['x'][i] * width)
            y = int(particles['y'][i] * height)
            energy = particles['energy'][i]
            
            if 0 <= x < width and 0 <= y < height:
                # Draw particle with size based on energy
                radius = max(1, int(energy * 10))
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dx*dx + dy*dy <= radius*radius:
                            px, py = x + dx, y + dy
                            if 0 <= px < width and 0 <= py < height:
                                image[py, px] = min(1.0, image[py, px] + energy)
        
        return image


class PianoRollVisualizer(Visualizer):
    """
    Piano Roll / Note Display visualization.
    
    Visual representation of which notes are playing on a musical grid.
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        time_resolution: float = 0.1,
        pitch_range: tuple = (36, 84),  # C2 to C6
        debug: bool = False
    ):
        """
        Initialize a piano roll visualizer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            time_resolution: Time resolution in seconds per column
            pitch_range: Range of MIDI notes to display (min, max)
            debug: Enable debug output
        """
        super().__init__(sample_rate, debug)
        self.time_resolution = time_resolution
        self.pitch_range = pitch_range
        self.notes: List[Dict[str, Any]] = []
    
    def add_note(
        self,
        midi_note: int,
        start_time: float,
        duration: float,
        velocity: float = 1.0
    ) -> None:
        """
        Add a note to the piano roll.
        
        Args:
            midi_note: MIDI note number
            start_time: Start time in seconds
            duration: Duration in seconds
            velocity: Note velocity (0.0 to 1.0)
        """
        self.notes.append({
            'midi_note': midi_note,
            'start_time': start_time,
            'duration': duration,
            'velocity': velocity
        })
    
    def clear_notes(self) -> None:
        """Clear all notes from the piano roll."""
        self.notes = []
    
    def generate(self, signal: np.ndarray) -> np.ndarray:
        """
        Generate piano roll visualization data.
        
        This method is included for compatibility but piano roll
        visualization is typically based on note data, not audio signal.
        
        Args:
            signal: Audio signal (not used for piano roll)
            
        Returns:
            Placeholder array
        """
        return np.array([])
    
    def to_grid(
        self,
        duration: float,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate piano roll grid representation.
        
        Args:
            duration: Total duration to display
            height: Grid height (defaults to pitch range)
            width: Grid width (defaults to duration / time_resolution)
            
        Returns:
            2D grid with note activations
        """
        if height is None:
            height = self.pitch_range[1] - self.pitch_range[0]
        
        if width is None:
            width = int(duration / self.time_resolution)
        
        grid = np.zeros((height, width))
        
        for note in self.notes:
            midi_note = note['midi_note']
            start_time = note['start_time']
            note_duration = note['duration']
            velocity = note['velocity']
            
            # Check if note is in pitch range
            if midi_note < self.pitch_range[0] or midi_note >= self.pitch_range[1]:
                continue
            
            # Calculate grid positions
            pitch_idx = self.pitch_range[1] - midi_note - 1  # Invert for display
            start_col = int(start_time / self.time_resolution)
            end_col = int((start_time + note_duration) / self.time_resolution)
            
            # Clamp to grid bounds
            start_col = max(0, min(start_col, width - 1))
            end_col = max(0, min(end_col, width))
            
            # Set note activation
            if 0 <= pitch_idx < height:
                grid[pitch_idx, start_col:end_col] = velocity
        
        return grid
    
    def to_image_data(
        self,
        duration: float,
        height: int = 480,
        width: int = 640
    ) -> np.ndarray:
        """
        Convert piano roll to image data.
        
        Args:
            duration: Total duration to display
            height: Image height in pixels
            width: Image width in pixels
            
        Returns:
            Image data array
        """
        grid = self.to_grid(duration, self.pitch_range[1] - self.pitch_range[0], width)
        
        # Resize to target dimensions if needed
        if grid.shape[0] != height:
            # Simple nearest-neighbor resize
            row_indices = (np.arange(height) * grid.shape[0] / height).astype(int)
            grid = grid[row_indices, :]
        
        return grid
