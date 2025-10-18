"""
Tests for visualization components.
"""

import pytest
import numpy as np
from algorythm.visualization import (
    WaveformVisualizer,
    SpectrogramVisualizer,
    FrequencyScopeVisualizer,
    VideoRenderer
)


class TestWaveformVisualizer:
    """Tests for waveform visualizer."""
    
    def test_waveform_creation(self):
        """Test waveform visualizer creation."""
        viz = WaveformVisualizer(sample_rate=44100, window_size=1024)
        assert viz.sample_rate == 44100
        assert viz.window_size == 1024
    
    def test_waveform_generate(self):
        """Test generating waveform data."""
        viz = WaveformVisualizer(window_size=512)
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        
        result = viz.generate(signal)
        assert result.shape[0] > 0
    
    def test_waveform_to_image_data(self):
        """Test converting waveform to image."""
        viz = WaveformVisualizer()
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 1024))
        
        image = viz.to_image_data(signal, height=256, width=1024)
        assert image.shape == (256, 1024)


class TestSpectrogramVisualizer:
    """Tests for spectrogram visualizer."""
    
    def test_spectrogram_creation(self):
        """Test spectrogram visualizer creation."""
        viz = SpectrogramVisualizer(sample_rate=44100, window_size=2048)
        assert viz.sample_rate == 44100
        assert viz.window_size == 2048
    
    def test_spectrogram_generate(self):
        """Test generating spectrogram."""
        viz = SpectrogramVisualizer(window_size=512, hop_size=256)
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        
        spec = viz.generate(signal)
        assert spec.shape[0] > 0  # Frequency bins
        assert spec.shape[1] > 0  # Time frames
    
    def test_spectrogram_get_time_axis(self):
        """Test getting time axis."""
        viz = SpectrogramVisualizer(hop_size=512)
        time_axis = viz.get_time_axis(num_frames=10)
        
        assert len(time_axis) == 10
        assert time_axis[0] < time_axis[-1]
    
    def test_spectrogram_get_frequency_axis(self):
        """Test getting frequency axis."""
        viz = SpectrogramVisualizer(sample_rate=44100, window_size=2048)
        freq_axis = viz.get_frequency_axis()
        
        assert len(freq_axis) > 0
        assert freq_axis[0] == 0.0
        assert freq_axis[-1] == 22050.0  # Nyquist frequency


class TestFrequencyScopeVisualizer:
    """Tests for frequency scope visualizer."""
    
    def test_frequency_scope_creation(self):
        """Test frequency scope creation."""
        viz = FrequencyScopeVisualizer(sample_rate=44100, fft_size=2048)
        assert viz.sample_rate == 44100
        assert viz.fft_size == 2048
    
    def test_frequency_scope_generate(self):
        """Test generating frequency spectrum."""
        viz = FrequencyScopeVisualizer(fft_size=2048)
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 2048))
        
        spectrum = viz.generate(signal)
        assert len(spectrum) > 0
    
    def test_frequency_scope_get_bins(self):
        """Test getting frequency bins."""
        viz = FrequencyScopeVisualizer(sample_rate=44100, fft_size=2048)
        bins = viz.get_frequency_bins()
        
        assert len(bins) == 1025  # fft_size // 2 + 1
        assert bins[0] == 0.0
    
    def test_frequency_scope_filter_range(self):
        """Test filtering frequency range."""
        viz = FrequencyScopeVisualizer(
            sample_rate=44100,
            fft_size=2048,
            freq_range=(100.0, 1000.0)
        )
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 2048))
        spectrum = viz.generate(signal)
        
        freqs, mags = viz.filter_frequency_range(spectrum)
        
        assert len(freqs) == len(mags)
        assert all(100.0 <= f <= 1000.0 for f in freqs)


class TestVideoRenderer:
    """Tests for video renderer."""
    
    def test_video_renderer_creation(self):
        """Test video renderer creation."""
        renderer = VideoRenderer(width=1920, height=1080, fps=30)
        assert renderer.width == 1920
        assert renderer.height == 1080
        assert renderer.fps == 30
    
    def test_video_renderer_render_frames(self):
        """Test rendering video frames."""
        renderer = VideoRenderer(width=640, height=480, fps=10)
        viz = WaveformVisualizer()
        
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        frames = renderer.render_frames(signal, viz)
        
        assert len(frames) > 0
        assert all(isinstance(f, np.ndarray) for f in frames)
