"""Tests for algorythm.export module."""

import numpy as np
import pytest
import os
import tempfile
from algorythm.export import RenderEngine, Exporter


class TestRenderEngine:
    """Tests for RenderEngine class."""
    
    def test_render_engine_creation(self):
        """Test creating a render engine."""
        engine = RenderEngine(sample_rate=44100)
        assert engine.sample_rate == 44100
    
    def test_normalize(self):
        """Test signal normalization."""
        engine = RenderEngine()
        signal = np.array([0.5, 1.0, -0.5, -1.0])
        normalized = engine.normalize(signal, target_level=0.9)
        assert np.max(np.abs(normalized)) <= 0.9
    
    def test_apply_fade(self):
        """Test applying fade in/out."""
        engine = RenderEngine(sample_rate=44100)
        signal = np.ones(44100)
        faded = engine.apply_fade(signal, fade_in=0.1, fade_out=0.1)
        assert len(faded) == len(signal)
        # Check that fade in starts at 0
        assert faded[0] < 0.1
        # Check that fade out ends at 0
        assert abs(faded[-1]) < 0.1


class TestExporter:
    """Tests for Exporter class."""
    
    def test_exporter_creation(self):
        """Test creating an exporter."""
        exporter = Exporter()
        assert exporter.render_engine is not None
    
    def test_export_wav(self):
        """Test exporting to WAV format."""
        exporter = Exporter()
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_file = f.name
        
        try:
            exporter.export(signal, temp_file, sample_rate=44100)
            assert os.path.exists(temp_file)
            assert os.path.getsize(temp_file) > 0
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_export_different_bit_depths(self):
        """Test exporting with different bit depths."""
        exporter = Exporter()
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        
        for bit_depth in [16, 24]:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_file = f.name
            
            try:
                exporter._export_wav(signal, temp_file, 44100, bit_depth)
                assert os.path.exists(temp_file)
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
    
    def test_export_mp4_with_visualizer(self):
        """Test exporting to MP4 format with visualizer."""
        try:
            import cv2
            
            from algorythm.visualization import WaveformVisualizer
            
            exporter = Exporter()
            signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 22050))  # Short 0.5s signal
            visualizer = WaveformVisualizer(sample_rate=44100)
            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                temp_file = f.name
            
            try:
                exporter.export(
                    signal,
                    temp_file,
                    sample_rate=44100,
                    visualizer=visualizer,
                    video_width=640,
                    video_height=480,
                    video_fps=30
                )
                assert os.path.exists(temp_file)
                assert os.path.getsize(temp_file) > 0
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
        except ImportError:
            pytest.skip("opencv-python not installed, skipping MP4 test")
    
    def test_export_mp4_no_visualizer(self):
        """Test exporting to MP4 without specifying visualizer (uses default)."""
        try:
            import cv2
            
            exporter = Exporter()
            signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 22050))  # Short 0.5s signal
            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                temp_file = f.name
            
            try:
                exporter.export(
                    signal,
                    temp_file,
                    sample_rate=44100,
                    video_width=640,
                    video_height=480,
                    video_fps=30
                )
                assert os.path.exists(temp_file)
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
        except ImportError:
            pytest.skip("opencv-python not installed, skipping MP4 test")
