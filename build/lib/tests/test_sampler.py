"""
Tests for sampler and sample playback.
"""

import pytest
import numpy as np
from algorythm.sampler import Sample, Sampler
import tempfile
import wave
import os


class TestSample:
    """Tests for Sample class."""
    
    def test_sample_creation_from_data(self):
        """Test creating sample from data."""
        data = np.random.randn(1000)
        sample = Sample(data=data, sample_rate=44100)
        
        assert len(sample.data) == 1000
        assert sample.sample_rate == 44100
    
    def test_sample_get_duration(self):
        """Test getting sample duration."""
        data = np.random.randn(44100)  # 1 second at 44.1kHz
        sample = Sample(data=data, sample_rate=44100)
        
        duration = sample.get_duration()
        assert abs(duration - 1.0) < 0.01
    
    def test_sample_resample(self):
        """Test resampling."""
        data = np.random.randn(44100)
        sample = Sample(data=data, sample_rate=44100)
        
        resampled = sample.resample(22050)
        assert resampled.sample_rate == 22050
        assert len(resampled.data) < len(sample.data)
    
    def test_sample_trim(self):
        """Test trimming sample."""
        data = np.random.randn(44100)  # 1 second
        sample = Sample(data=data, sample_rate=44100)
        
        trimmed = sample.trim(start_time=0.25, end_time=0.75)
        expected_length = int(0.5 * 44100)
        
        assert abs(len(trimmed.data) - expected_length) < 10
    
    def test_sample_normalize(self):
        """Test normalizing sample."""
        data = np.random.randn(1000) * 0.5
        sample = Sample(data=data, sample_rate=44100)
        
        normalized = sample.normalize(target_level=1.0)
        max_amp = np.max(np.abs(normalized.data))
        
        assert abs(max_amp - 1.0) < 0.01
    
    def test_sample_load_wav(self):
        """Test loading WAV file."""
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Generate test audio
            test_data = (np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)) * 32767).astype(np.int16)
            
            # Write WAV file
            with wave.open(tmp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(44100)
                wav_file.writeframes(test_data.tobytes())
            
            # Load sample
            sample = Sample(file_path=tmp_path)
            
            assert len(sample.data) > 0
            assert sample.sample_rate == 44100
        
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestSampler:
    """Tests for Sampler class."""
    
    def test_sampler_creation(self):
        """Test sampler creation."""
        data = np.random.randn(1000)
        sample = Sample(data=data, sample_rate=44100)
        sampler = Sampler(sample)
        
        assert sampler.sample == sample
    
    def test_sampler_trigger(self):
        """Test triggering sample."""
        data = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410))
        sample = Sample(data=data, sample_rate=44100)
        sampler = Sampler(sample)
        
        output = sampler.trigger(pitch_shift=0.0, volume=1.0)
        
        assert len(output) > 0
    
    def test_sampler_trigger_with_pitch_shift(self):
        """Test triggering with pitch shift."""
        data = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410))
        sample = Sample(data=data, sample_rate=44100)
        sampler = Sampler(sample)
        
        # Pitch shift up by 12 semitones (1 octave)
        output = sampler.trigger(pitch_shift=12.0, volume=1.0)
        
        # Pitched up sample should be shorter
        assert len(output) < len(data)
    
    def test_sampler_trigger_note(self):
        """Test triggering at specific frequency."""
        data = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410))
        sample = Sample(data=data, sample_rate=44100)
        sampler = Sampler(sample)
        
        # Trigger at 880 Hz (one octave up from 440 Hz)
        output = sampler.trigger_note(frequency=880.0, base_frequency=440.0, volume=1.0)
        
        assert len(output) > 0
    
    def test_sampler_create_loop(self):
        """Test creating looped sample."""
        data = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410))
        sample = Sample(data=data, sample_rate=44100)
        sampler = Sampler(sample)
        
        looped = sampler.create_loop(num_loops=3)
        
        # Looped sample should be approximately 3x the length
        assert len(looped) >= len(data) * 2.5
    
    def test_sampler_from_file(self):
        """Test creating sampler from file."""
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Generate test audio
            test_data = (np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)) * 32767).astype(np.int16)
            
            # Write WAV file
            with wave.open(tmp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(44100)
                wav_file.writeframes(test_data.tobytes())
            
            # Create sampler
            sampler = Sampler.from_file(tmp_path)
            
            assert len(sampler.sample.data) > 0
        
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
