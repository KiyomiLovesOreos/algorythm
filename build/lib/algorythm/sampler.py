"""
algorythm.sampler - Sample playback and manipulation

This module provides tools for loading and triggering external audio files
as instruments or one-shots.
"""

from typing import Optional, Literal
import numpy as np
import wave
import struct


class Sample:
    """
    Audio sample loaded from a file.
    
    Supports WAV and other audio formats.
    """
    
    def __init__(
        self,
        file_path: Optional[str] = None,
        data: Optional[np.ndarray] = None,
        sample_rate: int = 44100
    ):
        """
        Initialize a sample.
        
        Args:
            file_path: Path to audio file (WAV, AIFF, etc.)
            data: Pre-loaded audio data (if not loading from file)
            sample_rate: Sample rate in Hz
        """
        self.file_path = file_path
        self.sample_rate = sample_rate
        
        if data is not None:
            self.data = data
        elif file_path:
            self.data, self.sample_rate = self._load_from_file(file_path)
        else:
            self.data = np.array([])
    
    def _load_from_file(self, file_path: str) -> tuple[np.ndarray, int]:
        """
        Load audio data from file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio data, sample rate)
        """
        # Try to load as WAV first
        if file_path.lower().endswith('.wav'):
            return self._load_wav(file_path)
        elif file_path.lower().endswith('.aiff') or file_path.lower().endswith('.aif'):
            # AIFF support would require additional library
            print(f"Note: AIFF support requires additional dependencies. Please convert to WAV.")
            return np.array([]), 44100
        else:
            print(f"Note: Unsupported file format. Please use WAV files.")
            return np.array([]), 44100
    
    def _load_wav(self, file_path: str) -> tuple[np.ndarray, int]:
        """
        Load WAV file.
        
        Args:
            file_path: Path to WAV file
            
        Returns:
            Tuple of (audio data, sample rate)
        """
        try:
            with wave.open(file_path, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                num_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                num_frames = wav_file.getnframes()
                
                # Read raw data
                raw_data = wav_file.readframes(num_frames)
                
                # Convert to numpy array based on bit depth
                if sample_width == 1:
                    # 8-bit unsigned
                    data = np.frombuffer(raw_data, dtype=np.uint8)
                    data = (data.astype(np.float32) - 128) / 128.0
                elif sample_width == 2:
                    # 16-bit signed
                    data = np.frombuffer(raw_data, dtype=np.int16)
                    data = data.astype(np.float32) / 32768.0
                elif sample_width == 3:
                    # 24-bit signed (convert to int32)
                    data = []
                    for i in range(0, len(raw_data), 3):
                        sample = struct.unpack('<i', raw_data[i:i+3] + b'\x00')[0]
                        data.append(sample / 8388608.0)
                    data = np.array(data, dtype=np.float32)
                else:
                    # Assume 32-bit float
                    data = np.frombuffer(raw_data, dtype=np.float32)
                
                # Convert stereo to mono if needed
                if num_channels == 2:
                    data = data.reshape(-1, 2).mean(axis=1)
                elif num_channels > 2:
                    data = data.reshape(-1, num_channels).mean(axis=1)
                
                return data, sample_rate
        
        except Exception as e:
            print(f"Error loading WAV file: {e}")
            return np.array([]), 44100
    
    def get_duration(self) -> float:
        """
        Get sample duration in seconds.
        
        Returns:
            Duration in seconds
        """
        if len(self.data) == 0:
            return 0.0
        return len(self.data) / self.sample_rate
    
    def resample(self, target_rate: int) -> 'Sample':
        """
        Resample to a different sample rate.
        
        Args:
            target_rate: Target sample rate in Hz
            
        Returns:
            New Sample instance with resampled data
        """
        if target_rate == self.sample_rate or len(self.data) == 0:
            return Sample(data=self.data.copy(), sample_rate=self.sample_rate)
        
        # Simple linear interpolation resampling
        ratio = target_rate / self.sample_rate
        new_length = int(len(self.data) * ratio)
        
        old_indices = np.linspace(0, len(self.data) - 1, len(self.data))
        new_indices = np.linspace(0, len(self.data) - 1, new_length)
        
        resampled_data = np.interp(new_indices, old_indices, self.data)
        
        return Sample(data=resampled_data, sample_rate=target_rate)
    
    def trim(self, start_time: float = 0.0, end_time: Optional[float] = None) -> 'Sample':
        """
        Trim sample to a specific time range.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds (None for end of sample)
            
        Returns:
            New Sample instance with trimmed data
        """
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate) if end_time else len(self.data)
        
        start_sample = max(0, start_sample)
        end_sample = min(len(self.data), end_sample)
        
        trimmed_data = self.data[start_sample:end_sample]
        
        return Sample(data=trimmed_data, sample_rate=self.sample_rate)
    
    def normalize(self, target_level: float = 1.0) -> 'Sample':
        """
        Normalize sample to target level.
        
        Args:
            target_level: Target peak level
            
        Returns:
            New Sample instance with normalized data
        """
        if len(self.data) == 0:
            return Sample(data=self.data.copy(), sample_rate=self.sample_rate)
        
        max_amplitude = np.max(np.abs(self.data))
        if max_amplitude > 0:
            normalized_data = self.data * (target_level / max_amplitude)
        else:
            normalized_data = self.data.copy()
        
        return Sample(data=normalized_data, sample_rate=self.sample_rate)


class Sampler:
    """
    Sample playback engine.
    
    Triggers and manipulates audio samples.
    """
    
    def __init__(self, sample: Sample):
        """
        Initialize a sampler.
        
        Args:
            sample: Sample to use for playback
        """
        self.sample = sample
    
    def trigger(
        self,
        pitch_shift: float = 0.0,
        volume: float = 1.0,
        duration: Optional[float] = None
    ) -> np.ndarray:
        """
        Trigger sample playback.
        
        Args:
            pitch_shift: Pitch shift in semitones
            volume: Volume multiplier
            duration: Playback duration (None for full sample)
            
        Returns:
            Audio signal
        """
        if len(self.sample.data) == 0:
            return np.array([])
        
        output = self.sample.data.copy()
        
        # Apply pitch shift
        if pitch_shift != 0.0:
            # Simple pitch shifting via sample rate modification
            # (real implementation would use phase vocoder or similar)
            ratio = 2.0 ** (pitch_shift / 12.0)
            new_length = int(len(output) / ratio)
            
            old_indices = np.linspace(0, len(output) - 1, len(output))
            new_indices = np.linspace(0, len(output) - 1, new_length)
            
            output = np.interp(new_indices, old_indices, output)
        
        # Trim to duration if specified
        if duration is not None:
            target_samples = int(duration * self.sample.sample_rate)
            if len(output) > target_samples:
                output = output[:target_samples]
            elif len(output) < target_samples:
                output = np.pad(output, (0, target_samples - len(output)))
        
        # Apply volume
        output = output * volume
        
        return output
    
    def trigger_note(
        self,
        frequency: float,
        base_frequency: float = 440.0,
        volume: float = 1.0,
        duration: Optional[float] = None
    ) -> np.ndarray:
        """
        Trigger sample at a specific pitch/frequency.
        
        Args:
            frequency: Target frequency in Hz
            base_frequency: Base frequency of the sample
            volume: Volume multiplier
            duration: Playback duration
            
        Returns:
            Audio signal
        """
        # Calculate pitch shift in semitones
        pitch_shift = 12.0 * np.log2(frequency / base_frequency)
        
        return self.trigger(pitch_shift, volume, duration)
    
    def create_loop(self, num_loops: int = 2, fade_time: float = 0.01) -> np.ndarray:
        """
        Create a looped version of the sample.
        
        Args:
            num_loops: Number of times to loop
            fade_time: Crossfade time between loops in seconds
            
        Returns:
            Looped audio signal
        """
        if len(self.sample.data) == 0:
            return np.array([])
        
        # Create output array
        loop_length = len(self.sample.data)
        total_length = loop_length * num_loops
        output = np.zeros(total_length)
        
        fade_samples = int(fade_time * self.sample.sample_rate)
        
        for i in range(num_loops):
            start = i * loop_length
            end = start + loop_length
            
            # Copy loop data
            output[start:end] = self.sample.data
            
            # Apply crossfade at loop boundaries (except first)
            if i > 0 and fade_samples > 0:
                fade_start = start - fade_samples
                fade_end = start + fade_samples
                
                if fade_start >= 0 and fade_end < total_length:
                    # Create crossfade
                    fade_out = np.linspace(1, 0, fade_samples)
                    fade_in = np.linspace(0, 1, fade_samples)
                    
                    output[fade_start:start] *= fade_out
                    output[start:fade_end] *= fade_in
        
        return output
    
    @classmethod
    def from_file(cls, file_path: str) -> 'Sampler':
        """
        Create a sampler from an audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Sampler instance
        """
        sample = Sample(file_path=file_path)
        return cls(sample)


class GranularSynth:
    """
    Granular synthesis engine for creating rich textures and soundscapes.
    
    Breaks audio samples into tiny "grains" with programmatic control over
    grain density, size, position, speed, and spatial distribution.
    """
    
    def __init__(
        self,
        sample: Sample,
        grain_size: float = 0.05,
        grain_density: float = 20.0,
        grain_envelope: Literal['rectangular', 'triangular', 'gaussian', 'hann'] = 'hann'
    ):
        """
        Initialize a granular synthesizer.
        
        Args:
            sample: Audio sample to granulate
            grain_size: Size of each grain in seconds
            grain_density: Number of grains per second
            grain_envelope: Envelope shape for each grain
        """
        self.sample = sample
        self.grain_size = grain_size
        self.grain_density = grain_density
        self.grain_envelope = grain_envelope
    
    def generate_grain(
        self,
        position: float,
        size: float,
        pitch_shift: float = 0.0,
        pan: float = 0.5
    ) -> np.ndarray:
        """
        Generate a single grain from the sample.
        
        Args:
            position: Position in the sample (0.0 to 1.0)
            size: Grain size in seconds
            pitch_shift: Pitch shift in semitones
            pan: Stereo pan position (0.0 = left, 1.0 = right)
            
        Returns:
            Grain audio data (mono or stereo)
        """
        if len(self.sample.data) == 0:
            return np.array([])
        
        # Calculate grain start position in samples
        start_sample = int(position * len(self.sample.data))
        grain_samples = int(size * self.sample.sample_rate)
        
        # Extract grain
        end_sample = min(start_sample + grain_samples, len(self.sample.data))
        grain = self.sample.data[start_sample:end_sample].copy()
        
        # Apply pitch shift if needed
        if pitch_shift != 0.0:
            ratio = 2.0 ** (pitch_shift / 12.0)
            new_length = int(len(grain) / ratio)
            old_indices = np.linspace(0, len(grain) - 1, len(grain))
            new_indices = np.linspace(0, len(grain) - 1, new_length)
            grain = np.interp(new_indices, old_indices, grain)
        
        # Apply envelope
        envelope = self._create_envelope(len(grain))
        grain = grain * envelope
        
        return grain
    
    def _create_envelope(self, length: int) -> np.ndarray:
        """
        Create envelope for a grain.
        
        Args:
            length: Length of grain in samples
            
        Returns:
            Envelope array
        """
        t = np.linspace(0, 1, length)
        
        if self.grain_envelope == 'rectangular':
            return np.ones(length)
        elif self.grain_envelope == 'triangular':
            return 1 - np.abs(2 * t - 1)
        elif self.grain_envelope == 'gaussian':
            # Gaussian envelope
            center = 0.5
            width = 0.2
            return np.exp(-((t - center) ** 2) / (2 * width ** 2))
        elif self.grain_envelope == 'hann':
            # Hann window
            return 0.5 * (1 - np.cos(2 * np.pi * t))
        else:
            return np.ones(length)
    
    def synthesize(
        self,
        duration: float,
        position_range: tuple = (0.0, 1.0),
        pitch_range: tuple = (0.0, 0.0),
        spatial_spread: float = 0.0,
        density_variation: float = 0.0
    ) -> np.ndarray:
        """
        Synthesize granular texture over a specified duration.
        
        Args:
            duration: Output duration in seconds
            position_range: Range of sample positions to use (min, max) in 0.0-1.0
            pitch_range: Range of pitch shifts in semitones (min, max)
            spatial_spread: Amount of stereo spread (0.0 = mono, 1.0 = full stereo)
            density_variation: Random variation in grain density (0.0 to 1.0)
            
        Returns:
            Synthesized audio signal
        """
        total_samples = int(duration * self.sample.sample_rate)
        output = np.zeros(total_samples)
        
        # Calculate number of grains
        base_num_grains = int(duration * self.grain_density)
        
        # Apply density variation
        if density_variation > 0:
            variation = int(base_num_grains * density_variation)
            num_grains = base_num_grains + np.random.randint(-variation, variation + 1)
            num_grains = max(1, num_grains)
        else:
            num_grains = base_num_grains
        
        # Generate grains at random times
        grain_times = np.sort(np.random.uniform(0, duration, num_grains))
        
        for grain_time in grain_times:
            # Random position within range
            position = np.random.uniform(position_range[0], position_range[1])
            
            # Random pitch shift within range
            pitch_shift = np.random.uniform(pitch_range[0], pitch_range[1])
            
            # Random pan position
            pan = 0.5 + np.random.uniform(-spatial_spread, spatial_spread) * 0.5
            pan = np.clip(pan, 0.0, 1.0)
            
            # Generate grain
            grain = self.generate_grain(position, self.grain_size, pitch_shift, pan)
            
            # Place grain in output
            start_sample = int(grain_time * self.sample.sample_rate)
            end_sample = min(start_sample + len(grain), total_samples)
            grain_length = end_sample - start_sample
            
            if grain_length > 0:
                output[start_sample:end_sample] += grain[:grain_length]
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val * 0.9
        
        return output
    
    @classmethod
    def from_file(cls, file_path: str, **kwargs) -> 'GranularSynth':
        """
        Create a granular synth from an audio file.
        
        Args:
            file_path: Path to audio file
            **kwargs: Additional arguments for GranularSynth
            
        Returns:
            GranularSynth instance
        """
        sample = Sample(file_path=file_path)
        return cls(sample, **kwargs)
