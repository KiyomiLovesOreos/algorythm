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
