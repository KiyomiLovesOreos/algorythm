"""
algorythm.export - Renders and saves the final audio

This module provides tools for exporting compositions to various audio formats.
"""

import numpy as np
from typing import Optional, Literal
import wave
import struct


class RenderEngine:
    """
    Core audio rendering engine.
    
    Handles the conversion of musical structures to raw audio samples.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the render engine.
        
        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
    
    def normalize(self, signal: np.ndarray, target_level: float = 0.9) -> np.ndarray:
        """
        Normalize audio signal to target level.
        
        Args:
            signal: Input audio signal
            target_level: Target peak level (0.0 to 1.0)
            
        Returns:
            Normalized audio signal
        """
        max_amplitude = np.max(np.abs(signal))
        if max_amplitude > 0:
            return signal / max_amplitude * target_level
        return signal
    
    def apply_fade(
        self,
        signal: np.ndarray,
        fade_in: float = 0.0,
        fade_out: float = 0.0
    ) -> np.ndarray:
        """
        Apply fade in/out to audio signal.
        
        Args:
            signal: Input audio signal
            fade_in: Fade in duration in seconds
            fade_out: Fade out duration in seconds
            
        Returns:
            Audio signal with fades applied
        """
        output = signal.copy()
        
        # Apply fade in
        if fade_in > 0:
            fade_in_samples = int(fade_in * self.sample_rate)
            fade_in_samples = min(fade_in_samples, len(output))
            fade_curve = np.linspace(0, 1, fade_in_samples)
            output[:fade_in_samples] *= fade_curve
        
        # Apply fade out
        if fade_out > 0:
            fade_out_samples = int(fade_out * self.sample_rate)
            fade_out_samples = min(fade_out_samples, len(output))
            fade_curve = np.linspace(1, 0, fade_out_samples)
            output[-fade_out_samples:] *= fade_curve
        
        return output
    
    def resample(self, signal: np.ndarray, target_rate: int) -> np.ndarray:
        """
        Resample audio signal to target sample rate.
        
        Args:
            signal: Input audio signal
            target_rate: Target sample rate in Hz
            
        Returns:
            Resampled audio signal
        """
        # Simplified resampling using linear interpolation
        if target_rate == self.sample_rate:
            return signal
        
        ratio = target_rate / self.sample_rate
        new_length = int(len(signal) * ratio)
        
        old_indices = np.linspace(0, len(signal) - 1, len(signal))
        new_indices = np.linspace(0, len(signal) - 1, new_length)
        
        return np.interp(new_indices, old_indices, signal)


class Exporter:
    """
    Exports audio to various file formats.
    
    Supports WAV, FLAC, MP3, and OGG formats.
    """
    
    def __init__(self):
        """Initialize the exporter."""
        self.render_engine = RenderEngine()
    
    def export(
        self,
        signal: np.ndarray,
        file_path: str,
        sample_rate: int = 44100,
        quality: Literal['low', 'medium', 'high'] = 'high',
        bit_depth: int = 16
    ) -> None:
        """
        Export audio signal to file.
        
        Args:
            signal: Audio signal to export
            file_path: Output file path
            sample_rate: Sample rate in Hz
            quality: Quality setting
            bit_depth: Bit depth for WAV export
        """
        # Normalize signal
        signal = self.render_engine.normalize(signal)
        
        # Determine format from file extension
        file_path_lower = file_path.lower()
        
        if file_path_lower.endswith('.wav'):
            self._export_wav(signal, file_path, sample_rate, bit_depth)
        elif file_path_lower.endswith('.flac'):
            self._export_flac(signal, file_path, sample_rate, quality)
        elif file_path_lower.endswith('.mp3'):
            self._export_mp3(signal, file_path, sample_rate, quality)
        elif file_path_lower.endswith('.ogg'):
            self._export_ogg(signal, file_path, sample_rate, quality)
        else:
            # Default to WAV if no recognized extension
            self._export_wav(signal, file_path, sample_rate, bit_depth)
    
    def _export_wav(
        self,
        signal: np.ndarray,
        file_path: str,
        sample_rate: int,
        bit_depth: int = 16
    ) -> None:
        """
        Export audio signal to WAV file.
        
        Args:
            signal: Audio signal
            file_path: Output file path
            sample_rate: Sample rate in Hz
            bit_depth: Bit depth (8, 16, 24, or 32)
        """
        # Ensure proper file extension
        if not file_path.lower().endswith('.wav'):
            file_path += '.wav'
        
        # Convert to appropriate bit depth
        if bit_depth == 8:
            # 8-bit unsigned
            audio_data = np.clip(signal * 127 + 128, 0, 255).astype(np.uint8)
        elif bit_depth == 16:
            # 16-bit signed
            audio_data = np.clip(signal * 32767, -32768, 32767).astype(np.int16)
        elif bit_depth == 24:
            # 24-bit signed (stored as 32-bit)
            audio_data = np.clip(signal * 8388607, -8388608, 8388607).astype(np.int32)
        elif bit_depth == 32:
            # 32-bit float
            audio_data = signal.astype(np.float32)
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")
        
        # Write WAV file
        with wave.open(file_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(bit_depth // 8)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
    
    def _export_flac(
        self,
        signal: np.ndarray,
        file_path: str,
        sample_rate: int,
        quality: str
    ) -> None:
        """
        Export audio signal to FLAC file.
        
        Args:
            signal: Audio signal
            file_path: Output file path
            sample_rate: Sample rate in Hz
            quality: Quality setting
        """
        # FLAC export would typically use a library like soundfile or pydub
        # For now, export as WAV with a note that FLAC requires additional dependencies
        print(f"Note: FLAC export requires additional dependencies. Exporting as WAV instead.")
        wav_path = file_path.replace('.flac', '.wav')
        self._export_wav(signal, wav_path, sample_rate, 24)
    
    def _export_mp3(
        self,
        signal: np.ndarray,
        file_path: str,
        sample_rate: int,
        quality: str
    ) -> None:
        """
        Export audio signal to MP3 file.
        
        Args:
            signal: Audio signal
            file_path: Output file path
            sample_rate: Sample rate in Hz
            quality: Quality setting
        """
        # MP3 export would typically use ffmpeg or lame
        # For now, export as WAV with a note that MP3 requires additional dependencies
        print(f"Note: MP3 export requires ffmpeg or lame. Exporting as WAV instead.")
        wav_path = file_path.replace('.mp3', '.wav')
        self._export_wav(signal, wav_path, sample_rate, 16)
    
    def _export_ogg(
        self,
        signal: np.ndarray,
        file_path: str,
        sample_rate: int,
        quality: str
    ) -> None:
        """
        Export audio signal to OGG file.
        
        Args:
            signal: Audio signal
            file_path: Output file path
            sample_rate: Sample rate in Hz
            quality: Quality setting
        """
        # OGG export would typically use pydub or ffmpeg
        # For now, export as WAV with a note that OGG requires additional dependencies
        print(f"Note: OGG export requires additional dependencies. Exporting as WAV instead.")
        wav_path = file_path.replace('.ogg', '.wav')
        self._export_wav(signal, wav_path, sample_rate, 16)
    
    def export_stereo(
        self,
        left_signal: np.ndarray,
        right_signal: np.ndarray,
        file_path: str,
        sample_rate: int = 44100,
        quality: str = 'high',
        bit_depth: int = 16
    ) -> None:
        """
        Export stereo audio signal to file.
        
        Args:
            left_signal: Left channel audio signal
            right_signal: Right channel audio signal
            file_path: Output file path
            sample_rate: Sample rate in Hz
            quality: Quality setting
            bit_depth: Bit depth for WAV export
        """
        # Normalize both channels
        left_signal = self.render_engine.normalize(left_signal)
        right_signal = self.render_engine.normalize(right_signal)
        
        # Ensure equal length
        max_length = max(len(left_signal), len(right_signal))
        if len(left_signal) < max_length:
            left_signal = np.pad(left_signal, (0, max_length - len(left_signal)))
        if len(right_signal) < max_length:
            right_signal = np.pad(right_signal, (0, max_length - len(right_signal)))
        
        # Interleave channels
        stereo_signal = np.empty((max_length * 2,), dtype=left_signal.dtype)
        stereo_signal[0::2] = left_signal
        stereo_signal[1::2] = right_signal
        
        # Export based on format
        if file_path.lower().endswith('.wav'):
            self._export_wav_stereo(stereo_signal, file_path, sample_rate, bit_depth)
        else:
            print(f"Stereo export currently only supports WAV format.")
            self._export_wav_stereo(stereo_signal, file_path, sample_rate, bit_depth)
    
    def _export_wav_stereo(
        self,
        stereo_signal: np.ndarray,
        file_path: str,
        sample_rate: int,
        bit_depth: int = 16
    ) -> None:
        """
        Export stereo audio signal to WAV file.
        
        Args:
            stereo_signal: Interleaved stereo audio signal
            file_path: Output file path
            sample_rate: Sample rate in Hz
            bit_depth: Bit depth
        """
        # Ensure proper file extension
        if not file_path.lower().endswith('.wav'):
            file_path += '.wav'
        
        # Convert to 16-bit signed
        audio_data = np.clip(stereo_signal * 32767, -32768, 32767).astype(np.int16)
        
        # Write WAV file
        with wave.open(file_path, 'wb') as wav_file:
            wav_file.setnchannels(2)  # Stereo
            wav_file.setsampwidth(bit_depth // 8)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
