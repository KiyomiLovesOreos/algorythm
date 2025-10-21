"""
algorythm.export - Renders and saves the final audio

This module provides tools for exporting compositions to various audio formats.
"""

import numpy as np
from typing import Optional, Literal
import wave
import struct
import os
from pathlib import Path


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
    By default, files are saved to ~/Music directory.
    """
    
    def __init__(self, default_directory: Optional[str] = None):
        """
        Initialize the exporter.
        
        Args:
            default_directory: Default directory for exports. If None, uses ~/Music
        """
        self.render_engine = RenderEngine()
        
        if default_directory is None:
            # Default to ~/Music
            self.default_directory = Path.home() / "Music"
        else:
            self.default_directory = Path(default_directory)
        
        # Create directory if it doesn't exist
        self.default_directory.mkdir(parents=True, exist_ok=True)
    
    def _resolve_path(self, file_path: str) -> str:
        """
        Resolve file path, using default directory if path is relative.
        
        Args:
            file_path: File path (can be absolute or relative)
            
        Returns:
            Absolute file path
        """
        path = Path(file_path)
        
        # If absolute path, use as-is
        if path.is_absolute():
            return str(path)
        
        # If relative path, prepend default directory
        return str(self.default_directory / file_path)
    
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
            file_path: Output file path (relative paths use ~/Music directory)
            sample_rate: Sample rate in Hz
            quality: Quality setting
            bit_depth: Bit depth for WAV export
        """
        # Resolve path (use ~/Music for relative paths)
        file_path = self._resolve_path(file_path)
        
        # Create parent directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
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
        
        print(f"Exported to: {file_path}")
    
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
        try:
            import soundfile as sf
            
            # Ensure proper file extension
            if not file_path.lower().endswith('.flac'):
                file_path += '.flac'
            
            # Convert to 24-bit for FLAC
            audio_data = np.clip(signal, -1.0, 1.0)
            
            # Write FLAC file
            sf.write(file_path, audio_data, sample_rate, subtype='PCM_24')
            
        except ImportError:
            print(f"Note: FLAC export requires soundfile library. Install with: pip install soundfile")
            print(f"Exporting as WAV instead.")
            wav_path = file_path.replace('.flac', '.wav')
            self._export_wav(signal, wav_path, sample_rate, 24)
        except Exception as e:
            print(f"Error exporting FLAC: {e}")
            print(f"Exporting as WAV instead.")
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
        try:
            from pydub import AudioSegment
            import tempfile
            import os
            
            # Ensure proper file extension
            if not file_path.lower().endswith('.mp3'):
                file_path += '.mp3'
            
            # Convert to 16-bit PCM first (required for pydub)
            audio_data = np.clip(signal * 32767, -32768, 32767).astype(np.int16)
            
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
                
                with wave.open(temp_wav_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data.tobytes())
            
            # Convert to MP3 using pydub
            audio = AudioSegment.from_wav(temp_wav_path)
            
            # Set bitrate based on quality
            bitrate_map = {
                'low': '128k',
                'medium': '192k',
                'high': '320k'
            }
            bitrate = bitrate_map.get(quality, '192k')
            
            audio.export(file_path, format='mp3', bitrate=bitrate)
            
            # Clean up temporary file
            os.unlink(temp_wav_path)
            
        except ImportError:
            print(f"Note: MP3 export requires pydub and ffmpeg. Install with: pip install pydub")
            print(f"Exporting as WAV instead.")
            wav_path = file_path.replace('.mp3', '.wav')
            self._export_wav(signal, wav_path, sample_rate, 16)
        except Exception as e:
            print(f"Error exporting MP3: {e}")
            print(f"Exporting as WAV instead.")
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
        try:
            from pydub import AudioSegment
            import tempfile
            import os
            
            # Ensure proper file extension
            if not file_path.lower().endswith('.ogg'):
                file_path += '.ogg'
            
            # Convert to 16-bit PCM first
            audio_data = np.clip(signal * 32767, -32768, 32767).astype(np.int16)
            
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
                
                with wave.open(temp_wav_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data.tobytes())
            
            # Convert to OGG using pydub
            audio = AudioSegment.from_wav(temp_wav_path)
            
            # Quality parameter for OGG (0-10, higher is better)
            quality_map = {
                'low': '3',
                'medium': '6',
                'high': '9'
            }
            ogg_quality = quality_map.get(quality, '6')
            
            audio.export(file_path, format='ogg', parameters=['-q:a', ogg_quality])
            
            # Clean up temporary file
            os.unlink(temp_wav_path)
            
        except ImportError:
            print(f"Note: OGG export requires pydub and ffmpeg. Install with: pip install pydub")
            print(f"Exporting as WAV instead.")
            wav_path = file_path.replace('.ogg', '.wav')
            self._export_wav(signal, wav_path, sample_rate, 16)
        except Exception as e:
            print(f"Error exporting OGG: {e}")
            print(f"Exporting as WAV instead.")
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
            file_path: Output file path (relative paths use ~/Music directory)
            sample_rate: Sample rate in Hz
            quality: Quality setting
            bit_depth: Bit depth for WAV export
        """
        # Resolve path (use ~/Music for relative paths)
        file_path = self._resolve_path(file_path)
        
        # Create parent directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
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
        
        print(f"Exported to: {file_path}")
    
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
