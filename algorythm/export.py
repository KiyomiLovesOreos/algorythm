"""
algorythm.export - Renders and saves the final audio

This module provides tools for exporting compositions to various audio and video formats.
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
    
    Supports WAV, FLAC, MP3, OGG, and MP4 (video with visualization) formats.
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
        bit_depth: int = 16,
        visualizer: Optional[object] = None,
        video_width: int = 1920,
        video_height: int = 1080,
        video_fps: int = 30
    ) -> None:
        """
        Export audio signal to file.
        
        Args:
            signal: Audio signal to export
            file_path: Output file path (relative paths use ~/Music directory)
            sample_rate: Sample rate in Hz
            quality: Quality setting
            bit_depth: Bit depth for WAV export
            visualizer: Optional visualizer for MP4 export (from algorythm.visualization)
            video_width: Video width in pixels (for MP4)
            video_height: Video height in pixels (for MP4)
            video_fps: Video frames per second (for MP4)
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
        elif file_path_lower.endswith('.mp4'):
            self._export_mp4(signal, file_path, sample_rate, visualizer, 
                           video_width, video_height, video_fps)
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
    
    def _export_mp4(
        self,
        signal: np.ndarray,
        file_path: str,
        sample_rate: int,
        visualizer: Optional[object],
        video_width: int,
        video_height: int,
        video_fps: int
    ) -> None:
        """
        Export audio signal to MP4 video file with visualization.
        Tries multiple backends in order: VideoRenderer (OpenCV/matplotlib) > PIL fallback.
        
        Args:
            signal: Audio signal
            file_path: Output file path
            sample_rate: Sample rate in Hz
            visualizer: Visualizer instance from algorythm.visualization
            video_width: Video width in pixels
            video_height: Video height in pixels
            video_fps: Video frames per second
        """
        from algorythm.visualization import VideoRenderer, WaveformVisualizer, HAS_OPENCV, HAS_MATPLOTLIB
        
        # Ensure proper file extension
        if not file_path.lower().endswith('.mp4'):
            file_path += '.mp4'
        
        # If no visualizer provided, use default waveform visualizer
        if visualizer is None:
            visualizer = WaveformVisualizer(sample_rate=sample_rate)
            print("No visualizer provided, using WaveformVisualizer by default")
        
        print(f"Rendering video to {file_path}...")
        
        # Try using VideoRenderer if backends are available
        if HAS_OPENCV or HAS_MATPLOTLIB:
            try:
                renderer = VideoRenderer(
                    width=video_width,
                    height=video_height,
                    fps=video_fps,
                    sample_rate=sample_rate,
                    debug=False
                )
                
                # Render frames and save video
                renderer.render_frames(
                    signal,
                    visualizer,
                    output_path=file_path
                )
                
                print(f"✓ Video exported successfully using VideoRenderer!")
                return
                
            except Exception as e:
                print(f"VideoRenderer failed: {e}")
                print("Trying PIL fallback...")
        
        # PIL fallback
        try:
            self._export_mp4_with_pil(
                signal, file_path, sample_rate, visualizer,
                video_width, video_height, video_fps
            )
        except Exception as e:
            print(f"❌ Error exporting MP4: {e}")
            import traceback
            traceback.print_exc()
            print(f"Exporting as WAV instead.")
            wav_path = file_path.replace('.mp4', '.wav')
            self._export_wav(signal, wav_path, sample_rate, 16)
    
    def _export_mp4_with_pil(
        self,
        signal: np.ndarray,
        file_path: str,
        sample_rate: int,
        visualizer: object,
        video_width: int,
        video_height: int,
        video_fps: int
    ) -> None:
        """
        Export MP4 using PIL/Pillow for frame generation and ffmpeg for encoding.
        This is a fallback when opencv-python and matplotlib are not available.
        """
        try:
            from PIL import Image
            import subprocess
            import tempfile
            import os
            
            print(f"Using PIL fallback for video rendering...")
            
            # Calculate frames
            samples_per_frame = sample_rate // video_fps
            num_frames = int(len(signal) / samples_per_frame)
            
            print(f"Rendering {num_frames} frames...")
            
            # Create temporary directory for frames
            temp_dir = tempfile.mkdtemp()
            frame_paths = []
            
            # Render each frame
            for frame_idx in range(num_frames):
                start = frame_idx * samples_per_frame
                end = min(start + samples_per_frame, len(signal))
                chunk = signal[start:end]
                
                # Generate visualization
                frame_data = self._generate_vis_frame_pil(
                    chunk, visualizer, video_height, video_width
                )
                
                # Convert to PIL Image
                frame_img = Image.fromarray((frame_data * 255).astype(np.uint8))
                if frame_img.mode != 'RGB':
                    frame_img = frame_img.convert('RGB')
                
                # Save frame
                frame_path = os.path.join(temp_dir, f'frame_{frame_idx:06d}.png')
                frame_img.save(frame_path)
                frame_paths.append(frame_path)
                
                if (frame_idx + 1) % max(1, num_frames // 10) == 0:
                    print(f"  Progress: {100 * (frame_idx + 1) / num_frames:.1f}%")
            
            print("Creating video with ffmpeg...")
            
            # Create temp audio file
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()
            
            audio_data = np.clip(signal * 32767, -32768, 32767).astype(np.int16)
            with wave.open(temp_audio_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            # Use ffmpeg to create video from frames
            pattern = os.path.join(temp_dir, 'frame_%06d.png')
            subprocess.run([
                'ffmpeg', '-y',
                '-framerate', str(video_fps),
                '-i', pattern,
                '-i', temp_audio_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                '-pix_fmt', 'yuv420p',
                file_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Cleanup
            for frame_path in frame_paths:
                os.unlink(frame_path)
            os.rmdir(temp_dir)
            os.unlink(temp_audio_path)
            
            print(f"✓ Video exported successfully using PIL fallback!")
            
        except ImportError:
            print(f"❌ PIL (Pillow) not available. Install with: pip install Pillow")
            print(f"Or install opencv-python or matplotlib for video export.")
            raise
        except FileNotFoundError:
            print(f"❌ ffmpeg not found. Please install ffmpeg to create MP4 videos.")
            print(f"  Ubuntu/Debian: sudo apt install ffmpeg")
            print(f"  macOS: brew install ffmpeg")
            raise
        except subprocess.CalledProcessError as e:
            print(f"❌ ffmpeg error: {e}")
            raise
    
    def _generate_vis_frame_pil(
        self,
        chunk: np.ndarray,
        visualizer: object,
        height: int,
        width: int
    ) -> np.ndarray:
        """
        Generate visualization frame for PIL export.
        Returns RGB array.
        """
        from algorythm.visualization import (
            WaveformVisualizer, CircularVisualizer,
            SpectrogramVisualizer, FrequencyScopeVisualizer
        )
        
        # Get grayscale visualization
        if hasattr(visualizer, 'to_image_data'):
            gray_data = visualizer.to_image_data(chunk, height, width)
        elif isinstance(visualizer, SpectrogramVisualizer):
            spec = visualizer.generate(chunk)
            if spec.size > 0:
                # Resize spectrogram
                h, w = spec.shape
                if h > 0 and w > 0:
                    row_idx = (np.arange(height) * h / height).astype(int)
                    col_idx = (np.arange(width) * w / width).astype(int)
                    gray_data = spec[row_idx][:, col_idx]
                    # Normalize
                    gray_data = (gray_data - gray_data.min()) / (gray_data.max() - gray_data.min() + 1e-10)
                else:
                    gray_data = np.zeros((height, width))
            else:
                gray_data = np.zeros((height, width))
        else:
            gray_data = np.zeros((height, width))
        
        # Convert grayscale to RGB (white on black)
        rgb_data = np.zeros((height, width, 3))
        rgb_data[:, :, 0] = gray_data
        rgb_data[:, :, 1] = gray_data
        rgb_data[:, :, 2] = gray_data
        
        return rgb_data
    
    def _generate_visualization_frame(
        self,
        chunk: np.ndarray,
        visualizer: object,
        height: int,
        width: int
    ) -> np.ndarray:
        """
        Generate a visualization frame from audio chunk.
        
        Args:
            chunk: Audio chunk for this frame
            visualizer: Visualizer instance
            height: Frame height
            width: Frame width
            
        Returns:
            Frame data as 2D array
        """
        # Import visualization types
        from algorythm.visualization import (
            WaveformVisualizer, SpectrogramVisualizer, 
            FrequencyScopeVisualizer, OscilloscopeVisualizer,
            CircularVisualizer, ParticleVisualizer
        )
        
        # Generate frame based on visualizer type
        if isinstance(visualizer, WaveformVisualizer):
            frame_data = visualizer.to_image_data(chunk, height, width)
        elif isinstance(visualizer, SpectrogramVisualizer):
            spec_data = visualizer.generate(chunk)
            frame_data = self._resize_visualization(spec_data, height, width)
        elif isinstance(visualizer, FrequencyScopeVisualizer):
            spectrum = visualizer.generate(chunk)
            frame_data = self._spectrum_to_frame(spectrum, height, width)
        elif isinstance(visualizer, (OscilloscopeVisualizer, CircularVisualizer, 
                                     ParticleVisualizer)):
            # These visualizers have their own to_image_data method
            if hasattr(visualizer, 'to_image_data'):
                frame_data = visualizer.to_image_data(chunk, height, width)
            else:
                # Fallback to basic waveform
                frame_data = np.zeros((height, width))
        else:
            # Unknown visualizer, use zeros
            frame_data = np.zeros((height, width))
        
        return frame_data
    
    def _resize_visualization(self, data: np.ndarray, height: int, width: int) -> np.ndarray:
        """Resize visualization data to fit frame dimensions."""
        if data.size == 0:
            return np.zeros((height, width))
        
        h, w = data.shape
        if h == 0 or w == 0:
            return np.zeros((height, width))
        
        row_indices = np.linspace(0, h - 1, height).astype(int)
        col_indices = np.linspace(0, w - 1, width).astype(int)
        
        resized = data[row_indices][:, col_indices]
        
        # Normalize to 0-1 range
        min_val = np.min(resized)
        max_val = np.max(resized)
        if max_val > min_val:
            resized = (resized - min_val) / (max_val - min_val)
        
        return resized
    
    def _spectrum_to_frame(self, spectrum: np.ndarray, height: int, width: int) -> np.ndarray:
        """Convert frequency spectrum to visualization frame."""
        frame = np.zeros((height, width))
        
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
                frame[-bar_height:, i] = 1.0
        
        return frame
    
    def _convert_frame_to_bgr(
        self,
        frame_data: np.ndarray,
        height: int,
        width: int,
        background_color: tuple = (0, 0, 0),
        foreground_color: tuple = (255, 255, 255)
    ) -> np.ndarray:
        """
        Convert normalized grayscale frame to BGR for OpenCV.
        
        Args:
            frame_data: Normalized frame data (0-1)
            height: Frame height
            width: Frame width
            background_color: RGB background color
            foreground_color: RGB foreground color
            
        Returns:
            BGR frame for OpenCV
        """
        import cv2
        
        # Create RGB image with background color
        frame_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        frame_rgb[:] = background_color
        
        # Apply foreground color where frame_data is non-zero
        for c in range(3):
            channel = frame_rgb[:, :, c].astype(float)
            channel += frame_data * (foreground_color[c] - background_color[c])
            frame_rgb[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        return frame_bgr
    
    def _add_audio_to_mp4(
        self,
        video_path: str,
        audio_signal: np.ndarray,
        output_path: str,
        sample_rate: int
    ) -> None:
        """
        Add audio track to MP4 video using ffmpeg.
        
        Args:
            video_path: Path to video file without audio
            audio_signal: Audio signal to add
            output_path: Final output path
            sample_rate: Audio sample rate
        """
        import subprocess
        import tempfile
        
        # Save audio to temporary WAV file
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        # Write WAV file
        audio_data = np.clip(audio_signal * 32767, -32768, 32767).astype(np.int16)
        with wave.open(temp_audio_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        # Use ffmpeg to combine video and audio
        try:
            print("Combining video and audio with ffmpeg...")
            subprocess.run([
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', temp_audio_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                output_path
            ], check=True, capture_output=True, text=True)
            print("Video and audio combined successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error running ffmpeg: {e}")
            print(f"stderr: {e.stderr}")
            print("Make sure ffmpeg is installed on your system.")
            raise
        except FileNotFoundError:
            print("ffmpeg not found. Please install ffmpeg to export MP4 with audio.")
            raise
        finally:
            # Clean up temporary audio file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
    
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
