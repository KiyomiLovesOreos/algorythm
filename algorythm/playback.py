"""
algorythm.playback - Real-time audio playback and streaming

This module provides real-time audio playback capabilities for interactive
composition and live coding scenarios.
"""

import numpy as np
from typing import Optional, Callable
import threading
import queue
import time


class AudioPlayer:
    """
    Real-time audio player using system audio output.
    
    Provides playback of pre-rendered audio or streaming audio generation.
    """
    
    def __init__(self, sample_rate: int = 44100, buffer_size: int = 1024):
        """
        Initialize audio player.
        
        Args:
            sample_rate: Sample rate in Hz
            buffer_size: Audio buffer size in samples
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.is_playing = False
        self.audio_queue = queue.Queue(maxsize=10)
        self.playback_thread = None
        self._pyaudio = None
        self._stream = None
        self.volume = 1.0
        
        try:
            import pyaudio
            self._pyaudio = pyaudio.PyAudio()
        except ImportError:
            print("Warning: pyaudio not installed. Install with: pip install pyaudio")
            self._pyaudio = None
    
    def set_volume(self, volume: float):
        """
        Set playback volume.
        
        Args:
            volume: Volume level (0.0 to 1.0, or higher for amplification)
        """
        self.volume = max(0.0, volume)
    
    def play(self, audio: np.ndarray, blocking: bool = False):
        """
        Play audio buffer.
        
        Args:
            audio: Audio samples to play
            blocking: If True, wait for playback to complete
        """
        if self._pyaudio is None:
            print("Cannot play audio: pyaudio not available")
            return
        
        # Apply volume
        audio = audio * self.volume
        
        # Normalize audio to prevent clipping
        audio = np.clip(audio, -1.0, 1.0)
        
        if blocking:
            self._play_blocking(audio)
        else:
            self._play_async(audio)
    
    def _play_blocking(self, audio: np.ndarray):
        """Play audio synchronously."""
        if self._stream is None or not self._stream.is_active():
            self._stream = self._pyaudio.open(
                format=self._pyaudio.get_format_from_width(2),  # 16-bit
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.buffer_size
            )
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Write audio in chunks
        for i in range(0, len(audio_int16), self.buffer_size):
            chunk = audio_int16[i:i + self.buffer_size]
            self._stream.write(chunk.tobytes())
    
    def _play_async(self, audio: np.ndarray):
        """Play audio asynchronously in a separate thread."""
        if self.is_playing:
            self.stop()
        
        self.is_playing = True
        self.playback_thread = threading.Thread(
            target=self._playback_worker,
            args=(audio,)
        )
        self.playback_thread.start()
    
    def _playback_worker(self, audio: np.ndarray):
        """Worker thread for audio playback."""
        self._play_blocking(audio)
        self.is_playing = False
    
    def stop(self):
        """Stop current playback."""
        self.is_playing = False
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)
        
        if self._stream:
            self._stream.stop_stream()
    
    def close(self):
        """Clean up audio resources."""
        self.stop()
        if self._stream:
            self._stream.close()
        if self._pyaudio:
            self._pyaudio.terminate()
    
    def __del__(self):
        """Ensure resources are cleaned up."""
        try:
            self.close()
        except:
            pass


class StreamingPlayer:
    """
    Streaming audio player for real-time generation.
    
    Generates audio on-the-fly using a callback function.
    """
    
    def __init__(
        self,
        generator_callback: Callable[[int], np.ndarray],
        sample_rate: int = 44100,
        buffer_size: int = 1024
    ):
        """
        Initialize streaming player.
        
        Args:
            generator_callback: Function that generates audio chunks
            sample_rate: Sample rate in Hz
            buffer_size: Audio buffer size in samples
        """
        self.generator_callback = generator_callback
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.is_streaming = False
        self.stream_thread = None
        self._pyaudio = None
        self._stream = None
        self.volume = 1.0
        
        try:
            import pyaudio
            self._pyaudio = pyaudio.PyAudio()
        except ImportError:
            print("Warning: pyaudio not installed. Install with: pip install pyaudio")
            self._pyaudio = None
    
    def set_volume(self, volume: float):
        """
        Set playback volume.
        
        Args:
            volume: Volume level (0.0 to 1.0, or higher for amplification)
        """
        self.volume = max(0.0, volume)
    
    def start(self):
        """Start streaming audio."""
        if self._pyaudio is None:
            print("Cannot stream audio: pyaudio not available")
            return
        
        if self.is_streaming:
            return
        
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._streaming_worker)
        self.stream_thread.start()
    
    def _streaming_worker(self):
        """Worker thread for streaming audio."""
        self._stream = self._pyaudio.open(
            format=self._pyaudio.get_format_from_width(2),  # 16-bit
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.buffer_size,
            stream_callback=self._audio_callback
        )
        
        self._stream.start_stream()
        
        while self.is_streaming and self._stream.is_active():
            time.sleep(0.1)
        
        self._stream.stop_stream()
        self._stream.close()
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio streaming."""
        try:
            # Generate audio chunk
            audio = self.generator_callback(frame_count)
            
            # Apply volume
            audio = audio * self.volume
            
            # Normalize and convert to 16-bit PCM
            audio = np.clip(audio, -1.0, 1.0)
            audio_int16 = (audio * 32767).astype(np.int16)
            
            import pyaudio
            return (audio_int16.tobytes(), pyaudio.paContinue)
        except Exception as e:
            print(f"Error in audio callback: {e}")
            import pyaudio
            return (bytes(frame_count * 2), pyaudio.paComplete)
    
    def stop(self):
        """Stop streaming audio."""
        self.is_streaming = False
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=1.0)
    
    def close(self):
        """Clean up audio resources."""
        self.stop()
        if self._pyaudio:
            self._pyaudio.terminate()
    
    def __del__(self):
        """Ensure resources are cleaned up."""
        try:
            self.close()
        except:
            pass


class LiveCompositionPlayer:
    """
    Live composition player for real-time updates.
    
    Allows updating the composition while it's playing.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize live composition player.
        
        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.player = AudioPlayer(sample_rate=sample_rate)
        self.current_audio = None
        self.is_looping = False
        self.loop_thread = None
    
    def play(self, audio: np.ndarray, loop: bool = False):
        """
        Play audio with optional looping.
        
        Args:
            audio: Audio samples to play
            loop: If True, loop the audio continuously
        """
        self.current_audio = audio
        self.is_looping = loop
        
        if loop:
            self.loop_thread = threading.Thread(target=self._loop_worker)
            self.loop_thread.start()
        else:
            self.player.play(audio, blocking=False)
    
    def _loop_worker(self):
        """Worker thread for looping audio."""
        while self.is_looping and self.current_audio is not None:
            self.player.play(self.current_audio, blocking=True)
    
    def update_audio(self, audio: np.ndarray):
        """
        Update the audio buffer while playing.
        
        Args:
            audio: New audio samples
        """
        self.current_audio = audio
    
    def stop(self):
        """Stop playback."""
        self.is_looping = False
        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_thread.join(timeout=1.0)
        self.player.stop()
    
    def close(self):
        """Clean up resources."""
        self.stop()
        self.player.close()
