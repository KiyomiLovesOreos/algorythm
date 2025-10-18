"""
algorythm.structure - Arranges and composes the final track

This module provides high-level composition tools for arranging tracks,
applying effects, and creating complete musical pieces.
"""

from typing import List, Optional, Dict, Any, Literal
import numpy as np
from algorythm.synth import Synth
from algorythm.sequence import Motif


class Reverb:
    """
    Reverb effect for adding spatial depth to audio.
    """
    
    def __init__(
        self,
        mix: float = 0.3,
        room_size: float = 0.5,
        damping: float = 0.5
    ):
        """
        Initialize a reverb effect.
        
        Args:
            mix: Wet/dry mix (0.0 = dry, 1.0 = wet)
            room_size: Room size parameter (0.0 to 1.0)
            damping: High frequency damping (0.0 to 1.0)
        """
        self.mix = mix
        self.room_size = room_size
        self.damping = damping
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply reverb effect to audio signal.
        
        Args:
            signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal
        """
        # Simplified reverb implementation
        # In a real implementation, this would use convolution or feedback delay network
        wet = signal.copy()
        dry = signal.copy()
        
        # Simple delay-based reverb approximation
        delay_samples = int(0.05 * sample_rate * self.room_size)
        if delay_samples > 0 and len(wet) > delay_samples:
            wet[delay_samples:] += wet[:-delay_samples] * 0.3 * (1 - self.damping)
        
        return dry * (1 - self.mix) + wet * self.mix


class Delay:
    """
    Delay effect for creating echoes.
    """
    
    def __init__(
        self,
        delay_time: float = 0.5,
        feedback: float = 0.3,
        mix: float = 0.3
    ):
        """
        Initialize a delay effect.
        
        Args:
            delay_time: Delay time in seconds
            feedback: Feedback amount (0.0 to 1.0)
            mix: Wet/dry mix (0.0 = dry, 1.0 = wet)
        """
        self.delay_time = delay_time
        self.feedback = feedback
        self.mix = mix
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply delay effect to audio signal.
        
        Args:
            signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal
        """
        delay_samples = int(self.delay_time * sample_rate)
        output = signal.copy()
        
        if delay_samples > 0 and len(output) > delay_samples:
            delayed = np.zeros_like(output)
            delayed[delay_samples:] = output[:-delay_samples]
            output = output * (1 - self.mix) + delayed * self.mix
        
        return output


class Chorus:
    """
    Chorus effect for creating a thicker, richer sound.
    """
    
    def __init__(
        self,
        rate: float = 0.5,
        depth: float = 0.3,
        mix: float = 0.5
    ):
        """
        Initialize a chorus effect.
        
        Args:
            rate: LFO rate in Hz
            depth: Modulation depth (0.0 to 1.0)
            mix: Wet/dry mix (0.0 = dry, 1.0 = wet)
        """
        self.rate = rate
        self.depth = depth
        self.mix = mix
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply chorus effect to audio signal.
        
        Args:
            signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal
        """
        # Simplified chorus implementation
        return signal


class Flanger:
    """
    Flanger effect for creating a sweeping, jet-like sound.
    """
    
    def __init__(
        self,
        rate: float = 0.5,
        depth: float = 0.5,
        feedback: float = 0.3,
        mix: float = 0.5
    ):
        """
        Initialize a flanger effect.
        
        Args:
            rate: LFO rate in Hz
            depth: Modulation depth (0.0 to 1.0)
            feedback: Feedback amount (0.0 to 1.0)
            mix: Wet/dry mix (0.0 = dry, 1.0 = wet)
        """
        self.rate = rate
        self.depth = depth
        self.feedback = feedback
        self.mix = mix
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply flanger effect to audio signal.
        
        Args:
            signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal
        """
        # Simplified flanger implementation using time-varying delay
        output = signal.copy()
        
        # Create LFO for delay modulation
        duration = len(signal) / sample_rate
        t = np.linspace(0, duration, len(signal))
        lfo = np.sin(2 * np.pi * self.rate * t)
        
        # Calculate varying delay (1-10ms range)
        max_delay_samples = int(0.01 * sample_rate)
        min_delay_samples = int(0.001 * sample_rate)
        delay_range = max_delay_samples - min_delay_samples
        
        # Apply modulated delay
        wet = np.zeros_like(signal)
        for i in range(len(signal)):
            delay_samples = int(min_delay_samples + (lfo[i] * 0.5 + 0.5) * delay_range * self.depth)
            if i >= delay_samples:
                wet[i] = signal[i - delay_samples]
        
        # Add feedback
        wet = wet + wet * self.feedback * 0.3
        
        # Mix wet and dry
        return output * (1 - self.mix) + wet * self.mix


class Distortion:
    """
    Distortion effect for adding harmonic content and saturation.
    """
    
    def __init__(
        self,
        drive: float = 0.5,
        tone: float = 0.5,
        mix: float = 1.0
    ):
        """
        Initialize a distortion effect.
        
        Args:
            drive: Distortion amount (0.0 to 1.0)
            tone: Tone control for high-frequency rolloff (0.0 to 1.0)
            mix: Wet/dry mix (0.0 = dry, 1.0 = wet)
        """
        self.drive = drive
        self.tone = tone
        self.mix = mix
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply distortion effect to audio signal.
        
        Args:
            signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal
        """
        dry = signal.copy()
        
        # Apply gain based on drive
        gain = 1.0 + self.drive * 20.0
        wet = signal * gain
        
        # Soft clipping using tanh
        wet = np.tanh(wet)
        
        # Simple tone control (low-pass filtering)
        if self.tone < 1.0:
            # Simple moving average for low-pass effect
            window_size = int((1.0 - self.tone) * 10) + 1
            if window_size > 1:
                kernel = np.ones(window_size) / window_size
                wet = np.convolve(wet, kernel, mode='same')
        
        # Mix wet and dry
        return dry * (1 - self.mix) + wet * self.mix


class Compression:
    """
    Compression effect for dynamic range control.
    """
    
    def __init__(
        self,
        threshold: float = -20.0,
        ratio: float = 4.0,
        attack: float = 0.005,
        release: float = 0.1,
        makeup_gain: float = 1.0
    ):
        """
        Initialize a compression effect.
        
        Args:
            threshold: Threshold in dB
            ratio: Compression ratio (e.g., 4.0 means 4:1)
            attack: Attack time in seconds
            release: Release time in seconds
            makeup_gain: Makeup gain multiplier
        """
        self.threshold = threshold
        self.ratio = ratio
        self.attack = attack
        self.release = release
        self.makeup_gain = makeup_gain
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply compression effect to audio signal.
        
        Args:
            signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal
        """
        # Simplified compression implementation
        output = signal.copy()
        
        # Convert threshold from dB to linear
        threshold_linear = 10 ** (self.threshold / 20.0)
        
        # Calculate attack and release coefficients
        attack_coef = np.exp(-1.0 / (self.attack * sample_rate))
        release_coef = np.exp(-1.0 / (self.release * sample_rate))
        
        # Simple envelope follower and gain reduction
        envelope = 0.0
        for i in range(len(output)):
            input_level = abs(output[i])
            
            # Envelope follower
            if input_level > envelope:
                envelope = attack_coef * envelope + (1 - attack_coef) * input_level
            else:
                envelope = release_coef * envelope + (1 - release_coef) * input_level
            
            # Calculate gain reduction
            if envelope > threshold_linear:
                # Calculate compression
                gain_reduction = threshold_linear + (envelope - threshold_linear) / self.ratio
                output[i] = output[i] * (gain_reduction / max(envelope, 1e-10))
        
        # Apply makeup gain
        output = output * self.makeup_gain
        
        return output


class EffectChain:
    """
    Chain of audio effects to be applied in sequence.
    """
    
    def __init__(self):
        """Initialize an empty effect chain."""
        self.effects: List[Any] = []
    
    def add_effect(self, effect: Any) -> 'EffectChain':
        """
        Add an effect to the chain.
        
        Args:
            effect: Effect object (Reverb, Delay, etc.)
            
        Returns:
            Self for method chaining
        """
        self.effects.append(effect)
        return self
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply all effects in the chain to audio signal.
        
        Args:
            signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal
        """
        output = signal.copy()
        for effect in self.effects:
            output = effect.apply(output, sample_rate)
        return output


class SpatialAudio:
    """
    Spatial audio mixer for 3D positioning of sound sources.
    
    Programmatically controls the 3D position (X/Y/Z) of any sound source,
    automating panning and volume based on virtual distance.
    """
    
    def __init__(
        self,
        position: tuple = (0.0, 0.0, 0.0),
        listener_position: tuple = (0.0, 0.0, 0.0),
        distance_model: Literal['linear', 'inverse', 'exponential'] = 'inverse'
    ):
        """
        Initialize spatial audio positioning.
        
        Args:
            position: 3D position of sound source (x, y, z)
            listener_position: 3D position of listener
            distance_model: Distance attenuation model
        """
        self.position = np.array(position)
        self.listener_position = np.array(listener_position)
        self.distance_model = distance_model
        self.max_distance = 10.0  # Maximum audible distance
        self.reference_distance = 1.0  # Reference distance for attenuation
    
    def set_position(self, x: float, y: float, z: float) -> None:
        """
        Set the 3D position of the sound source.
        
        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
        """
        self.position = np.array([x, y, z])
    
    def set_listener_position(self, x: float, y: float, z: float) -> None:
        """
        Set the 3D position of the listener.
        
        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
        """
        self.listener_position = np.array([x, y, z])
    
    def calculate_distance(self) -> float:
        """
        Calculate distance between source and listener.
        
        Returns:
            Distance in 3D space
        """
        return np.linalg.norm(self.position - self.listener_position)
    
    def calculate_pan(self) -> float:
        """
        Calculate stereo pan position based on X position.
        
        Returns:
            Pan value (-1.0 = left, 0.0 = center, 1.0 = right)
        """
        # Calculate relative X position
        relative_x = self.position[0] - self.listener_position[0]
        
        # Normalize to pan range
        pan = np.clip(relative_x / self.max_distance, -1.0, 1.0)
        
        return pan
    
    def calculate_attenuation(self) -> float:
        """
        Calculate volume attenuation based on distance.
        
        Returns:
            Attenuation factor (0.0 to 1.0)
        """
        distance = self.calculate_distance()
        
        if distance <= self.reference_distance:
            return 1.0
        
        if self.distance_model == 'linear':
            # Linear distance model
            attenuation = 1.0 - (distance - self.reference_distance) / (self.max_distance - self.reference_distance)
        elif self.distance_model == 'inverse':
            # Inverse distance model
            attenuation = self.reference_distance / distance
        elif self.distance_model == 'exponential':
            # Exponential distance model
            attenuation = (self.reference_distance / distance) ** 2
        else:
            attenuation = 1.0
        
        return np.clip(attenuation, 0.0, 1.0)
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply spatial audio processing to signal.
        
        Args:
            signal: Mono input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Stereo audio signal with spatial positioning
        """
        # Calculate pan and attenuation
        pan = self.calculate_pan()
        attenuation = self.calculate_attenuation()
        
        # Apply attenuation
        processed = signal * attenuation
        
        # Create stereo output
        left_gain = np.sqrt((1.0 - pan) / 2.0)
        right_gain = np.sqrt((1.0 + pan) / 2.0)
        
        # Create stereo signal (2, N) shape
        stereo = np.vstack([processed * left_gain, processed * right_gain])
        
        return stereo


class Track:
    """
    A single track in a composition.
    
    Contains a synth instrument and a sequence of notes/motifs to play.
    """
    
    def __init__(self, name: str, synth: Synth):
        """
        Initialize a track.
        
        Args:
            name: Track name
            synth: Synthesizer instrument for this track
        """
        self.name = name
        self.synth = synth
        self.notes: List[Dict[str, Any]] = []
        self.effect_chain = EffectChain()
        self.volume = 1.0
        self.pan = 0.0  # -1.0 (left) to 1.0 (right)
    
    def add_note(
        self,
        frequency: float,
        start_time: float,
        duration: float
    ) -> 'Track':
        """
        Add a single note to the track.
        
        Args:
            frequency: Note frequency in Hz
            start_time: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            Self for method chaining
        """
        self.notes.append({
            'frequency': frequency,
            'start_time': start_time,
            'duration': duration
        })
        return self
    
    def add_motif(
        self,
        motif: Motif,
        start_time: float,
        tempo: float = 120.0
    ) -> 'Track':
        """
        Add a motif to the track.
        
        Args:
            motif: Motif to add
            start_time: Start time in seconds
            tempo: Tempo in BPM
            
        Returns:
            Self for method chaining
        """
        beat_duration = 60.0 / tempo
        current_time = start_time
        frequencies = motif.get_frequencies()
        
        for freq, duration in zip(frequencies, motif.durations):
            self.add_note(freq, current_time, duration * beat_duration)
            current_time += duration * beat_duration
        
        return self
    
    def add_fx(self, effect: Any) -> 'Track':
        """
        Add an effect to the track.
        
        Args:
            effect: Effect object
            
        Returns:
            Self for method chaining
        """
        self.effect_chain.add_effect(effect)
        return self
    
    def render(self, sample_rate: int = 44100) -> np.ndarray:
        """
        Render the track to audio.
        
        Args:
            sample_rate: Sample rate in Hz
            
        Returns:
            Rendered audio signal
        """
        if not self.notes:
            return np.array([])
        
        # Calculate total duration
        max_time = max(note['start_time'] + note['duration'] for note in self.notes)
        total_samples = int(max_time * sample_rate)
        output = np.zeros(total_samples)
        
        # Render each note
        for note in self.notes:
            note_signal = self.synth.generate_note(
                note['frequency'],
                note['duration'],
                sample_rate
            )
            start_sample = int(note['start_time'] * sample_rate)
            end_sample = start_sample + len(note_signal)
            
            if end_sample <= total_samples:
                output[start_sample:end_sample] += note_signal
            else:
                output[start_sample:] += note_signal[:total_samples - start_sample]
        
        # Apply effects
        output = self.effect_chain.apply(output, sample_rate)
        
        # Apply volume
        output = output * self.volume
        
        return output


class Composition:
    """
    Main composition class that combines multiple tracks.
    
    This is the top-level container for a musical piece.
    """
    
    def __init__(self, tempo: float = 120.0, sample_rate: int = 44100):
        """
        Initialize a composition.
        
        Args:
            tempo: Tempo in beats per minute
            sample_rate: Sample rate in Hz
        """
        self.tempo = tempo
        self.sample_rate = sample_rate
        self.tracks: Dict[str, Track] = {}
        self.current_track: Optional[Track] = None
    
    def add_track(self, name: str, synth: Synth) -> 'Composition':
        """
        Add a new track to the composition.
        
        Args:
            name: Track name
            synth: Synthesizer for the track
            
        Returns:
            Self for method chaining
        """
        track = Track(name, synth)
        self.tracks[name] = track
        self.current_track = track
        return self
    
    def repeat_motif(
        self,
        motif: Motif,
        bars: int = 1,
        start_bar: float = 0
    ) -> 'Composition':
        """
        Repeat a motif for a specified number of bars.
        
        Args:
            motif: Motif to repeat
            bars: Number of bars to repeat
            start_bar: Starting bar number
            
        Returns:
            Self for method chaining
        """
        if not self.current_track:
            raise ValueError("No track selected. Call add_track first.")
        
        beats_per_bar = 4  # Assuming 4/4 time
        bar_duration = (beats_per_bar * 60.0) / self.tempo
        
        for bar in range(bars):
            start_time = (start_bar + bar) * bar_duration
            self.current_track.add_motif(motif, start_time, self.tempo)
        
        return self
    
    def transpose(self, semitones: int) -> 'Composition':
        """
        Transpose the current track by semitones.
        
        Args:
            semitones: Number of semitones to transpose
            
        Returns:
            Self for method chaining
        """
        if not self.current_track:
            raise ValueError("No track selected. Call add_track first.")
        
        # Transpose all notes in the current track
        for note in self.current_track.notes:
            # Frequency ratio for semitone transposition
            ratio = 2.0 ** (semitones / 12.0)
            note['frequency'] *= ratio
        
        return self
    
    def add_fx(self, effect: Any) -> 'Composition':
        """
        Add an effect to the current track.
        
        Args:
            effect: Effect object
            
        Returns:
            Self for method chaining
        """
        if not self.current_track:
            raise ValueError("No track selected. Call add_track first.")
        
        self.current_track.add_fx(effect)
        return self
    
    def render(
        self,
        file_path: Optional[str] = None,
        quality: str = 'high',
        formats: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Render the composition to audio.
        
        Args:
            file_path: Output file path (without extension if multiple formats)
            quality: Quality setting ('low', 'medium', 'high')
            formats: List of output formats (e.g., ['wav', 'mp3', 'flac'])
            
        Returns:
            Rendered audio signal
        """
        if not self.tracks:
            return np.array([])
        
        # Render all tracks
        track_signals = [track.render(self.sample_rate) for track in self.tracks.values()]
        
        # Find maximum length
        max_length = max(len(signal) for signal in track_signals if len(signal) > 0)
        
        # Mix tracks
        output = np.zeros(max_length)
        for signal in track_signals:
            if len(signal) > 0:
                output[:len(signal)] += signal
        
        # Normalize to prevent clipping
        max_amplitude = np.max(np.abs(output))
        if max_amplitude > 0:
            output = output / max_amplitude * 0.9
        
        # Export if file path provided
        if file_path:
            from algorythm.export import Exporter
            exporter = Exporter()
            
            if formats:
                for fmt in formats:
                    # Remove extension from file_path if present
                    base_path = file_path.rsplit('.', 1)[0] if '.' in file_path else file_path
                    output_path = f"{base_path}.{fmt}"
                    exporter.export(output, output_path, self.sample_rate, quality)
            else:
                exporter.export(output, file_path, self.sample_rate, quality)
        
        return output
