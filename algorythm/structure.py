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


class EQ:
    """
    Multi-band equalizer effect for frequency shaping.
    """
    
    def __init__(
        self,
        low_gain: float = 1.0,
        mid_gain: float = 1.0,
        high_gain: float = 1.0,
        low_freq: float = 200.0,
        high_freq: float = 2000.0
    ):
        """
        Initialize a 3-band EQ effect.
        
        Args:
            low_gain: Low frequency gain multiplier (0.0 to 2.0)
            mid_gain: Mid frequency gain multiplier (0.0 to 2.0)
            high_gain: High frequency gain multiplier (0.0 to 2.0)
            low_freq: Low/mid crossover frequency in Hz
            high_freq: Mid/high crossover frequency in Hz
        """
        self.low_gain = low_gain
        self.mid_gain = mid_gain
        self.high_gain = high_gain
        self.low_freq = low_freq
        self.high_freq = high_freq
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply EQ effect to audio signal.
        
        Args:
            signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal
        """
        # Simple 3-band EQ using FFT
        spectrum = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1/sample_rate)
        
        # Apply gain to each band
        low_mask = freqs < self.low_freq
        mid_mask = (freqs >= self.low_freq) & (freqs < self.high_freq)
        high_mask = freqs >= self.high_freq
        
        spectrum[low_mask] *= self.low_gain
        spectrum[mid_mask] *= self.mid_gain
        spectrum[high_mask] *= self.high_gain
        
        # Convert back to time domain
        return np.fft.irfft(spectrum, n=len(signal))


class Phaser:
    """
    Phaser effect for creating sweeping notch filter effects.
    """
    
    def __init__(
        self,
        rate: float = 0.5,
        depth: float = 0.5,
        stages: int = 4,
        feedback: float = 0.5,
        mix: float = 0.5
    ):
        """
        Initialize a phaser effect.
        
        Args:
            rate: LFO rate in Hz
            depth: Modulation depth (0.0 to 1.0)
            stages: Number of all-pass filter stages (2-12)
            feedback: Feedback amount (0.0 to 1.0)
            mix: Wet/dry mix (0.0 = dry, 1.0 = wet)
        """
        self.rate = rate
        self.depth = depth
        self.stages = max(2, min(12, stages))
        self.feedback = feedback
        self.mix = mix
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply phaser effect to audio signal.
        
        Args:
            signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal
        """
        dry = signal.copy()
        
        # Create LFO for frequency modulation
        duration = len(signal) / sample_rate
        t = np.linspace(0, duration, len(signal))
        lfo = np.sin(2 * np.pi * self.rate * t)
        
        # Simple all-pass filter approximation
        wet = signal.copy()
        for stage in range(self.stages):
            # Vary delay based on LFO
            max_delay = int(0.005 * sample_rate)  # 5ms max delay
            delayed = np.zeros_like(wet)
            for i in range(len(wet)):
                delay_samples = int((lfo[i] * 0.5 + 0.5) * max_delay * self.depth)
                if i >= delay_samples:
                    delayed[i] = wet[i - delay_samples]
            
            # All-pass filter approximation
            wet = wet - delayed
        
        # Add feedback
        wet = wet + wet * self.feedback * 0.3
        
        # Mix wet and dry
        return dry * (1 - self.mix) + wet * self.mix


class Tremolo:
    """
    Tremolo effect for amplitude modulation.
    """
    
    def __init__(
        self,
        rate: float = 5.0,
        depth: float = 0.5,
        waveform: Literal['sine', 'square', 'triangle'] = 'sine'
    ):
        """
        Initialize a tremolo effect.
        
        Args:
            rate: Modulation rate in Hz
            depth: Modulation depth (0.0 to 1.0)
            waveform: LFO waveform type
        """
        self.rate = rate
        self.depth = depth
        self.waveform = waveform
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply tremolo effect to audio signal.
        
        Args:
            signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal
        """
        # Create LFO for amplitude modulation
        duration = len(signal) / sample_rate
        t = np.linspace(0, duration, len(signal))
        
        if self.waveform == 'sine':
            lfo = np.sin(2 * np.pi * self.rate * t)
        elif self.waveform == 'square':
            lfo = np.sign(np.sin(2 * np.pi * self.rate * t))
        elif self.waveform == 'triangle':
            lfo = 2 * np.abs(2 * (t * self.rate - np.floor(0.5 + t * self.rate))) - 1
        else:
            lfo = np.sin(2 * np.pi * self.rate * t)
        
        # Scale LFO to modulation range
        modulation = 1.0 - self.depth * (1.0 - (lfo * 0.5 + 0.5))
        
        return signal * modulation


class Bitcrusher:
    """
    Bitcrusher effect for digital distortion and lo-fi sounds.
    """
    
    def __init__(
        self,
        bit_depth: int = 8,
        sample_rate_reduction: float = 1.0,
        mix: float = 1.0
    ):
        """
        Initialize a bitcrusher effect.
        
        Args:
            bit_depth: Bit depth for quantization (1-16)
            sample_rate_reduction: Sample rate reduction factor (1.0 = no reduction)
            mix: Wet/dry mix (0.0 = dry, 1.0 = wet)
        """
        self.bit_depth = max(1, min(16, bit_depth))
        self.sample_rate_reduction = max(1.0, sample_rate_reduction)
        self.mix = mix
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply bitcrusher effect to audio signal.
        
        Args:
            signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal
        """
        dry = signal.copy()
        wet = signal.copy()
        
        # Bit depth reduction (quantization)
        levels = 2 ** self.bit_depth
        wet = np.round(wet * levels) / levels
        
        # Sample rate reduction (decimation)
        if self.sample_rate_reduction > 1.0:
            step = int(self.sample_rate_reduction)
            for i in range(len(wet)):
                if i % step != 0:
                    wet[i] = wet[i - (i % step)]
        
        # Mix wet and dry
        return dry * (1 - self.mix) + wet * self.mix


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
        self.master_volume = 1.0
    
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
    
    def set_track_volume(self, track_name: str, volume: float) -> 'Composition':
        """
        Set volume for a specific track.
        
        Args:
            track_name: Name of the track
            volume: Volume level (0.0 to 1.0, or higher for amplification)
            
        Returns:
            Self for method chaining
        """
        if track_name not in self.tracks:
            raise ValueError(f"Track '{track_name}' not found")
        
        self.tracks[track_name].volume = volume
        return self
    
    def set_master_volume(self, volume: float) -> 'Composition':
        """
        Set master volume for the entire composition.
        
        Args:
            volume: Volume level (0.0 to 1.0, or higher for amplification)
            
        Returns:
            Self for method chaining
        """
        self.master_volume = volume
        return self
    
    def fade_in(self, duration: float) -> 'Composition':
        """
        Apply a fade-in to the composition.
        
        Args:
            duration: Fade-in duration in seconds
            
        Returns:
            Self for method chaining
        """
        self._fade_in_duration = duration
        return self
    
    def fade_out(self, duration: float) -> 'Composition':
        """
        Apply a fade-out to the composition.
        
        Args:
            duration: Fade-out duration in seconds
            
        Returns:
            Self for method chaining
        """
        self._fade_out_duration = duration
        return self
    
    def render(
        self,
        file_path: Optional[str] = None,
        quality: str = 'high',
        formats: Optional[List[str]] = None,
        video: bool = False,
        video_config: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Render the composition to audio and optionally video.
        
        Args:
            file_path: Output file path (without extension if multiple formats)
            quality: Quality setting ('low', 'medium', 'high')
            formats: List of output formats (e.g., ['wav', 'mp3', 'flac'])
            video: If True, export as MP4 video with visualization
            video_config: Video configuration dictionary with keys:
                - visualizer: Visualizer type ('waveform', 'spectrum', 'circular', 'particle', 'spectrogram')
                - width: Video width (default: 1920)
                - height: Video height (default: 1080)
                - fps: Frames per second (default: 30)
                - background_color: RGB tuple (default: (0, 0, 0))
                - foreground_color: RGB tuple (default: (255, 255, 255))
                - Additional visualizer-specific options
            
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
        
        # Apply master volume
        output = output * self.master_volume
        
        # Apply fade in/out if specified
        if hasattr(self, '_fade_in_duration') and self._fade_in_duration > 0:
            fade_in_samples = int(self._fade_in_duration * self.sample_rate)
            fade_in_samples = min(fade_in_samples, len(output))
            fade_curve = np.linspace(0, 1, fade_in_samples)
            output[:fade_in_samples] *= fade_curve
        
        if hasattr(self, '_fade_out_duration') and self._fade_out_duration > 0:
            fade_out_samples = int(self._fade_out_duration * self.sample_rate)
            fade_out_samples = min(fade_out_samples, len(output))
            fade_curve = np.linspace(1, 0, fade_out_samples)
            output[-fade_out_samples:] *= fade_curve
        
        # Normalize to prevent clipping
        max_amplitude = np.max(np.abs(output))
        if max_amplitude > 0:
            output = output / max_amplitude * 0.9
        
        # Export if file path provided
        if file_path:
            if video:
                # Export as video
                self._export_video(output, file_path, video_config or {})
            else:
                # Export as audio
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
    
    def _export_video(self, audio: np.ndarray, file_path: str, config: Dict[str, Any]) -> None:
        """
        Export composition as video with visualization.
        
        Args:
            audio: Rendered audio signal
            file_path: Output video file path
            config: Video configuration dictionary
        """
        from algorythm.visualization import (
            VideoRenderer, WaveformVisualizer, FrequencyScopeVisualizer,
            SpectrogramVisualizer, CircularVisualizer, ParticleVisualizer
        )
        
        # Extract configuration
        visualizer_type = config.get('visualizer', 'spectrum')
        width = config.get('width', 1920)
        height = config.get('height', 1080)
        fps = config.get('fps', 30)
        bg_color = config.get('background_color', (0, 0, 0))
        fg_color = config.get('foreground_color', (255, 255, 255))
        
        # Create visualizer based on type
        if visualizer_type == 'waveform':
            visualizer = WaveformVisualizer(
                sample_rate=self.sample_rate,
                window_size=config.get('window_size', 1024)
            )
        elif visualizer_type == 'spectrum':
            visualizer = FrequencyScopeVisualizer(
                sample_rate=self.sample_rate,
                fft_size=config.get('fft_size', 2048),
                freq_range=config.get('freq_range', (20.0, 20000.0))
            )
        elif visualizer_type == 'spectrogram':
            visualizer = SpectrogramVisualizer(
                sample_rate=self.sample_rate,
                window_size=config.get('window_size', 2048),
                hop_size=config.get('hop_size', 512)
            )
        elif visualizer_type == 'circular':
            visualizer = CircularVisualizer(
                sample_rate=self.sample_rate,
                num_bars=config.get('num_bars', 64),
                inner_radius=config.get('inner_radius', 0.2),
                bar_width=config.get('bar_width', 0.8),
                smoothing=config.get('smoothing', 0.5)
            )
        elif visualizer_type == 'particle':
            visualizer = ParticleVisualizer(
                sample_rate=self.sample_rate,
                num_particles=config.get('num_particles', 100),
                decay=config.get('decay', 0.95),
                sensitivity=config.get('sensitivity', 1.0)
            )
        else:
            # Default to spectrum
            visualizer = FrequencyScopeVisualizer(sample_rate=self.sample_rate)
        
        # Create video renderer
        renderer = VideoRenderer(
            width=width,
            height=height,
            fps=fps,
            sample_rate=self.sample_rate,
            background_color=bg_color,
            foreground_color=fg_color
        )
        
        # Render video
        print(f"Rendering video with {visualizer_type} visualizer...")
        renderer.render_frames(audio, visualizer, output_path=file_path)


class VolumeControl:
    """
    Utility class for volume control and conversions.
    
    Provides methods for converting between different volume representations
    and applying volume changes to audio signals.
    """
    
    @staticmethod
    def db_to_linear(db: float) -> float:
        """
        Convert decibels to linear amplitude.
        
        Args:
            db: Volume in decibels
            
        Returns:
            Linear amplitude multiplier
        """
        return 10.0 ** (db / 20.0)
    
    @staticmethod
    def linear_to_db(linear: float) -> float:
        """
        Convert linear amplitude to decibels.
        
        Args:
            linear: Linear amplitude multiplier
            
        Returns:
            Volume in decibels
        """
        if linear <= 0:
            return -np.inf
        return 20.0 * np.log10(linear)
    
    @staticmethod
    def apply_volume(signal: np.ndarray, volume: float) -> np.ndarray:
        """
        Apply volume to audio signal.
        
        Args:
            signal: Input audio signal
            volume: Volume multiplier (0.0 to 1.0, or higher for amplification)
            
        Returns:
            Audio signal with volume applied
        """
        return signal * volume
    
    @staticmethod
    def apply_db_volume(signal: np.ndarray, db: float) -> np.ndarray:
        """
        Apply volume in decibels to audio signal.
        
        Args:
            signal: Input audio signal
            db: Volume in decibels (negative for attenuation, positive for amplification)
            
        Returns:
            Audio signal with volume applied
        """
        return signal * VolumeControl.db_to_linear(db)
    
    @staticmethod
    def normalize(signal: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """
        Normalize audio signal to target dB level.
        
        Args:
            signal: Input audio signal
            target_db: Target peak level in dB
            
        Returns:
            Normalized audio signal
        """
        max_amplitude = np.max(np.abs(signal))
        if max_amplitude > 0:
            target_linear = VolumeControl.db_to_linear(target_db)
            return signal / max_amplitude * target_linear
        return signal
    
    @staticmethod
    def fade(
        signal: np.ndarray,
        fade_in: float = 0.0,
        fade_out: float = 0.0,
        sample_rate: int = 44100,
        curve: str = 'linear'
    ) -> np.ndarray:
        """
        Apply fade in/out to audio signal.
        
        Args:
            signal: Input audio signal
            fade_in: Fade in duration in seconds
            fade_out: Fade out duration in seconds
            sample_rate: Sample rate in Hz
            curve: Fade curve type ('linear', 'exponential', 'logarithmic')
            
        Returns:
            Audio signal with fades applied
        """
        output = signal.copy()
        
        # Apply fade in
        if fade_in > 0:
            fade_in_samples = int(fade_in * sample_rate)
            fade_in_samples = min(fade_in_samples, len(output))
            
            if curve == 'linear':
                fade_curve = np.linspace(0, 1, fade_in_samples)
            elif curve == 'exponential':
                fade_curve = np.exp(np.linspace(-5, 0, fade_in_samples))
            elif curve == 'logarithmic':
                fade_curve = np.log10(np.linspace(1, 10, fade_in_samples)) / np.log10(10)
            else:
                fade_curve = np.linspace(0, 1, fade_in_samples)
            
            output[:fade_in_samples] *= fade_curve
        
        # Apply fade out
        if fade_out > 0:
            fade_out_samples = int(fade_out * sample_rate)
            fade_out_samples = min(fade_out_samples, len(output))
            
            if curve == 'linear':
                fade_curve = np.linspace(1, 0, fade_out_samples)
            elif curve == 'exponential':
                fade_curve = np.exp(np.linspace(0, -5, fade_out_samples))
            elif curve == 'logarithmic':
                fade_curve = np.log10(np.linspace(10, 1, fade_out_samples)) / np.log10(10)
            else:
                fade_curve = np.linspace(1, 0, fade_out_samples)
            
            output[-fade_out_samples:] *= fade_curve
        
        return output

