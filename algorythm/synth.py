"""
algorythm.synth - Defines sound sources and timbres

This module provides the core synthesis components including oscillators,
filters, envelopes, and synth presets.
"""

import numpy as np
from typing import Optional, Literal, List


class Oscillator:
    """
    Generates basic waveforms for sound synthesis.
    
    Supports various waveform types including sine, square, saw, and triangle.
    """
    
    def __init__(
        self,
        waveform: Literal['sine', 'square', 'saw', 'triangle', 'noise'] = 'sine',
        frequency: float = 440.0,
        amplitude: float = 1.0,
        phase: float = 0.0
    ):
        """
        Initialize an oscillator.
        
        Args:
            waveform: Type of waveform to generate
            frequency: Frequency in Hz
            amplitude: Amplitude (0.0 to 1.0)
            phase: Initial phase offset in radians
        """
        self.waveform = waveform
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
    
    def generate(self, duration: float, sample_rate: int = 44100) -> np.ndarray:
        """
        Generate waveform samples.
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            NumPy array of audio samples
        """
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        if self.waveform == 'sine':
            signal = np.sin(2 * np.pi * self.frequency * t + self.phase)
        elif self.waveform == 'square':
            signal = np.sign(np.sin(2 * np.pi * self.frequency * t + self.phase))
        elif self.waveform == 'saw':
            signal = 2 * (t * self.frequency - np.floor(0.5 + t * self.frequency))
        elif self.waveform == 'triangle':
            signal = 2 * np.abs(2 * (t * self.frequency - np.floor(0.5 + t * self.frequency))) - 1
        elif self.waveform == 'noise':
            # White noise: random values between -1 and 1
            signal = np.random.uniform(-1, 1, num_samples)
        else:
            raise ValueError(f"Unknown waveform type: {self.waveform}")
        
        return signal * self.amplitude


class Filter:
    """
    Applies frequency filtering to audio signals.
    
    Supports lowpass, highpass, bandpass, and notch filters.
    """
    
    def __init__(
        self,
        filter_type: Literal['lowpass', 'highpass', 'bandpass', 'notch'],
        cutoff: float,
        resonance: float = 0.5
    ):
        """
        Initialize a filter.
        
        Args:
            filter_type: Type of filter to apply
            cutoff: Cutoff frequency in Hz
            resonance: Resonance/Q factor (0.0 to 1.0)
        """
        self.filter_type = filter_type
        self.cutoff = cutoff
        self.resonance = resonance
    
    @classmethod
    def lowpass(cls, cutoff: float, resonance: float = 0.5) -> 'Filter':
        """Create a lowpass filter."""
        return cls('lowpass', cutoff, resonance)
    
    @classmethod
    def highpass(cls, cutoff: float, resonance: float = 0.5) -> 'Filter':
        """Create a highpass filter."""
        return cls('highpass', cutoff, resonance)
    
    @classmethod
    def bandpass(cls, cutoff: float, resonance: float = 0.5) -> 'Filter':
        """Create a bandpass filter."""
        return cls('bandpass', cutoff, resonance)
    
    @classmethod
    def notch(cls, cutoff: float, resonance: float = 0.5) -> 'Filter':
        """Create a notch filter."""
        return cls('notch', cutoff, resonance)
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply filter to an audio signal.
        
        Args:
            signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Filtered audio signal
        """
        # Simplified filter implementation
        # In a real implementation, this would use proper DSP filter design
        return signal


class ADSR:
    """
    Attack, Decay, Sustain, Release envelope generator.
    
    Controls the amplitude of a sound over time.
    """
    
    def __init__(
        self,
        attack: float = 0.1,
        decay: float = 0.1,
        sustain: float = 0.7,
        release: float = 0.3
    ):
        """
        Initialize an ADSR envelope.
        
        Args:
            attack: Attack time in seconds
            decay: Decay time in seconds
            sustain: Sustain level (0.0 to 1.0)
            release: Release time in seconds
        """
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
    
    def generate(self, duration: float, sample_rate: int = 44100) -> np.ndarray:
        """
        Generate ADSR envelope.
        
        Args:
            duration: Total duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            NumPy array of envelope values
        """
        num_samples = int(duration * sample_rate)
        envelope = np.zeros(num_samples)
        
        attack_samples = int(self.attack * sample_rate)
        decay_samples = int(self.decay * sample_rate)
        release_samples = int(self.release * sample_rate)
        
        # Calculate available samples for sustain
        sustain_samples = num_samples - attack_samples - decay_samples - release_samples
        
        # If envelope phases exceed duration, scale them proportionally
        if sustain_samples < 0:
            total_env_samples = attack_samples + decay_samples + release_samples
            if total_env_samples > 0:
                scale = num_samples / total_env_samples
                attack_samples = int(attack_samples * scale)
                decay_samples = int(decay_samples * scale)
                release_samples = num_samples - attack_samples - decay_samples
                sustain_samples = 0
        
        idx = 0
        
        # Attack
        if attack_samples > 0 and idx < num_samples:
            end_idx = min(idx + attack_samples, num_samples)
            envelope[idx:end_idx] = np.linspace(0, 1, end_idx - idx)
            idx = end_idx
        
        # Decay
        if decay_samples > 0 and idx < num_samples:
            end_idx = min(idx + decay_samples, num_samples)
            envelope[idx:end_idx] = np.linspace(1, self.sustain, end_idx - idx)
            idx = end_idx
        
        # Sustain
        if sustain_samples > 0 and idx < num_samples:
            end_idx = min(idx + sustain_samples, num_samples)
            envelope[idx:end_idx] = self.sustain
            idx = end_idx
        
        # Release
        if release_samples > 0 and idx < num_samples:
            end_idx = min(idx + release_samples, num_samples)
            envelope[idx:end_idx] = np.linspace(self.sustain, 0, end_idx - idx)
        
        return envelope


class SynthPresets:
    """
    Collection of pre-configured synth sounds.
    """
    
    @staticmethod
    def warm_pad() -> 'Synth':
        """Create a warm pad synth preset."""
        return Synth(
            waveform='saw',
            filter=Filter.lowpass(cutoff=2000, resonance=0.6),
            envelope=ADSR(attack=1.5, decay=0.5, sustain=0.8, release=2.0)
        )
    
    @staticmethod
    def pluck() -> 'Synth':
        """Create a plucked string synth preset."""
        return Synth(
            waveform='triangle',
            filter=Filter.lowpass(cutoff=3000, resonance=0.3),
            envelope=ADSR(attack=0.01, decay=0.2, sustain=0.3, release=0.5)
        )
    
    @staticmethod
    def bass() -> 'Synth':
        """Create a bass synth preset."""
        return Synth(
            waveform='square',
            filter=Filter.lowpass(cutoff=800, resonance=0.7),
            envelope=ADSR(attack=0.05, decay=0.3, sustain=0.6, release=0.2)
        )


class Synth:
    """
    Main synthesizer class that combines oscillator, filter, and envelope.
    
    This class represents a complete instrument sound.
    """
    
    def __init__(
        self,
        waveform: Literal['sine', 'square', 'saw', 'triangle', 'noise'] = 'sine',
        filter: Optional[Filter] = None,
        envelope: Optional[ADSR] = None
    ):
        """
        Initialize a synthesizer.
        
        Args:
            waveform: Base waveform type
            filter: Optional filter to apply
            envelope: Optional ADSR envelope
        """
        self.waveform = waveform
        self.filter = filter
        self.envelope = envelope or ADSR()
        self.oscillator = Oscillator(waveform=waveform)
    
    def generate_note(
        self,
        frequency: float,
        duration: float,
        sample_rate: int = 44100
    ) -> np.ndarray:
        """
        Generate a single note.
        
        Args:
            frequency: Frequency in Hz
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            NumPy array of audio samples
        """
        # Generate base oscillator signal
        self.oscillator.frequency = frequency
        signal = self.oscillator.generate(duration, sample_rate)
        
        # Apply filter if present
        if self.filter:
            signal = self.filter.apply(signal, sample_rate)
        
        # Apply envelope
        envelope = self.envelope.generate(duration, sample_rate)
        signal = signal * envelope
        
        return signal


class FMSynth:
    """
    FM (Frequency Modulation) Synthesizer for complex harmonic sounds.
    
    Uses one oscillator to modulate the frequency of another, creating
    rich, metallic, and bell-like timbres.
    """
    
    def __init__(
        self,
        carrier_waveform: Literal['sine', 'square', 'saw', 'triangle'] = 'sine',
        modulator_waveform: Literal['sine', 'square', 'saw', 'triangle'] = 'sine',
        modulation_index: float = 2.0,
        mod_freq_ratio: float = 2.0,
        envelope: Optional[ADSR] = None,
        filter: Optional[Filter] = None
    ):
        """
        Initialize an FM synthesizer.
        
        Args:
            carrier_waveform: Carrier oscillator waveform
            modulator_waveform: Modulator oscillator waveform
            modulation_index: FM modulation index (amount of modulation)
            mod_freq_ratio: Modulator frequency as ratio of carrier frequency
            envelope: Optional ADSR envelope
            filter: Optional filter to apply
        """
        self.carrier_waveform = carrier_waveform
        self.modulator_waveform = modulator_waveform
        self.modulation_index = modulation_index
        self.mod_freq_ratio = mod_freq_ratio
        self.envelope = envelope or ADSR(attack=0.01, decay=0.3, sustain=0.5, release=0.5)
        self.filter = filter
    
    def generate_note(
        self,
        frequency: float,
        duration: float,
        sample_rate: int = 44100
    ) -> np.ndarray:
        """
        Generate a single FM synthesized note.
        
        Args:
            frequency: Carrier frequency in Hz
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            NumPy array of audio samples
        """
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Modulator frequency
        mod_freq = frequency * self.mod_freq_ratio
        
        # Generate modulator signal
        if self.modulator_waveform == 'sine':
            modulator = np.sin(2 * np.pi * mod_freq * t)
        elif self.modulator_waveform == 'square':
            modulator = np.sign(np.sin(2 * np.pi * mod_freq * t))
        elif self.modulator_waveform == 'saw':
            modulator = 2 * (t * mod_freq - np.floor(0.5 + t * mod_freq))
        elif self.modulator_waveform == 'triangle':
            modulator = 2 * np.abs(2 * (t * mod_freq - np.floor(0.5 + t * mod_freq))) - 1
        else:
            modulator = np.sin(2 * np.pi * mod_freq * t)
        
        # FM synthesis: carrier frequency modulated by modulator
        modulation = frequency * self.modulation_index * modulator
        
        # Generate carrier with modulated frequency
        if self.carrier_waveform == 'sine':
            signal = np.sin(2 * np.pi * frequency * t + modulation)
        elif self.carrier_waveform == 'square':
            signal = np.sign(np.sin(2 * np.pi * frequency * t + modulation))
        elif self.carrier_waveform == 'saw':
            # For saw, apply modulation differently
            phase = (frequency * t + self.modulation_index * np.sin(2 * np.pi * mod_freq * t) / (2 * np.pi))
            signal = 2 * (phase - np.floor(0.5 + phase))
        elif self.carrier_waveform == 'triangle':
            phase = (frequency * t + self.modulation_index * np.sin(2 * np.pi * mod_freq * t) / (2 * np.pi))
            signal = 2 * np.abs(2 * (phase - np.floor(0.5 + phase))) - 1
        else:
            signal = np.sin(2 * np.pi * frequency * t + modulation)
        
        # Apply filter if present
        if self.filter:
            signal = self.filter.apply(signal, sample_rate)
        
        # Apply envelope
        envelope = self.envelope.generate(duration, sample_rate)
        signal = signal * envelope
        
        return signal


class WavetableSynth:
    """
    Wavetable Synthesizer for morphing between different waveforms.
    
    Stores multiple waveforms and interpolates between them for
    evolving, dynamic timbres.
    """
    
    def __init__(
        self,
        wavetable: Optional[List[np.ndarray]] = None,
        envelope: Optional[ADSR] = None,
        filter: Optional[Filter] = None
    ):
        """
        Initialize a wavetable synthesizer.
        
        Args:
            wavetable: List of waveform arrays (each should be one cycle)
            envelope: Optional ADSR envelope
            filter: Optional filter to apply
        """
        if wavetable is None:
            # Default wavetable: sine -> triangle -> saw -> square
            wavetable = self._create_default_wavetable()
        
        self.wavetable = wavetable
        self.envelope = envelope or ADSR()
        self.filter = filter
    
    def _create_default_wavetable(self, table_size: int = 2048) -> List[np.ndarray]:
        """Create a default wavetable with basic waveforms."""
        t = np.linspace(0, 1, table_size, endpoint=False)
        
        wavetable = [
            np.sin(2 * np.pi * t),  # Sine
            2 * np.abs(2 * (t - np.floor(0.5 + t))) - 1,  # Triangle
            2 * (t - np.floor(0.5 + t)),  # Saw
            np.sign(np.sin(2 * np.pi * t))  # Square
        ]
        
        return wavetable
    
    def generate_note(
        self,
        frequency: float,
        duration: float,
        sample_rate: int = 44100,
        position: float = 0.0,
        morph_automation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate a single wavetable synthesized note.
        
        Args:
            frequency: Frequency in Hz
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            position: Position in wavetable (0.0 to 1.0)
            morph_automation: Optional array for morphing position over time
            
        Returns:
            NumPy array of audio samples
        """
        num_samples = int(duration * sample_rate)
        signal = np.zeros(num_samples)
        
        # Calculate phase increment
        table_size = len(self.wavetable[0])
        phase_increment = frequency * table_size / sample_rate
        phase = 0.0
        
        for i in range(num_samples):
            # Determine wavetable position
            if morph_automation is not None and i < len(morph_automation):
                pos = morph_automation[i]
            else:
                pos = position
            
            # Clamp position to valid range
            pos = np.clip(pos, 0.0, 1.0)
            
            # Calculate which wavetables to interpolate between
            pos_scaled = pos * (len(self.wavetable) - 1)
            table_idx1 = int(np.floor(pos_scaled))
            table_idx2 = min(table_idx1 + 1, len(self.wavetable) - 1)
            blend = pos_scaled - table_idx1
            
            # Get samples from both wavetables
            phase_int = int(phase) % table_size
            sample1 = self.wavetable[table_idx1][phase_int]
            sample2 = self.wavetable[table_idx2][phase_int]
            
            # Interpolate between wavetables
            signal[i] = sample1 * (1 - blend) + sample2 * blend
            
            # Increment phase
            phase += phase_increment
        
        # Apply filter if present
        if self.filter:
            signal = self.filter.apply(signal, sample_rate)
        
        # Apply envelope
        envelope = self.envelope.generate(duration, sample_rate)
        signal = signal * envelope
        
        return signal
    
    @classmethod
    def from_waveforms(
        cls,
        waveforms: List[Literal['sine', 'square', 'saw', 'triangle']],
        envelope: Optional[ADSR] = None,
        filter: Optional[Filter] = None,
        table_size: int = 2048
    ) -> 'WavetableSynth':
        """
        Create a wavetable synth from named waveform types.
        
        Args:
            waveforms: List of waveform names
            envelope: Optional ADSR envelope
            filter: Optional filter to apply
            table_size: Size of each wavetable
            
        Returns:
            WavetableSynth instance
        """
        t = np.linspace(0, 1, table_size, endpoint=False)
        wavetable = []
        
        for waveform in waveforms:
            if waveform == 'sine':
                wave = np.sin(2 * np.pi * t)
            elif waveform == 'square':
                wave = np.sign(np.sin(2 * np.pi * t))
            elif waveform == 'saw':
                wave = 2 * (t - np.floor(0.5 + t))
            elif waveform == 'triangle':
                wave = 2 * np.abs(2 * (t - np.floor(0.5 + t))) - 1
            else:
                wave = np.sin(2 * np.pi * t)
            
            wavetable.append(wave)
        
        return cls(wavetable=wavetable, envelope=envelope, filter=filter)
