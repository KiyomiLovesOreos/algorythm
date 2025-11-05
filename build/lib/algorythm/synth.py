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
    
    @staticmethod
    def lead() -> 'Synth':
        """Create a lead synth preset."""
        return Synth(
            waveform='saw',
            filter=Filter.lowpass(cutoff=4000, resonance=0.8),
            envelope=ADSR(attack=0.01, decay=0.1, sustain=0.9, release=0.3)
        )
    
    @staticmethod
    def organ() -> 'AdditiveeSynth':
        """Create an organ preset."""
        return AdditiveeSynth(
            num_harmonics=8,
            harmonic_amplitudes=[1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05],
            envelope=ADSR(attack=0.01, decay=0.1, sustain=1.0, release=0.1)
        )
    
    @staticmethod
    def bell() -> 'FMSynth':
        """Create a bell preset."""
        return FMSynth(
            carrier_waveform='sine',
            modulator_waveform='sine',
            modulation_index=3.5,
            mod_freq_ratio=3.5,
            envelope=ADSR(attack=0.001, decay=1.0, sustain=0.3, release=2.0)
        )
    
    @staticmethod
    def strings() -> 'PadSynth':
        """Create a strings preset."""
        return PadSynth(
            num_voices=9,
            detune_amount=0.08,
            waveform='saw',
            envelope=ADSR(attack=1.0, decay=0.5, sustain=0.9, release=2.0),
            filter=Filter.lowpass(cutoff=3000, resonance=0.4)
        )
    
    @staticmethod
    def guitar() -> 'PhysicalModelSynth':
        """Create a guitar preset."""
        return PhysicalModelSynth(
            model_type='string',
            damping=0.996,
            brightness=0.7,
            envelope=ADSR(attack=0.001, decay=0.5, sustain=0.5, release=0.5)
        )
    
    @staticmethod
    def drum() -> 'PhysicalModelSynth':
        """Create a drum preset."""
        return PhysicalModelSynth(
            model_type='drum',
            damping=0.98,
            brightness=0.6,
            envelope=ADSR(attack=0.001, decay=0.2, sustain=0.1, release=0.3)
        )
    
    @staticmethod
    def brass() -> 'FMSynth':
        """Create a brass preset."""
        return FMSynth(
            carrier_waveform='saw',
            modulator_waveform='sine',
            modulation_index=1.5,
            mod_freq_ratio=1.0,
            envelope=ADSR(attack=0.1, decay=0.2, sustain=0.8, release=0.3),
            filter=Filter.lowpass(cutoff=3500, resonance=0.5)
        )
    
    @staticmethod
    def piano() -> 'AdditiveeSynth':
        """Create an acoustic piano preset."""
        return AdditiveeSynth(
            num_harmonics=12,
            harmonic_amplitudes=[1.0, 0.4, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03],
            envelope=ADSR(attack=0.002, decay=0.8, sustain=0.2, release=1.5)
        )
    
    @staticmethod
    def electric_piano() -> 'FMSynth':
        """Create an electric piano preset."""
        return FMSynth(
            carrier_waveform='sine',
            modulator_waveform='sine',
            modulation_index=2.0,
            mod_freq_ratio=1.414,
            envelope=ADSR(attack=0.005, decay=0.5, sustain=0.3, release=0.8),
            filter=Filter.lowpass(cutoff=5000, resonance=0.3)
        )
    
    @staticmethod
    def harpsichord() -> 'Synth':
        """Create a harpsichord preset."""
        return Synth(
            waveform='triangle',
            filter=Filter.highpass(cutoff=800, resonance=0.2),
            envelope=ADSR(attack=0.001, decay=0.4, sustain=0.1, release=0.2)
        )
    
    @staticmethod
    def clavinet() -> 'PhysicalModelSynth':
        """Create a clavinet preset."""
        return PhysicalModelSynth(
            model_type='string',
            damping=0.992,
            brightness=0.9,
            envelope=ADSR(attack=0.001, decay=0.15, sustain=0.4, release=0.1)
        )
    
    @staticmethod
    def violin() -> 'PadSynth':
        """Create a violin preset."""
        return PadSynth(
            num_voices=3,
            detune_amount=0.03,
            waveform='saw',
            envelope=ADSR(attack=0.15, decay=0.1, sustain=0.9, release=0.4),
            filter=Filter.bandpass(cutoff=2500, resonance=0.6)
        )
    
    @staticmethod
    def cello() -> 'PadSynth':
        """Create a cello preset."""
        return PadSynth(
            num_voices=3,
            detune_amount=0.02,
            waveform='saw',
            envelope=ADSR(attack=0.2, decay=0.15, sustain=0.85, release=0.6),
            filter=Filter.lowpass(cutoff=1800, resonance=0.7)
        )
    
    @staticmethod
    def flute() -> 'PhysicalModelSynth':
        """Create a flute preset."""
        return PhysicalModelSynth(
            model_type='wind',
            damping=0.997,
            brightness=0.4,
            envelope=ADSR(attack=0.08, decay=0.1, sustain=0.9, release=0.3)
        )
    
    @staticmethod
    def clarinet() -> 'AdditiveeSynth':
        """Create a clarinet preset."""
        return AdditiveeSynth(
            num_harmonics=8,
            harmonic_amplitudes=[1.0, 0.0, 0.5, 0.0, 0.3, 0.0, 0.2, 0.0],
            envelope=ADSR(attack=0.06, decay=0.1, sustain=0.85, release=0.25)
        )
    
    @staticmethod
    def trumpet() -> 'FMSynth':
        """Create a trumpet preset."""
        return FMSynth(
            carrier_waveform='saw',
            modulator_waveform='sine',
            modulation_index=1.8,
            mod_freq_ratio=1.5,
            envelope=ADSR(attack=0.08, decay=0.15, sustain=0.85, release=0.2),
            filter=Filter.lowpass(cutoff=4000, resonance=0.6)
        )
    
    @staticmethod
    def saxophone() -> 'FMSynth':
        """Create a saxophone preset."""
        return FMSynth(
            carrier_waveform='saw',
            modulator_waveform='sine',
            modulation_index=2.2,
            mod_freq_ratio=1.2,
            envelope=ADSR(attack=0.05, decay=0.2, sustain=0.8, release=0.3),
            filter=Filter.bandpass(cutoff=2000, resonance=0.7)
        )
    
    @staticmethod
    def choir() -> 'PadSynth':
        """Create a choir/vocal pad preset."""
        return PadSynth(
            num_voices=11,
            detune_amount=0.15,
            waveform='triangle',
            envelope=ADSR(attack=2.5, decay=1.0, sustain=0.9, release=3.5),
            filter=Filter.bandpass(cutoff=1500, resonance=0.5)
        )
    
    @staticmethod
    def synth_brass() -> 'PadSynth':
        """Create a synth brass preset."""
        return PadSynth(
            num_voices=5,
            detune_amount=0.05,
            waveform='saw',
            envelope=ADSR(attack=0.2, decay=0.3, sustain=0.9, release=0.4),
            filter=Filter.lowpass(cutoff=3000, resonance=0.8)
        )
    
    @staticmethod
    def synth_lead() -> 'Synth':
        """Create a bright synth lead preset."""
        return Synth(
            waveform='square',
            filter=Filter.lowpass(cutoff=5000, resonance=0.9),
            envelope=ADSR(attack=0.005, decay=0.05, sustain=1.0, release=0.2)
        )
    
    @staticmethod
    def acid_bass() -> 'Synth':
        """Create an acid bass preset."""
        return Synth(
            waveform='square',
            filter=Filter.lowpass(cutoff=600, resonance=0.95),
            envelope=ADSR(attack=0.001, decay=0.15, sustain=0.4, release=0.15)
        )
    
    @staticmethod
    def fat_bass() -> 'PadSynth':
        """Create a fat detuned bass preset."""
        return PadSynth(
            num_voices=5,
            detune_amount=0.08,
            waveform='saw',
            envelope=ADSR(attack=0.01, decay=0.3, sustain=0.7, release=0.2),
            filter=Filter.lowpass(cutoff=500, resonance=0.6)
        )
    
    @staticmethod
    def sub_bass() -> 'Synth':
        """Create a sub bass preset."""
        return Synth(
            waveform='sine',
            filter=Filter.lowpass(cutoff=200, resonance=0.3),
            envelope=ADSR(attack=0.01, decay=0.2, sustain=0.8, release=0.3)
        )
    
    @staticmethod
    def marimba() -> 'FMSynth':
        """Create a marimba preset."""
        return FMSynth(
            carrier_waveform='sine',
            modulator_waveform='sine',
            modulation_index=2.5,
            mod_freq_ratio=2.5,
            envelope=ADSR(attack=0.001, decay=0.6, sustain=0.1, release=0.8)
        )
    
    @staticmethod
    def xylophone() -> 'FMSynth':
        """Create a xylophone preset."""
        return FMSynth(
            carrier_waveform='sine',
            modulator_waveform='sine',
            modulation_index=3.0,
            mod_freq_ratio=4.0,
            envelope=ADSR(attack=0.001, decay=0.3, sustain=0.05, release=0.4),
            filter=Filter.highpass(cutoff=800, resonance=0.3)
        )
    
    @staticmethod
    def vibraphone() -> 'FMSynth':
        """Create a vibraphone preset."""
        return FMSynth(
            carrier_waveform='sine',
            modulator_waveform='sine',
            modulation_index=2.0,
            mod_freq_ratio=3.0,
            envelope=ADSR(attack=0.01, decay=1.5, sustain=0.4, release=2.0)
        )
    
    @staticmethod
    def glockenspiel() -> 'FMSynth':
        """Create a glockenspiel preset."""
        return FMSynth(
            carrier_waveform='sine',
            modulator_waveform='sine',
            modulation_index=4.0,
            mod_freq_ratio=5.5,
            envelope=ADSR(attack=0.001, decay=0.8, sustain=0.1, release=1.2),
            filter=Filter.highpass(cutoff=1200, resonance=0.2)
        )
    
    @staticmethod
    def harp() -> 'PhysicalModelSynth':
        """Create a harp preset."""
        return PhysicalModelSynth(
            model_type='string',
            damping=0.998,
            brightness=0.8,
            envelope=ADSR(attack=0.001, decay=0.8, sustain=0.3, release=1.2)
        )
    
    @staticmethod
    def banjo() -> 'PhysicalModelSynth':
        """Create a banjo preset."""
        return PhysicalModelSynth(
            model_type='string',
            damping=0.990,
            brightness=0.95,
            envelope=ADSR(attack=0.001, decay=0.4, sustain=0.2, release=0.3)
        )
    
    @staticmethod
    def sitar() -> 'PhysicalModelSynth':
        """Create a sitar preset."""
        return PhysicalModelSynth(
            model_type='string',
            damping=0.994,
            brightness=0.85,
            envelope=ADSR(attack=0.002, decay=1.2, sustain=0.4, release=1.5)
        )
    
    @staticmethod
    def kalimba() -> 'FMSynth':
        """Create a kalimba/thumb piano preset."""
        return FMSynth(
            carrier_waveform='sine',
            modulator_waveform='sine',
            modulation_index=1.5,
            mod_freq_ratio=2.0,
            envelope=ADSR(attack=0.002, decay=0.5, sustain=0.2, release=0.8),
            filter=Filter.bandpass(cutoff=1500, resonance=0.4)
        )
    
    @staticmethod
    def music_box() -> 'FMSynth':
        """Create a music box preset."""
        return FMSynth(
            carrier_waveform='sine',
            modulator_waveform='sine',
            modulation_index=3.5,
            mod_freq_ratio=4.5,
            envelope=ADSR(attack=0.001, decay=1.0, sustain=0.1, release=1.5),
            filter=Filter.highpass(cutoff=1000, resonance=0.2)
        )
    
    @staticmethod
    def steel_drum() -> 'FMSynth':
        """Create a steel drum preset."""
        return FMSynth(
            carrier_waveform='sine',
            modulator_waveform='sine',
            modulation_index=2.8,
            mod_freq_ratio=3.2,
            envelope=ADSR(attack=0.001, decay=0.6, sustain=0.3, release=1.0),
            filter=Filter.bandpass(cutoff=2000, resonance=0.5)
        )
    
    @staticmethod
    def synth_pluck() -> 'Synth':
        """Create a synth pluck preset."""
        return Synth(
            waveform='saw',
            filter=Filter.lowpass(cutoff=3500, resonance=0.5),
            envelope=ADSR(attack=0.001, decay=0.15, sustain=0.2, release=0.3)
        )
    
    @staticmethod
    def arp_synth() -> 'Synth':
        """Create an arpeggiator-style synth preset."""
        return Synth(
            waveform='square',
            filter=Filter.lowpass(cutoff=4500, resonance=0.7),
            envelope=ADSR(attack=0.002, decay=0.08, sustain=0.5, release=0.15)
        )
    
    @staticmethod
    def ambient_pad() -> 'WavetableSynth':
        """Create an ambient pad preset."""
        return WavetableSynth.from_waveforms(
            waveforms=['sine', 'triangle', 'sine', 'saw'],
            envelope=ADSR(attack=3.0, decay=1.5, sustain=0.85, release=4.0),
            filter=Filter.lowpass(cutoff=2500, resonance=0.3)
        )
    
    @staticmethod
    def noise_sweep() -> 'Synth':
        """Create a noise sweep preset for transitions."""
        return Synth(
            waveform='noise',
            filter=Filter.lowpass(cutoff=5000, resonance=0.8),
            envelope=ADSR(attack=0.5, decay=1.0, sustain=0.3, release=1.5)
        )
    
    @staticmethod
    def kick_drum() -> 'PhysicalModelSynth':
        """Create a kick drum preset."""
        return PhysicalModelSynth(
            model_type='drum',
            damping=0.96,
            brightness=0.3,
            envelope=ADSR(attack=0.001, decay=0.15, sustain=0.05, release=0.2)
        )
    
    @staticmethod
    def snare_drum() -> 'PhysicalModelSynth':
        """Create a snare drum preset."""
        return PhysicalModelSynth(
            model_type='drum',
            damping=0.94,
            brightness=0.8,
            envelope=ADSR(attack=0.001, decay=0.1, sustain=0.05, release=0.15)
        )
    
    @staticmethod
    def hi_hat() -> 'Synth':
        """Create a hi-hat preset."""
        return Synth(
            waveform='noise',
            filter=Filter.highpass(cutoff=5000, resonance=0.3),
            envelope=ADSR(attack=0.001, decay=0.05, sustain=0.02, release=0.08)
        )
    
    @staticmethod
    def tom_drum() -> 'PhysicalModelSynth':
        """Create a tom drum preset."""
        return PhysicalModelSynth(
            model_type='drum',
            damping=0.97,
            brightness=0.5,
            envelope=ADSR(attack=0.001, decay=0.25, sustain=0.1, release=0.3)
        )
    
    @staticmethod
    def cymbal() -> 'Synth':
        """Create a cymbal preset."""
        return Synth(
            waveform='noise',
            filter=Filter.bandpass(cutoff=4000, resonance=0.7),
            envelope=ADSR(attack=0.001, decay=0.5, sustain=0.2, release=1.0)
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


class PhysicalModelSynth:
    """
    Physical modeling synthesizer for realistic instrument sounds.
    
    Simulates the physics of real instruments like strings, drums, and wind instruments.
    """
    
    def __init__(
        self,
        model_type: Literal['string', 'drum', 'wind'] = 'string',
        damping: float = 0.995,
        brightness: float = 0.5,
        envelope: Optional[ADSR] = None
    ):
        self.model_type = model_type
        self.damping = damping
        self.brightness = brightness
        self.envelope = envelope or ADSR(attack=0.001, decay=0.1, sustain=0.7, release=0.3)
    
    def generate_note(
        self,
        frequency: float,
        duration: float,
        sample_rate: int = 44100
    ) -> np.ndarray:
        num_samples = int(duration * sample_rate)
        
        if self.model_type == 'string':
            signal = self._karplus_strong(frequency, num_samples, sample_rate)
        elif self.model_type == 'drum':
            signal = self._drum_model(frequency, num_samples, sample_rate)
        elif self.model_type == 'wind':
            signal = self._wind_model(frequency, num_samples, sample_rate)
        else:
            signal = np.zeros(num_samples)
        
        envelope = self.envelope.generate(duration, sample_rate)
        return signal * envelope
    
    def _karplus_strong(self, frequency: float, num_samples: int, sample_rate: int) -> np.ndarray:
        delay_length = int(sample_rate / frequency)
        buffer = np.random.uniform(-1, 1, delay_length)
        output = np.zeros(num_samples)
        
        for i in range(num_samples):
            output[i] = buffer[0]
            avg = (buffer[0] + buffer[1]) / 2 * self.damping
            buffer = np.append(buffer[1:], avg)
        
        return output
    
    def _drum_model(self, frequency: float, num_samples: int, sample_rate: int) -> np.ndarray:
        t = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False)
        signal = np.random.uniform(-1, 1, num_samples) * np.exp(-t * 10)
        
        resonance = np.sin(2 * np.pi * frequency * t) * np.exp(-t * 5)
        return signal * 0.5 + resonance * 0.5
    
    def _wind_model(self, frequency: float, num_samples: int, sample_rate: int) -> np.ndarray:
        t = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False)
        
        fundamental = np.sin(2 * np.pi * frequency * t)
        harmonics = sum(np.sin(2 * np.pi * frequency * (i + 1) * t) / (i + 1) 
                       for i in range(1, 5))
        
        noise = np.random.uniform(-1, 1, num_samples) * 0.1 * self.brightness
        return (fundamental + harmonics * 0.3 + noise) * 0.5


class AdditiveeSynth:
    """
    Additive synthesizer combining multiple sine wave harmonics.
    """
    
    def __init__(
        self,
        num_harmonics: int = 8,
        harmonic_amplitudes: Optional[List[float]] = None,
        envelope: Optional[ADSR] = None
    ):
        self.num_harmonics = num_harmonics
        self.harmonic_amplitudes = harmonic_amplitudes or [1.0 / (i + 1) for i in range(num_harmonics)]
        self.envelope = envelope or ADSR()
    
    def generate_note(
        self,
        frequency: float,
        duration: float,
        sample_rate: int = 44100
    ) -> np.ndarray:
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        signal = np.zeros(num_samples)
        
        for i, amp in enumerate(self.harmonic_amplitudes[:self.num_harmonics]):
            harmonic_freq = frequency * (i + 1)
            signal += amp * np.sin(2 * np.pi * harmonic_freq * t)
        
        signal = signal / np.max(np.abs(signal))
        envelope = self.envelope.generate(duration, sample_rate)
        return signal * envelope


class PadSynth:
    """
    Creates lush pad sounds with thick, detuned oscillators.
    """
    
    def __init__(
        self,
        num_voices: int = 7,
        detune_amount: float = 0.1,
        waveform: Literal['sine', 'saw', 'square', 'triangle'] = 'saw',
        envelope: Optional[ADSR] = None,
        filter: Optional[Filter] = None
    ):
        self.num_voices = num_voices
        self.detune_amount = detune_amount
        self.waveform = waveform
        self.envelope = envelope or ADSR(attack=2.0, decay=1.0, sustain=0.8, release=3.0)
        self.filter = filter or Filter.lowpass(cutoff=2000, resonance=0.5)
    
    def generate_note(
        self,
        frequency: float,
        duration: float,
        sample_rate: int = 44100
    ) -> np.ndarray:
        num_samples = int(duration * sample_rate)
        signal = np.zeros(num_samples)
        
        for i in range(self.num_voices):
            detune = (i - self.num_voices // 2) * self.detune_amount
            voice_freq = frequency * (1 + detune / 100)
            
            osc = Oscillator(waveform=self.waveform, frequency=voice_freq)
            voice = osc.generate(duration, sample_rate)
            signal += voice
        
        signal = signal / self.num_voices
        
        if self.filter:
            signal = self.filter.apply(signal, sample_rate)
        
        envelope = self.envelope.generate(duration, sample_rate)
        return signal * envelope


class Effect:
    """Base class for audio effects."""
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        return signal


class Reverb(Effect):
    """
    Reverb effect for adding space and depth.
    """
    
    def __init__(
        self,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet_level: float = 0.3
    ):
        self.room_size = room_size
        self.damping = damping
        self.wet_level = wet_level
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        delay_times = [int(sample_rate * 0.037 * self.room_size),
                      int(sample_rate * 0.041 * self.room_size),
                      int(sample_rate * 0.043 * self.room_size),
                      int(sample_rate * 0.047 * self.room_size)]
        
        wet = np.zeros_like(signal)
        for delay_time in delay_times:
            delayed = np.concatenate([np.zeros(delay_time), signal[:-delay_time]])
            wet += delayed * self.damping
        
        wet = wet / len(delay_times)
        return signal * (1 - self.wet_level) + wet * self.wet_level


class Delay(Effect):
    """
    Delay effect for echoes and rhythmic patterns.
    """
    
    def __init__(
        self,
        delay_time: float = 0.5,
        feedback: float = 0.5,
        wet_level: float = 0.5
    ):
        self.delay_time = delay_time
        self.feedback = feedback
        self.wet_level = wet_level
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        delay_samples = int(self.delay_time * sample_rate)
        output = signal.copy()
        
        for i in range(delay_samples, len(signal)):
            output[i] += output[i - delay_samples] * self.feedback
        
        return signal * (1 - self.wet_level) + output * self.wet_level


class Chorus(Effect):
    """
    Chorus effect for thickening sounds.
    """
    
    def __init__(
        self,
        depth: float = 0.003,
        rate: float = 1.5,
        mix: float = 0.5
    ):
        self.depth = depth
        self.rate = rate
        self.mix = mix
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        num_samples = len(signal)
        t = np.arange(num_samples) / sample_rate
        
        lfo = np.sin(2 * np.pi * self.rate * t)
        delay_samples = (self.depth * sample_rate * (1 + lfo)).astype(int)
        
        wet = np.zeros_like(signal)
        for i in range(num_samples):
            delay = min(delay_samples[i], i)
            if delay > 0:
                wet[i] = signal[i - delay]
        
        return signal * (1 - self.mix) + wet * self.mix


class Distortion(Effect):
    """
    Distortion effect for adding harmonics and grit.
    """
    
    def __init__(
        self,
        drive: float = 5.0,
        tone: float = 0.5,
        mix: float = 1.0
    ):
        self.drive = drive
        self.tone = tone
        self.mix = mix
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        wet = np.tanh(signal * self.drive) / np.tanh(self.drive)
        
        if self.tone < 1.0:
            cutoff = 500 + self.tone * 4500
            filter_obj = Filter.lowpass(cutoff=cutoff)
            wet = filter_obj.apply(wet, sample_rate)
        
        return signal * (1 - self.mix) + wet * self.mix


class Compressor(Effect):
    """
    Dynamics compressor for controlling signal levels.
    """
    
    def __init__(
        self,
        threshold: float = -20.0,
        ratio: float = 4.0,
        attack: float = 0.005,
        release: float = 0.1
    ):
        self.threshold = threshold
        self.ratio = ratio
        self.attack = attack
        self.release = release
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        threshold_linear = 10 ** (self.threshold / 20)
        output = signal.copy()
        envelope = 0.0
        
        attack_coef = np.exp(-1 / (self.attack * sample_rate))
        release_coef = np.exp(-1 / (self.release * sample_rate))
        
        for i in range(len(signal)):
            input_level = abs(signal[i])
            
            if input_level > envelope:
                envelope = attack_coef * envelope + (1 - attack_coef) * input_level
            else:
                envelope = release_coef * envelope + (1 - release_coef) * input_level
            
            if envelope > threshold_linear:
                gain_reduction = (envelope / threshold_linear) ** (1 / self.ratio - 1)
                output[i] *= gain_reduction
        
        return output


class Phaser(Effect):
    """
    Phaser effect for sweeping comb-filter sounds.
    """
    
    def __init__(
        self,
        rate: float = 0.5,
        depth: float = 1.0,
        feedback: float = 0.5,
        mix: float = 0.5
    ):
        self.rate = rate
        self.depth = depth
        self.feedback = feedback
        self.mix = mix
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        num_samples = len(signal)
        t = np.arange(num_samples) / sample_rate
        
        lfo = np.sin(2 * np.pi * self.rate * t)
        delay_samples = (2 + self.depth * lfo).astype(int)
        
        wet = np.zeros_like(signal)
        for i in range(num_samples):
            delay = min(delay_samples[i], i)
            if delay > 0:
                wet[i] = signal[i] - wet[i - delay] * self.feedback
        
        return signal * (1 - self.mix) + wet * self.mix


class BitCrusher(Effect):
    """
    Bit crusher for lo-fi digital distortion.
    """
    
    def __init__(
        self,
        bit_depth: int = 8,
        sample_rate_reduction: float = 1.0,
        mix: float = 1.0
    ):
        self.bit_depth = bit_depth
        self.sample_rate_reduction = sample_rate_reduction
        self.mix = mix
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        levels = 2 ** self.bit_depth
        wet = np.round(signal * levels / 2) / (levels / 2)
        
        if self.sample_rate_reduction < 1.0:
            reduced_rate = int(len(signal) * self.sample_rate_reduction)
            indices = np.linspace(0, len(signal) - 1, reduced_rate).astype(int)
            reduced = wet[indices]
            wet = np.repeat(reduced, len(signal) // reduced_rate + 1)[:len(signal)]
        
        return signal * (1 - self.mix) + wet * self.mix
