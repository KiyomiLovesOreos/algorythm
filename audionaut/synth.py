"""
audionaut.synth - Defines sound sources and timbres

This module provides the core synthesis components including oscillators,
filters, envelopes, and synth presets.
"""

import numpy as np
from typing import Optional, Literal


class Oscillator:
    """
    Generates basic waveforms for sound synthesis.
    
    Supports various waveform types including sine, square, saw, and triangle.
    """
    
    def __init__(
        self,
        waveform: Literal['sine', 'square', 'saw', 'triangle'] = 'sine',
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
        sustain_samples = num_samples - attack_samples - decay_samples - release_samples
        
        # Ensure non-negative sample counts
        if sustain_samples < 0:
            sustain_samples = 0
        
        idx = 0
        
        # Attack
        if attack_samples > 0:
            envelope[idx:idx + attack_samples] = np.linspace(0, 1, attack_samples)
            idx += attack_samples
        
        # Decay
        if decay_samples > 0:
            envelope[idx:idx + decay_samples] = np.linspace(1, self.sustain, decay_samples)
            idx += decay_samples
        
        # Sustain
        if sustain_samples > 0:
            envelope[idx:idx + sustain_samples] = self.sustain
            idx += sustain_samples
        
        # Release
        if release_samples > 0 and idx < num_samples:
            envelope[idx:idx + release_samples] = np.linspace(
                self.sustain, 0, min(release_samples, num_samples - idx)
            )
        
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
        waveform: Literal['sine', 'square', 'saw', 'triangle'] = 'sine',
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
