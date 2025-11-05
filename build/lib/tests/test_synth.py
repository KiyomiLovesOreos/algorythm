"""Tests for algorythm.synth module."""

import numpy as np
import pytest
from algorythm.synth import Oscillator, Filter, ADSR, Synth, SynthPresets


class TestOscillator:
    """Tests for Oscillator class."""
    
    def test_oscillator_creation(self):
        """Test creating an oscillator."""
        osc = Oscillator(waveform='sine', frequency=440.0)
        assert osc.waveform == 'sine'
        assert osc.frequency == 440.0
    
    def test_oscillator_generate(self):
        """Test generating audio from oscillator."""
        osc = Oscillator(waveform='sine', frequency=440.0)
        signal = osc.generate(duration=1.0, sample_rate=44100)
        assert len(signal) == 44100
        assert isinstance(signal, np.ndarray)
    
    def test_oscillator_waveforms(self):
        """Test different waveform types."""
        waveforms = ['sine', 'square', 'saw', 'triangle', 'noise']
        for waveform in waveforms:
            osc = Oscillator(waveform=waveform, frequency=440.0)
            signal = osc.generate(duration=0.1)
            assert len(signal) > 0


class TestFilter:
    """Tests for Filter class."""
    
    def test_filter_creation(self):
        """Test creating a filter."""
        filt = Filter.lowpass(cutoff=1000, resonance=0.5)
        assert filt.filter_type == 'lowpass'
        assert filt.cutoff == 1000
        assert filt.resonance == 0.5
    
    def test_filter_types(self):
        """Test different filter types."""
        filters = [
            Filter.lowpass(1000),
            Filter.highpass(1000),
            Filter.bandpass(1000),
            Filter.notch(1000),
        ]
        for filt in filters:
            assert filt.cutoff == 1000


class TestADSR:
    """Tests for ADSR envelope."""
    
    def test_adsr_creation(self):
        """Test creating an ADSR envelope."""
        adsr = ADSR(attack=0.1, decay=0.2, sustain=0.7, release=0.3)
        assert adsr.attack == 0.1
        assert adsr.decay == 0.2
        assert adsr.sustain == 0.7
        assert adsr.release == 0.3
    
    def test_adsr_generate(self):
        """Test generating an envelope."""
        adsr = ADSR(attack=0.1, decay=0.1, sustain=0.7, release=0.2)
        envelope = adsr.generate(duration=1.0, sample_rate=44100)
        assert len(envelope) == 44100
        assert np.min(envelope) >= 0
        assert np.max(envelope) <= 1.0
    
    def test_adsr_short_duration(self):
        """Test ADSR with duration shorter than envelope phases."""
        adsr = ADSR(attack=1.0, decay=1.0, sustain=0.7, release=1.0)
        envelope = adsr.generate(duration=0.5, sample_rate=44100)
        assert len(envelope) == int(0.5 * 44100)


class TestSynth:
    """Tests for Synth class."""
    
    def test_synth_creation(self):
        """Test creating a synth."""
        synth = Synth(waveform='sine')
        assert synth.waveform == 'sine'
    
    def test_synth_with_filter(self):
        """Test synth with filter."""
        synth = Synth(
            waveform='saw',
            filter=Filter.lowpass(cutoff=2000, resonance=0.6)
        )
        assert synth.filter is not None
    
    def test_synth_generate_note(self):
        """Test generating a note."""
        synth = Synth(waveform='sine')
        note = synth.generate_note(frequency=440.0, duration=1.0)
        assert len(note) == 44100
        assert isinstance(note, np.ndarray)


class TestSynthPresets:
    """Tests for SynthPresets."""
    
    def test_warm_pad_preset(self):
        """Test warm pad preset."""
        synth = SynthPresets.warm_pad()
        assert isinstance(synth, Synth)
        assert synth.waveform == 'saw'
    
    def test_pluck_preset(self):
        """Test pluck preset."""
        synth = SynthPresets.pluck()
        assert isinstance(synth, Synth)
        assert synth.waveform == 'triangle'
    
    def test_bass_preset(self):
        """Test bass preset."""
        synth = SynthPresets.bass()
        assert isinstance(synth, Synth)
        assert synth.waveform == 'square'
