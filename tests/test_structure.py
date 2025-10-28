"""Tests for algorythm.structure module."""

import numpy as np
import pytest
from algorythm.synth import Synth, SynthPresets
from algorythm.sequence import Motif, Scale
from algorythm.structure import Reverb, Delay, Chorus, Flanger, Distortion, Compression, EffectChain, Track, Composition


class TestEffects:
    """Tests for audio effects."""
    
    def test_reverb_creation(self):
        """Test creating a reverb effect."""
        reverb = Reverb(mix=0.3, room_size=0.5, damping=0.5)
        assert reverb.mix == 0.3
        assert reverb.room_size == 0.5
    
    def test_reverb_apply(self):
        """Test applying reverb to signal."""
        reverb = Reverb(mix=0.3)
        signal = np.random.randn(1000)
        processed = reverb.apply(signal)
        assert len(processed) == len(signal)
    
    def test_delay_creation(self):
        """Test creating a delay effect."""
        delay = Delay(delay_time=0.5, feedback=0.3, mix=0.3)
        assert delay.delay_time == 0.5
        assert delay.feedback == 0.3
    
    def test_flanger_creation(self):
        """Test creating a flanger effect."""
        flanger = Flanger(rate=0.5, depth=0.5, feedback=0.3, mix=0.5)
        assert flanger.rate == 0.5
        assert flanger.depth == 0.5
    
    def test_flanger_apply(self):
        """Test applying flanger to signal."""
        flanger = Flanger(rate=0.5, mix=0.5)
        signal = np.random.randn(1000)
        processed = flanger.apply(signal)
        assert len(processed) == len(signal)
    
    def test_distortion_creation(self):
        """Test creating a distortion effect."""
        dist = Distortion(drive=0.5, tone=0.5, mix=1.0)
        assert dist.drive == 0.5
        assert dist.tone == 0.5
    
    def test_distortion_apply(self):
        """Test applying distortion to signal."""
        dist = Distortion(drive=0.5, mix=1.0)
        signal = np.random.randn(1000) * 0.5
        processed = dist.apply(signal)
        assert len(processed) == len(signal)
    
    def test_compression_creation(self):
        """Test creating a compression effect."""
        comp = Compression(threshold=-20.0, ratio=4.0, attack=0.005, release=0.1)
        assert comp.threshold == -20.0
        assert comp.ratio == 4.0
    
    def test_compression_apply(self):
        """Test applying compression to signal."""
        comp = Compression(threshold=-20.0, ratio=4.0)
        signal = np.random.randn(1000)
        processed = comp.apply(signal)
        assert len(processed) == len(signal)


class TestEffectChain:
    """Tests for EffectChain."""
    
    def test_effect_chain_creation(self):
        """Test creating an effect chain."""
        chain = EffectChain()
        assert len(chain.effects) == 0
    
    def test_effect_chain_add_effect(self):
        """Test adding effects to chain."""
        chain = EffectChain()
        chain.add_effect(Reverb(mix=0.3))
        chain.add_effect(Delay(delay_time=0.5))
        assert len(chain.effects) == 2
    
    def test_effect_chain_apply(self):
        """Test applying effect chain."""
        chain = EffectChain()
        chain.add_effect(Reverb(mix=0.3))
        signal = np.random.randn(1000)
        processed = chain.apply(signal)
        assert len(processed) == len(signal)


class TestTrack:
    """Tests for Track class."""
    
    def test_track_creation(self):
        """Test creating a track."""
        synth = Synth(waveform='sine')
        track = Track('Test Track', synth)
        assert track.name == 'Test Track'
        assert track.synth == synth
    
    def test_track_add_note(self):
        """Test adding a note to track."""
        synth = Synth(waveform='sine')
        track = Track('Test Track', synth)
        track.add_note(frequency=440.0, start_time=0.0, duration=1.0)
        assert len(track.notes) == 1
    
    def test_track_add_motif(self):
        """Test adding a motif to track."""
        synth = Synth(waveform='sine')
        track = Track('Test Track', synth)
        motif = Motif.from_intervals([0, 2, 4], scale=Scale.major('C'))
        track.add_motif(motif, start_time=0.0, tempo=120.0)
        assert len(track.notes) == 3
    
    def test_track_render(self):
        """Test rendering a track."""
        synth = Synth(waveform='sine')
        track = Track('Test Track', synth)
        track.add_note(frequency=440.0, start_time=0.0, duration=1.0)
        audio = track.render(sample_rate=44100)
        assert len(audio) > 0
        assert isinstance(audio, np.ndarray)


class TestComposition:
    """Tests for Composition class."""
    
    def test_composition_creation(self):
        """Test creating a composition."""
        comp = Composition(tempo=120)
        assert comp.tempo == 120
        assert len(comp.tracks) == 0
    
    def test_composition_add_track(self):
        """Test adding a track to composition."""
        comp = Composition(tempo=120)
        synth = Synth(waveform='sine')
        comp.add_track('Track 1', synth)
        assert len(comp.tracks) == 1
        assert 'Track 1' in comp.tracks
    
    def test_composition_repeat_motif(self):
        """Test repeating a motif in composition."""
        comp = Composition(tempo=120)
        synth = Synth(waveform='sine')
        motif = Motif.from_intervals([0, 2, 4], scale=Scale.major('C'))
        comp.add_track('Track 1', synth).repeat_motif(motif, bars=2)
        # Should have notes from 2 bars
        assert len(comp.tracks['Track 1'].notes) == 6  # 3 notes * 2 bars
    
    def test_composition_transpose(self):
        """Test transposing a track."""
        comp = Composition(tempo=120)
        synth = Synth(waveform='sine')
        motif = Motif.from_intervals([0], scale=Scale.major('C'))
        comp.add_track('Track 1', synth).repeat_motif(motif, bars=1)
        original_freq = comp.tracks['Track 1'].notes[0]['frequency']
        comp.transpose(semitones=12)
        new_freq = comp.tracks['Track 1'].notes[0]['frequency']
        # Transposing 12 semitones (1 octave) should double frequency
        assert abs(new_freq / original_freq - 2.0) < 0.01
    
    def test_composition_render(self):
        """Test rendering a composition."""
        comp = Composition(tempo=120)
        synth = Synth(waveform='sine')
        motif = Motif.from_intervals([0, 2, 4], scale=Scale.major('C'))
        comp.add_track('Track 1', synth).repeat_motif(motif, bars=1)
        audio = comp.render()
        assert len(audio) > 0
        assert isinstance(audio, np.ndarray)
