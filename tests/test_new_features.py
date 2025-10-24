"""
Test new instruments and effects
"""

import pytest
import numpy as np
from algorythm import *
from algorythm.effects import *


def test_physical_model_synth():
    """Test PhysicalModelSynth generates audio"""
    string = PhysicalModelSynth(model_type='string')
    audio = string.generate_note(440.0, 0.5, 44100)
    assert len(audio) == 22050
    assert audio.dtype == np.float64
    
    drum = PhysicalModelSynth(model_type='drum')
    audio = drum.generate_note(100.0, 0.5, 44100)
    assert len(audio) == 22050
    
    wind = PhysicalModelSynth(model_type='wind')
    audio = wind.generate_note(440.0, 0.5, 44100)
    assert len(audio) == 22050


def test_additive_synth():
    """Test AdditiveeSynth generates audio"""
    synth = AdditiveeSynth(num_harmonics=4)
    audio = synth.generate_note(440.0, 0.5, 44100)
    assert len(audio) == 22050
    assert audio.dtype == np.float64


def test_pad_synth():
    """Test PadSynth generates audio"""
    pad = PadSynth(num_voices=3, detune_amount=0.1)
    audio = pad.generate_note(440.0, 0.5, 44100)
    assert len(audio) == 22050
    assert audio.dtype == np.float64


def test_new_presets():
    """Test new synth presets load correctly"""
    lead = SynthPresets.lead()
    assert lead is not None
    
    organ = SynthPresets.organ()
    assert organ is not None
    
    strings = SynthPresets.strings()
    assert strings is not None
    
    guitar = SynthPresets.guitar()
    assert guitar is not None
    
    drum = SynthPresets.drum()
    assert drum is not None
    
    brass = SynthPresets.brass()
    assert brass is not None


def test_reverb_effect():
    """Test Reverb effect"""
    test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    reverb = ReverbFX(room_size=0.5, wet_level=0.3)
    processed = reverb.apply(test_signal, 44100)
    assert len(processed) == len(test_signal)
    assert processed.dtype == test_signal.dtype


def test_delay_effect():
    """Test Delay effect"""
    test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    delay = DelayFX(delay_time=0.5, feedback=0.5)
    processed = delay.apply(test_signal, 44100)
    assert len(processed) == len(test_signal)


def test_chorus_effect():
    """Test Chorus effect"""
    test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    chorus = ChorusFX(depth=0.003, rate=1.5)
    processed = chorus.apply(test_signal, 44100)
    assert len(processed) == len(test_signal)


def test_flanger_effect():
    """Test Flanger effect"""
    test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    flanger = FlangerFX(depth=0.002, rate=0.25)
    processed = flanger.apply(test_signal, 44100)
    assert len(processed) == len(test_signal)


def test_phaser_effect():
    """Test Phaser effect"""
    test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    phaser = PhaserFX(rate=0.5, depth=1.0)
    processed = phaser.apply(test_signal, 44100)
    assert len(processed) == len(test_signal)


def test_distortion_effect():
    """Test Distortion effect"""
    test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    distortion = DistortionFX(drive=5.0)
    processed = distortion.apply(test_signal, 44100)
    assert len(processed) == len(test_signal)


def test_overdrive_effect():
    """Test Overdrive effect"""
    test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    overdrive = Overdrive(drive=2.0)
    processed = overdrive.apply(test_signal, 44100)
    assert len(processed) == len(test_signal)


def test_fuzz_effect():
    """Test Fuzz effect"""
    test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    fuzz = Fuzz(gain=10.0)
    processed = fuzz.apply(test_signal, 44100)
    assert len(processed) == len(test_signal)


def test_compressor_effect():
    """Test Compressor effect"""
    test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    compressor = Compressor(threshold=-20.0, ratio=4.0)
    processed = compressor.apply(test_signal, 44100)
    assert len(processed) == len(test_signal)


def test_limiter_effect():
    """Test Limiter effect"""
    test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    limiter = Limiter(threshold=-1.0)
    processed = limiter.apply(test_signal, 44100)
    assert len(processed) == len(test_signal)


def test_gate_effect():
    """Test Gate effect"""
    test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    gate = Gate(threshold=-40.0)
    processed = gate.apply(test_signal, 44100)
    assert len(processed) == len(test_signal)


def test_tremolo_effect():
    """Test Tremolo effect"""
    test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    tremolo = TremoloFX(rate=5.0, depth=0.5)
    processed = tremolo.apply(test_signal, 44100)
    assert len(processed) == len(test_signal)


def test_vibrato_effect():
    """Test Vibrato effect"""
    test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    vibrato = Vibrato(rate=5.0, depth=0.002)
    processed = vibrato.apply(test_signal, 44100)
    assert len(processed) == len(test_signal)


def test_bitcrusher_effect():
    """Test BitCrusher effect"""
    test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    bitcrusher = BitCrusherFX(bit_depth=8)
    processed = bitcrusher.apply(test_signal, 44100)
    assert len(processed) == len(test_signal)


def test_autopan_effect():
    """Test AutoPan effect"""
    test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    autopan = AutoPan(rate=1.0, depth=1.0)
    processed = autopan.apply(test_signal, 44100)
    assert len(processed) == len(test_signal)


def test_ring_modulator_effect():
    """Test RingModulator effect"""
    test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    ring_mod = RingModulator(carrier_freq=440.0, mix=0.5)
    processed = ring_mod.apply(test_signal, 44100)
    assert len(processed) == len(test_signal)


def test_effect_chain():
    """Test EffectChain"""
    test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    
    chain = FXChain()
    chain.add_effect(DistortionFX(drive=2.0))
    chain.add_effect(ChorusFX(mix=0.3))
    chain.add_effect(ReverbFX(room_size=0.6))
    
    processed = chain.apply(test_signal, 44100)
    assert len(processed) == len(test_signal)
    
    # Test clear
    chain.clear()
    assert len(chain.effects) == 0


def test_presets_generate_audio():
    """Test that all presets can generate audio"""
    presets = [
        SynthPresets.warm_pad(),
        SynthPresets.pluck(),
        SynthPresets.bass(),
        SynthPresets.lead(),
        SynthPresets.organ(),
        SynthPresets.bell(),
        SynthPresets.strings(),
        SynthPresets.guitar(),
        SynthPresets.drum(),
        SynthPresets.brass(),
    ]
    
    for preset in presets:
        audio = preset.generate_note(440.0, 0.5, 44100)
        assert len(audio) > 0
        assert audio.dtype == np.float64


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
