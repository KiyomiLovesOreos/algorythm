"""
Demo of new features in Algorythm v0.2.0

This example demonstrates:
- FM Synthesis
- Wavetable Synthesis
- New effects (EQ, Phaser, Tremolo, Bitcrusher)
- Interactive playback
"""

import numpy as np
from algorythm.synth import FMSynth, WavetableSynth, Synth, ADSR, Filter
from algorythm.sequence import Scale, Motif
from algorythm.structure import (
    Composition, EQ, Phaser, Tremolo, Bitcrusher,
    Reverb, Delay, Distortion
)

print("=" * 60)
print("Algorythm v0.2.0 - New Features Demo")
print("=" * 60)

# Example 1: FM Synthesis
print("\n1. FM Synthesis - Bell-like tones")
fm_synth = FMSynth(
    carrier_waveform='sine',
    modulator_waveform='sine',
    modulation_index=3.0,
    mod_freq_ratio=2.5,
    envelope=ADSR(attack=0.01, decay=0.3, sustain=0.2, release=0.5)
)

scale = Scale.major('C', octave=5)
fm_notes = []
for i in range(5):
    note = fm_synth.generate_note(
        frequency=scale.get_frequency(i * 2),
        duration=0.5
    )
    fm_notes.append(note)

fm_audio = np.concatenate(fm_notes)
print(f"Generated FM audio: {len(fm_audio)} samples ({len(fm_audio)/44100:.2f}s)")

# Example 2: Wavetable Synthesis with Morphing
print("\n2. Wavetable Synthesis - Morphing waveforms")
wt_synth = WavetableSynth.from_waveforms(
    waveforms=['sine', 'triangle', 'saw', 'square'],
    envelope=ADSR(attack=0.05, decay=0.2, sustain=0.6, release=0.3)
)

scale = Scale.minor('A', octave=3)
wt_notes = []
for i in range(8):
    duration = 0.4
    # Create morphing automation that sweeps through wavetable
    morph = np.linspace(0, 1, int(duration * 44100))
    
    note = wt_synth.generate_note(
        frequency=scale.get_frequency(i),
        duration=duration,
        morph_automation=morph
    )
    wt_notes.append(note)

wt_audio = np.concatenate(wt_notes)
print(f"Generated wavetable audio: {len(wt_audio)} samples ({len(wt_audio)/44100:.2f}s)")

# Example 3: New Effects
print("\n3. Testing new effects:")

# Create a simple synth for testing effects
test_synth = Synth(
    waveform='saw',
    filter=Filter.lowpass(cutoff=2000, resonance=0.5),
    envelope=ADSR(attack=0.1, decay=0.2, sustain=0.6, release=0.3)
)

test_notes = []
for i in range(4):
    note = test_synth.generate_note(
        frequency=scale.get_frequency(i),
        duration=0.5
    )
    test_notes.append(note)
test_audio = np.concatenate(test_notes)

# EQ Effect
print("   - EQ: Boosting highs, cutting lows")
eq = EQ(low_gain=0.5, mid_gain=1.0, high_gain=1.5)
eq_audio = eq.apply(test_audio)

# Phaser Effect
print("   - Phaser: Sweeping notch filter")
phaser = Phaser(rate=0.5, depth=0.7, stages=4)
phaser_audio = phaser.apply(test_audio)

# Tremolo Effect
print("   - Tremolo: Amplitude modulation")
tremolo = Tremolo(rate=5.0, depth=0.6)
tremolo_audio = tremolo.apply(test_audio)

# Bitcrusher Effect
print("   - Bitcrusher: Lo-fi digital distortion")
bitcrusher = Bitcrusher(bit_depth=4, sample_rate_reduction=4.0)
bitcrusher_audio = bitcrusher.apply(test_audio)

# Example 4: Complex Effects Chain
print("\n4. Complex Effects Chain")
comp = Composition(tempo=120)

bass_synth = Synth(
    waveform='square',
    filter=Filter.lowpass(cutoff=800, resonance=0.7),
    envelope=ADSR(attack=0.05, decay=0.3, sustain=0.6, release=0.2)
)

bass_scale = Scale.pentatonic_minor('E', octave=2)
bass_motif = Motif.from_intervals([0, 2, 3, 5, 7], scale=bass_scale)

comp.add_track('bass', bass_synth) \
    .repeat_motif(bass_motif, bars=2) \
    .add_fx(Distortion(drive=0.3, tone=0.7)) \
    .add_fx(EQ(low_gain=1.5, mid_gain=0.8, high_gain=0.6)) \
    .add_fx(Phaser(rate=0.3, depth=0.5)) \
    .add_fx(Delay(delay_time=0.375, feedback=0.3, mix=0.2)) \
    .add_fx(Reverb(mix=0.15))

chain_audio = comp.render()
print(f"Generated chain audio: {len(chain_audio)} samples ({len(chain_audio)/44100:.2f}s)")

# Example 5: Save outputs
print("\n5. Saving audio files...")
from algorythm.export import Exporter

exporter = Exporter()
exporter.export(fm_audio, 'fm_synthesis_demo.wav')
print("   Saved: fm_synthesis_demo.wav")

exporter.export(wt_audio, 'wavetable_demo.wav')
print("   Saved: wavetable_demo.wav")

exporter.export(bitcrusher_audio, 'bitcrusher_demo.wav')
print("   Saved: bitcrusher_demo.wav")

exporter.export(chain_audio, 'effects_chain_demo.wav')
print("   Saved: effects_chain_demo.wav")

# Example 6: Interactive Playback (if pyaudio is available)
print("\n6. Interactive Playback:")
try:
    from algorythm.playback import AudioPlayer
    
    print("   pyaudio is available - testing playback")
    player = AudioPlayer()
    
    print("   Playing FM synthesis example...")
    player.play(fm_audio, blocking=True)
    
    print("   Playback complete!")
    player.close()
    
except ImportError:
    print("   pyaudio not available (install with: pip install pyaudio)")
    print("   Skipping playback demo")

print("\n" + "=" * 60)
print("Demo complete! Check the generated .wav files")
print("\n" + "To try the Live Coding GUI, run:")
print("   python -m algorythm.live_gui")
print("   or: algorythm-live")
print("=" * 60)
