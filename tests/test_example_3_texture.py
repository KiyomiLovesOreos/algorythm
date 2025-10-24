from algorythm.synth import WavetableSynth, ADSR, Filter
from algorythm.sequence import Scale, Motif
from algorythm.structure import Composition, Reverb, Delay, VolumeControl
import numpy as np

# 1. Initialize Composition
comp = Composition(tempo=60) # Slower tempo for ambient texture

# 2. Create a Wavetable Synth for evolving texture
# We'll morph between sine, triangle, saw, and square waves
wavetable_synth = WavetableSynth.from_waveforms(
    waveforms=['sine', 'triangle', 'saw', 'square'],
    envelope=ADSR(attack=2.0, decay=1.0, sustain=0.7, release=3.0), # Long attack and release
    filter=Filter.lowpass(cutoff=3000, resonance=0.4) # Gentle lowpass filter
)

# 3. Define a sustained chord motif
c_minor_scale = Scale.minor('C', octave=3)
# Play a C minor chord (C, Eb, G) sustained for a long duration
# Intervals: 0 (C), 3 (Eb), 7 (G) relative to C minor scale
texture_motif = Motif.from_intervals([0, 3, 7], scale=c_minor_scale, durations=[16.0, 16.0, 16.0]) # Very long notes

# 4. Add the wavetable track to the composition
# We'll repeat this motif for 2 bars, which at 60 BPM means 2 * 4 * 1 = 8 beats. 
# Since our motif notes are 16 beats long, this will effectively play the chord once.
# To make it 30 seconds, we need 30 seconds / (60 beats/min / 60 sec/min) = 30 beats.
# 30 beats / 4 beats/bar = 7.5 bars. Let's make it 8 bars for simplicity.
comp.add_track('Wavetable Texture', wavetable_synth) \
    .repeat_motif(texture_motif, bars=8) \
    .add_fx(VolumeControl(volume=0.6)) # Slightly lower volume

# 5. Add global effects for spaciousness
comp.add_fx(Reverb(mix=0.7, room_size=0.9, damping=0.6))
comp.add_fx(Delay(delay_time=0.5, feedback=0.5, mix=0.4))

# 6. Render the composition
print("Rendering granular texture...")
audio_output = comp.render(file_path='test_example_3_texture.wav', quality='high')
print(f"Rendered granular texture to test_example_3_texture.wav with {len(audio_output)} samples.")
print(f"Duration: {len(audio_output) / comp.sample_rate:.2f} seconds.")
