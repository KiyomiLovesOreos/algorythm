from algorythm.synth import SynthPresets, Synth, Oscillator, ADSR, Filter
from algorythm.sequence import Scale, Motif, Arpeggiator
from algorythm.structure import Composition, Reverb, Delay, VolumeControl
import numpy as np

# 1. Initialize Composition
comp = Composition(tempo=120) # A bit faster tempo for a melodic piece

# 2. Create a lead synth using a pluck preset
lead_synth = SynthPresets.pluck()
# Customize the pluck a bit for a brighter sound
lead_synth.envelope.release = 0.3 # Shorter release
lead_synth.filter.cutoff = 4000 # Brighter filter

# 3. Define a scale and a melodic motif
c_major_scale = Scale.major('C', octave=5) # Higher octave for lead
melody_intervals = [0, 2, 4, 7, 9, 12, 11, 7] # C, D, E, G, A, C(octave up), B, G
melody_durations = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # Eighth notes

melody_motif = Motif.from_intervals(melody_intervals, scale=c_major_scale, durations=melody_durations)

# 4. Arpeggiate the motif
arpeggiator = Arpeggiator(pattern='up', octaves=1)
arpeggiated_melody = arpeggiator.arpeggiate(melody_motif)

# 5. Add the melodic track to the composition
comp.add_track('Lead Melody', lead_synth) \
    .repeat_motif(arpeggiated_melody, bars=8) # Repeat for 8 bars
    
# 6. Add a bassline for harmonic support
bass_synth = SynthPresets.bass()
c_minor_scale_bass = Scale.minor('C', octave=2)
bass_motif = Motif.from_intervals([0, 0, 0, 0, 0, 0, 0, 0], scale=c_minor_scale_bass, durations=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) # Whole notes

comp.add_track('Bassline', bass_synth) \
    .repeat_motif(bass_motif, bars=8) \
    .add_fx(VolumeControl(volume=0.7)) # Lower bass volume

# 7. Add global effects
comp.add_fx(Reverb(mix=0.4, room_size=0.6, damping=0.5))
comp.add_fx(Delay(delay_time=0.25, feedback=0.3, mix=0.3))

# 8. Render the composition
print("Rendering melodic arpeggio...")
audio_output = comp.render(file_path='test_example_1_melody.wav', quality='high')
print(f"Rendered melodic arpeggio to test_example_1_melody.wav with {len(audio_output)} samples.")
print(f"Duration: {len(audio_output) / comp.sample_rate:.2f} seconds.")
