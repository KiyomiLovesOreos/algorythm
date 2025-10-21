from algorythm.synth import SynthPresets
from algorythm.sequence import Scale, Motif
from algorythm.structure import Composition, Reverb, Delay
import numpy as np

# 1. Initialize Composition
# Let's use a slower tempo for an ambient feel
comp = Composition(tempo=60)

# 2. Create a warm pad synth
# We'll use the built-in warm_pad preset for a rich, sustained sound
warm_pad_synth = SynthPresets.warm_pad()

# 3. Add a track for the pad
# We'll add a single, long note to create a drone-like effect
# Let's use a C2 note (MIDI 36) for a deep, resonant base
# The duration will be 8 bars, which at 60 BPM is 32 seconds (8 bars * 4 beats/bar * 1 second/beat)
# We'll create a simple motif with a single note at the root of a C major scale
c_major_scale = Scale.major('C', octave=2)
drone_motif = Motif.from_intervals([0], scale=c_major_scale, durations=[32.0]) # 32 beats duration

comp.add_track('Ambient Pad', warm_pad_synth) \
    .repeat_motif(drone_motif, bars=1) # Repeat once for the full duration

# 4. Add Reverb for spaciousness
comp.add_fx(Reverb(mix=0.6, room_size=0.8, damping=0.7))

# 5. Add a melodic layer with an arpeggiated pluck synth
pluck_synth = SynthPresets.pluck()

# Use a higher octave for the melody
c_major_scale_melody = Scale.major('C', octave=4)
melody_motif = Motif.from_intervals([0, 2, 4, 5, 7, 9, 11], scale=c_major_scale_melody)

# Arpeggiate the motif
from algorythm.sequence import Arpeggiator
arpeggiator = Arpeggiator(pattern='up-down', octaves=2)
arpeggiated_motif = arpeggiator.arpeggiate(melody_motif)

comp.add_track('Arpeggiated Pluck', pluck_synth) \
    .repeat_motif(arpeggiated_motif, bars=4) \
    .add_fx(Reverb(mix=0.2)) \
    .add_fx(Delay(delay_time=0.3, feedback=0.4, mix=0.5))

# 6. Render the composition
# This will create an audio file and return the numpy array of audio samples
print("Rendering ambient pad...")
audio_output = comp.render(file_path='ambient_pad.wav', quality='high')
print(f"Rendered ambient pad to ambient_pad.wav with {len(audio_output)} samples.")
print(f"Duration: {len(audio_output) / comp.sample_rate:.2f} seconds.")
